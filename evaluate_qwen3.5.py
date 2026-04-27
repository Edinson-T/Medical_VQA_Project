"""
evaluate_qwen3.5.py - Evaluate Qwen3.5 zero-shot vs. QLoRA fine-tuned performance
                      with detailed per-sample results saved to JSON.
"""

import os
import json
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel
from datasets import load_dataset
from tqdm import tqdm

# ==================== Configuration ====================
MODEL_ID = "Qwen/Qwen3.5-0.8B"
LORA_PATH = "./results/Qwen3.5_0.8b/best_model"   # Path to LoRA adapter
TEST_SAMPLES = 50                                 # Use all or a subset of test set
OUTPUT_DIR = "./results/Qwen3.5_0.8b"             # Where to save evaluation results

# ==================== Load data ====================
print("Loading test data...")
vqa_rad = load_dataset("flaviagiammarino/vqa-rad")
test_split = vqa_rad["test"].shuffle(seed=42)
test_samples = test_split.select(range(min(TEST_SAMPLES, len(test_split))))
print(f"✓ Loaded {len(test_samples)} test samples\n")

# ==================== Helper functions ====================
def build_inputs(processor, image, question):
    """Construct model inputs from an image and a question."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
        min_pixels=144 * 28 * 28,
        max_pixels=256 * 28 * 28,
    )
    # Move to GPU if available
    inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    return inputs


def generate_answer(model, processor, inputs):
    """Generate an answer using the model (greedy decoding for reproducibility)."""
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=False,            # deterministic for evaluation
        )
    predicted = processor.decode(outputs[0], skip_special_tokens=True)
    # Extract the assistant part if the template is present
    if "assistant" in predicted:
        predicted = predicted.split("assistant")[-1].strip()
    return predicted


def closed_accuracy(pred, gt):
    """Closed‑ended questions: check if ground truth appears in the prediction."""
    return gt.lower() in pred.lower()


def open_accuracy(pred, gt):
    """Open‑ended questions: at least half of the ground‑truth words overlap."""
    pred_words = set(pred.lower().split())
    gt_words = set(gt.lower().split())
    if len(gt_words) == 0:
        return False
    overlap = len(pred_words & gt_words) / len(gt_words)
    return overlap >= 0.5


def evaluate_model(model, processor, samples, model_name="Model"):
    """
    Evaluate a model on all test samples.
    Returns a dictionary with overall accuracy, per‑type accuracy,
    per‑modality accuracy, and a list of per‑sample results.
    """
    details = []                     # store per‑sample information
    stats = {
        "closed": {"correct": 0, "total": 0},
        "open":   {"correct": 0, "total": 0},
        "modality": {},               # will hold {modality: {"correct": cnt, "total": cnt}}
    }

    for i, sample in enumerate(tqdm(samples, desc=f"Evaluating {model_name}")):
        image = sample["image"]
        question = sample["question"]
        ground_truth = sample["answer"]
        phrase_type = sample.get("phrase_type", "freeform")   # "para" = closed, "freeform" = open
        modality = sample.get("modality", "unknown")          # X-ray, CT, MRI, etc.

        # Forward pass
        inputs = build_inputs(processor, image, question)
        predicted = generate_answer(model, processor, inputs)

        # Determine correctness
        if phrase_type == "para":
            is_correct = closed_accuracy(predicted, ground_truth)
        else:
            is_correct = open_accuracy(predicted, ground_truth)

        # Update per‑type counters
        if phrase_type == "para":
            stats["closed"]["total"] += 1
            if is_correct:
                stats["closed"]["correct"] += 1
        else:
            stats["open"]["total"] += 1
            if is_correct:
                stats["open"]["correct"] += 1

        # Update per‑modality counters
        if modality not in stats["modality"]:
            stats["modality"][modality] = {"correct": 0, "total": 0}
        stats["modality"][modality]["total"] += 1
        if is_correct:
            stats["modality"][modality]["correct"] += 1

        # Record detail
        details.append({
            "index": i,
            "image_id": sample.get("image_id", None),
            "question": question,
            "ground_truth": ground_truth,
            "predicted": predicted,
            "phrase_type": phrase_type,
            "modality": modality,
            "is_correct": is_correct,
        })

    # Compute accuracies
    overall_total = stats["closed"]["total"] + stats["open"]["total"]
    overall_correct = stats["closed"]["correct"] + stats["open"]["correct"]
    overall_accuracy = overall_correct / overall_total if overall_total > 0 else 0.0

    closed_accuracy_val = stats["closed"]["correct"] / stats["closed"]["total"] if stats["closed"]["total"] > 0 else 0.0
    open_accuracy_val = stats["open"]["correct"] / stats["open"]["total"] if stats["open"]["total"] > 0 else 0.0

    modality_accuracy = {}
    for mod, cnt in stats["modality"].items():
        modality_accuracy[mod] = cnt["correct"] / cnt["total"] if cnt["total"] > 0 else 0.0

    # Print summary to console
    print(f"\n✓ {model_name} Overall Accuracy: {overall_accuracy*100:.1f}% ({overall_correct}/{overall_total})")
    print(f"   Closed‑ended: {closed_accuracy_val*100:.1f}%")
    print(f"   Open‑ended:   {open_accuracy_val*100:.1f}%")
    for mod, acc in modality_accuracy.items():
        print(f"   Modality {mod}: {acc*100:.1f}%")

    # Return structured results
    results = {
        "model_name": model_name,
        "overall_accuracy": overall_accuracy,
        "closed_accuracy": closed_accuracy_val,
        "open_accuracy": open_accuracy_val,
        "per_modality_accuracy": modality_accuracy,
        "details": details,
    }
    return results


# ==================== 1. Zero‑shot evaluation ====================
print("=" * 80)
print("1. Zero-shot Evaluation (base Qwen3.5-0.8B)")
print("=" * 80)

processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
base_model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
base_model.eval()

zero_shot_results = evaluate_model(base_model, processor, test_samples, model_name="Zero-shot")

# Free memory
del base_model
torch.cuda.empty_cache()

# ==================== 2. Fine‑tuned (QLoRA) evaluation ====================
print("=" * 80)
print("2. Fine-tuned Evaluation (QLoRA)")
print("=" * 80)

# Load base model again and inject LoRA
base_model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
finetuned_model = PeftModel.from_pretrained(base_model, LORA_PATH)
finetuned_model.eval()

finetuned_results = evaluate_model(finetuned_model, processor, test_samples, model_name="QLoRA Fine-tuned")

# ==================== 3. Save results to JSON ====================
os.makedirs(OUTPUT_DIR, exist_ok=True)
output_path = os.path.join(OUTPUT_DIR, "eval_results.json")
combined = {
    "zero_shot": zero_shot_results,
    "finetuned": finetuned_results,
}
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(combined, f, indent=2, ensure_ascii=False)
print(f"\nEvaluation results saved to: {output_path}")

# ==================== 4. Final comparison ====================
print("=" * 80)
print("📊 Final Comparison")
print("=" * 80)
print(f"Zero-shot Accuracy:       {zero_shot_results['overall_accuracy']*100:.1f}%")
print(f"QLoRA Fine-tuned Accuracy: {finetuned_results['overall_accuracy']*100:.1f}%")
print(f"Improvement:              {(finetuned_results['overall_accuracy'] - zero_shot_results['overall_accuracy'])*100:+.1f}%")
print("=" * 80)