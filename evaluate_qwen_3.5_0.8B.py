"""
evaluate_qwen3.5_local.py

Evaluate Qwen3.5 zero-shot vs. QLoRA fine-tuned performance on VQA-RAD.

Metadata (question_type, answer_type, image_organ, etc.) is merged into the
HuggingFace test split by matching (question, answer) pairs against the
official VQA_RAD_Dataset_Public.json file stored locally.

The script computes:
  - Overall accuracy
  - Closed-ended vs. Open-ended accuracy
  - Accuracy per organ system (HEAD, CHEST, ABDOMEN)
  - Accuracy per question type (PRES, ABN, MODALITY, etc.)
  - Detailed per-sample results saved to JSON.
"""

import os
import json
import torch
from functools import partial
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel
from datasets import load_dataset
from tqdm import tqdm
import re
from evaluate import load
bertscore = load("bertscore")

# ============================================================================
# Configuration
# ============================================================================
MODEL_ID = "Qwen/Qwen3.5-0.8B"                 # Base model identifier
LORA_PATH = "./results/Qwen3.5_0.8b/best_model" # Path to QLoRA adapter
TEST_SAMPLES = 50                              # Number of test samples to evaluate  Train: 1793 | Val: 225 | Test: 226
OUTPUT_DIR = "./results/Qwen3.5_0.8b"           # Directory for evaluation outputs

# Path to the official VQA-RAD metadata JSON (local file)
LOCAL_JSON_PATH = r"D:\1MA2Semester\ML&BDP\Medical_vqa_project\VQA_RAD_Dataset_Public.json"

# ============================================================================
# Metadata loading and merging
# ============================================================================
def load_local_metadata(json_path):
    """
    Load the full VQA-RAD metadata from a local JSON file.
    """
    print(f"Loading metadata from local file: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    print(f"✓ Loaded {len(metadata)} metadata entries")
    return metadata

def build_metadata_map(full_metadata):
    """
    Build a mapping from (question, answer) to metadata fields.

    Keys are lowercased/stripped for robust matching.
    Returns a dict: (q, a) -> {image_organ, answer_type, question_type, image_id}
    """
    meta_map = {}
    for item in full_metadata:
        # Normalize question and answer strings
        q = item["question"].strip().lower()
        # Ensure answer is string (some entries may be numeric)
        a = str(item["answer"]).strip().lower()
        key = (q, a)

        # Keep only the fields needed for evaluation
        meta_map[key] = {
            "image_organ": item.get("image_organ", "unknown"),
            "answer_type": item.get("answer_type", "OPEN"),
            "question_type": item.get("question_type", "other"),
            "image_id": item.get("image_name", ""),
        }
    return meta_map

def add_metadata(example, meta_map):
    """
    Enrich a HuggingFace dataset sample with metadata fields.

    Matches by (question, answer) and fills in:
        image_organ, answer_type, question_type, image_id
    """
    q = example["question"].strip().lower()
    a = str(example["answer"]).strip().lower()
    meta = meta_map.get((q, a), {})

    example["image_organ"] = meta.get("image_organ", "unknown")
    example["answer_type"] = meta.get("answer_type", "OPEN")
    example["question_type"] = meta.get("question_type", "other")
    example["image_id"] = meta.get("image_id", "")
    return example

# ============================================================================
# Data preparation
# ============================================================================
print("Loading test split from HuggingFace dataset...")
vqa_rad = load_dataset("flaviagiammarino/vqa-rad")
test_split = vqa_rad["test"].shuffle(seed=42)
test_samples = test_split.select(range(min(TEST_SAMPLES, len(test_split))))

# Load local metadata and build lookup map
full_metadata = load_local_metadata(LOCAL_JSON_PATH)
meta_map = build_metadata_map(full_metadata)

# Merge metadata into test samples
test_samples = test_samples.map(partial(add_metadata, meta_map=meta_map))
print(f"✓ Test samples loaded and enriched: {len(test_samples)} samples\n")

# ============================================================================
# Helper functions for decoding and evaluation
# ============================================================================
def build_inputs(processor, image, question, answer_type="OPEN"):
    """
    Build model inputs from an image and a question.

    For closed-ended questions (answer_type == "CLOSED"), a prompt is
    appended to encourage a short yes/no answer.
    """
    if answer_type == "CLOSED":
        # Force model to output only yes/no
        question_guided = f"{question}\nAnswer with only 'yes' or 'no':"
    else:
        question_guided = question

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question_guided},
            ],
        }
    ]

    # Apply chat template and tokenize
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
        min_pixels=144 * 28 * 28,
        max_pixels=256 * 28 * 28,
    )

    # Move tensors to GPU if available
    inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    return inputs

def clean_think_tags(text):
    """
    Remove <think>...</think> blocks and surrounding whitespace.
    """
    cleaned = re.sub(r'<think>.*?</think>\n*', '', text, flags=re.DOTALL)
    return cleaned.strip()

def generate_answer(model, processor, inputs, answer_type="OPEN"):
    """
    Generate an answer from the model.

    For closed-ended questions, we use a very low temperature and short max tokens
    to force a concise response.
    """
    if answer_type == "CLOSED":
        max_new_tokens = 5
        temperature = 0.1
    else:
        max_new_tokens = 100
        temperature = 0.7

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=False,          # deterministic for reproducibility
        )

    predicted = processor.decode(outputs[0], skip_special_tokens=True)

    # Extract the assistant part if chat template is used
    if "assistant" in predicted:
        predicted = predicted.split("assistant")[-1].strip()
    # Remove trailing punctuation/periods
    predicted = predicted.strip().rstrip('.').rstrip('。')
    predicted = clean_think_tags(predicted)
    return predicted

def closed_accuracy(pred, gt):
    """
    Evaluate closed-ended (yes/no) questions.

    Strategy:
      1. Extract a yes/no token from the prediction.
      2. If extraction fails, fall back to checking if ground truth is a substring.
      3. Otherwise, return False.
    """
    pred_lower = pred.lower().strip()
    gt_lower = gt.lower().strip()

    # Look for an explicit yes/no token
    for token in pred_lower.split():
        if token in ["yes", "no"]:
            return token == gt_lower

    # Fallback: simple substring match
    if gt_lower in pred_lower:
        return True

    return False

def open_accuracy(pred, gt, threshold=0.8):
    """
    Evaluate open-ended questions using BERTScore F1.
    Returns True if F1 >= threshold, False otherwise.
    """
    if not pred.strip() or not gt.strip():
        return False
    results = bertscore.compute(
        predictions=[pred], references=[gt], lang="en"
    )
    return results["f1"][0] >= threshold

# ============================================================================
# Main evaluation loop
# ============================================================================
def evaluate_model(model, processor, samples, model_name="Model"):
    """
    Run evaluation over all test samples.

    Returns a dict with:
        - overall_accuracy
        - closed_accuracy / open_accuracy
        - per_organ_accuracy (image_organ: HEAD, CHEST, ABDOMEN)
        - per_question_type_accuracy
        - details list (per-sample predictions and correctness)
    """
    details = []
    stats = {
        "closed": {"correct": 0, "total": 0},
        "open":   {"correct": 0, "total": 0},
        "organ":  {},                # per image_organ
        "question_type": {},         # per question_type
    }

    for i, sample in enumerate(tqdm(samples, desc=f"Evaluating {model_name}")):
        image = sample["image"]
        question = sample["question"]
        ground_truth = sample["answer"]
        answer_type = sample["answer_type"]
        image_organ = sample["image_organ"]
        question_type = sample["question_type"]

        # Build inputs and generate
        inputs = build_inputs(processor, image, question, answer_type)
        predicted = generate_answer(model, processor, inputs, answer_type)

        # Determine correctness
        if answer_type == "CLOSED":
            is_correct = closed_accuracy(predicted, ground_truth)
        else:
            is_correct = open_accuracy(predicted, ground_truth)

        # Update global closed/open counters
        if answer_type == "CLOSED":
            stats["closed"]["total"] += 1
            if is_correct:
                stats["closed"]["correct"] += 1
        else:
            stats["open"]["total"] += 1
            if is_correct:
                stats["open"]["correct"] += 1

        # Update per-organ counters
        if image_organ not in stats["organ"]:
            stats["organ"][image_organ] = {"correct": 0, "total": 0}
        stats["organ"][image_organ]["total"] += 1
        if is_correct:
            stats["organ"][image_organ]["correct"] += 1

        # Update per-question-type counters
        if question_type not in stats["question_type"]:
            stats["question_type"][question_type] = {"correct": 0, "total": 0}
        stats["question_type"][question_type]["total"] += 1
        if is_correct:
            stats["question_type"][question_type]["correct"] += 1

        # Record detailed per-sample information
        details.append({
            "index": i,
            "image_id": sample.get("image_id", ""),
            "question": question,
            "ground_truth": ground_truth,
            "predicted": predicted,
            "answer_type": answer_type,
            "image_organ": image_organ,
            "question_type": question_type,
            "is_correct": is_correct,
        })

    # ----------------------------------------------------------------------
    # Compute final accuracies
    # ----------------------------------------------------------------------
    overall_total = stats["closed"]["total"] + stats["open"]["total"]
    overall_correct = stats["closed"]["correct"] + stats["open"]["correct"]
    overall_accuracy = overall_correct / overall_total if overall_total > 0 else 0.0

    closed_acc = stats["closed"]["correct"] / stats["closed"]["total"] if stats["closed"]["total"] > 0 else 0.0
    open_acc   = stats["open"]["correct"] / stats["open"]["total"] if stats["open"]["total"] > 0 else 0.0

    organ_acc = {
        org: cnt["correct"] / cnt["total"]
        for org, cnt in stats["organ"].items() if cnt["total"] > 0
    }
    question_type_acc = {
        qt: cnt["correct"] / cnt["total"]
        for qt, cnt in stats["question_type"].items() if cnt["total"] > 0
    }

    # Print summary to console
    print(f"\n✓ {model_name} Overall Accuracy: {overall_accuracy*100:.1f}% ({overall_correct}/{overall_total})")
    print(f"   Closed-ended Accuracy: {closed_acc*100:.1f}%")
    print(f"   Open-ended Accuracy:   {open_acc*100:.1f}%")
    print("   Per-organ Accuracy:")
    for org, acc in organ_acc.items():
        print(f"      {org}: {acc*100:.1f}%")
    print("   Per-question-type Accuracy:")
    for qt, acc in question_type_acc.items():
        print(f"      {qt}: {acc*100:.1f}%")

    return {
        "model_name": model_name,
        "overall_accuracy": overall_accuracy,
        "closed_accuracy": closed_acc,
        "open_accuracy": open_acc,
        "per_organ_accuracy": organ_acc,
        "per_question_type_accuracy": question_type_acc,
        "details": details,
    }

# ============================================================================
# 1. Zero-shot Evaluation (base model)
# ============================================================================
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

del base_model
torch.cuda.empty_cache()

# ============================================================================
# 2. QLoRA Fine-tuned Evaluation
# ============================================================================
print("=" * 80)
print("2. Fine-tuned Evaluation (QLoRA)")
print("=" * 80)

# Reload base model and inject LoRA adapter
base_model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
finetuned_model = PeftModel.from_pretrained(base_model, LORA_PATH)
finetuned_model.eval()

finetuned_results = evaluate_model(finetuned_model, processor, test_samples, model_name="QLoRA Fine-tuned")

# ============================================================================
# 3. Save results to JSON
# ============================================================================
os.makedirs(OUTPUT_DIR, exist_ok=True)
output_path = os.path.join(OUTPUT_DIR, "eval_results_0.8B.json")
combined = {
    "zero_shot": zero_shot_results,
    "finetuned": finetuned_results,
}
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(combined, f, indent=2, ensure_ascii=False)
print(f"\nEvaluation results saved to: {output_path}")

# ============================================================================
# 4. Final comparison summary
# ============================================================================
print("=" * 80)
print("Final Comparison")
print("=" * 80)
print(f"Zero-shot Accuracy:         {zero_shot_results['overall_accuracy']*100:.1f}%")
print(f"QLoRA Fine-tuned Accuracy:  {finetuned_results['overall_accuracy']*100:.1f}%")
print(f"Improvement:                {(finetuned_results['overall_accuracy'] - zero_shot_results['overall_accuracy'])*100:+.1f}%")
print("=" * 80)