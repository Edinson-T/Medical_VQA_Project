"""
evaluate_qwen3.5_0.8B.py

Evaluate Qwen3.5-0.8B zero-shot vs. fine-tuned.

Open-ended metric: Token F1 (SQuAD-style word overlap).
  - Consistent with the 0.8B evaluation script → results directly comparable.
  - No BERTScore: avoids left/right semantic confusion on short medical answers.
  - Reports both binary accuracy (F1 ≥ 0.5) and mean F1 (continuous).

Closed-ended metric: exact yes/no match + sklearn classification report.
"""

import os
import re
import json
import torch
from functools import partial
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel
from datasets import load_dataset
from tqdm import tqdm
from sklearn.metrics import classification_report
from evaluate import load
bertscore = load("bertscore")

# ============================================================================
# Configuration
# ============================================================================
MODEL_ID   = "Qwen/Qwen3.5-0.8B"
LORA_PATH  = "./checkpoints/Qwen3.5_0.8B/epoch_3"
TEST_SAMPLES = 226
OUTPUT_DIR = "./results/Qwen3.5_0.8B"
LOCAL_JSON_PATH = r"D:\1MA2Semester\ML&BDP\Medical_vqa_project\VQA_RAD_Dataset_Public.json"

BERTSCORE_THRESHOLD = 0.85   # BERTScore F1 >= threshold → correct
MIN_QT_SAMPLES    = 5     # skip question-type categories with fewer samples

# ============================================================================
# Metadata helpers
# ============================================================================
def load_local_metadata(json_path):
    print(f"Loading metadata: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    print(f"✓ {len(metadata)} entries")
    return metadata

def build_metadata_map(full_metadata):
    meta_map = {}
    for item in full_metadata:
        q = item["question"].strip().lower()
        a = str(item["answer"]).strip().lower()
        meta_map[(q, a)] = {
            "image_organ":   item.get("image_organ",   "unknown"),
            "answer_type":   item.get("answer_type",   "OPEN"),
            "question_type": item.get("question_type", "other"),
            "image_id":      item.get("image_name",    ""),
        }
    return meta_map

def add_metadata(example, meta_map):
    q    = example["question"].strip().lower()
    a    = str(example["answer"]).strip().lower()
    meta = meta_map.get((q, a), {})
    example["image_organ"]   = meta.get("image_organ",   "unknown")
    example["answer_type"]   = meta.get("answer_type",   "OPEN")
    example["question_type"] = meta.get("question_type", "other")
    example["image_id"]      = meta.get("image_id",      "")
    return example

# ============================================================================
# Dataset preparation
# ============================================================================
print("Loading dataset…")
vqa_rad = load_dataset("flaviagiammarino/vqa-rad")

raw_test     = vqa_rad["test"].shuffle(seed=42)
split_idx    = len(raw_test) // 2
test_split   = raw_test.select(range(split_idx, len(raw_test)))
test_samples = test_split.select(range(min(TEST_SAMPLES, len(test_split))))

full_metadata = load_local_metadata(LOCAL_JSON_PATH)
meta_map      = build_metadata_map(full_metadata)
test_samples  = test_samples.map(partial(add_metadata, meta_map=meta_map))
print(f"✓ {len(test_samples)} test samples ready\n")

# ============================================================================
# Inference helpers
# ============================================================================
def build_inputs(processor, image, question, answer_type="OPEN"):
    if answer_type == "CLOSED":
        question_guided = f"{question}\nAnswer with only 'yes' or 'no':"
    else:
        question_guided = f"{question}\nAnswer briefly."

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text",  "text":  question_guided},
        ],
    }]
    text   = processor.apply_chat_template(messages, tokenize=False,
                                           add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt",
                       max_pixels=144 * 28 * 28)
    return {k: v.cuda() if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()}

def clean_text(text):
    """
    Strip <think> blocks, markdown bold, and verbose preambles.
    """
    text = re.sub(r'<think>.*?</think>\n*', '', text, flags=re.DOTALL)
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(
        r'^(based on [^,]+,\s*|the (answer|finding|image) (is|shows)\s*[:–-]?\s*)',
        '', text, flags=re.IGNORECASE,
    )
    text = re.split(r'[.\n]', text)[0]
    return text.strip().rstrip('.').rstrip('。')

def generate_answer(model, processor, inputs, answer_type="OPEN"):
    input_len   = inputs["input_ids"].shape[1]
    max_new     = 10 if answer_type == "CLOSED" else 40
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens = max_new,
            do_sample      = False,   # deterministic; temperature is irrelevant here
        )
    generated = processor.decode(outputs[0][input_len:], skip_special_tokens=True)
    return clean_text(generated)

# ============================================================================
# Metric helpers
# ============================================================================
def closed_accuracy(pred, gt):
    p, g = pred.lower().strip(), gt.lower().strip()
    for token in p.split():
        if token in ("yes", "no"):
            return token == g
    if p.startswith("yes"):
        return "yes" == g
    if p.startswith("no"):
        return "no" == g
    return g in p

def extract_closed_pred(pred):
    p = pred.lower().strip()
    for token in p.split():
        if token in ("yes", "no"):
            return token
    if p.startswith("yes"):
        return "yes"
    if p.startswith("no"):
        return "no"
    return "unknown"

def open_accuracy_bertscore(pred, gt, threshold=BERTSCORE_THRESHOLD):
    """
    Direction-aware BERTScore evaluation for open-ended medical VQA.

    1. Check direction/organ keywords — if pred and gt disagree on
       left/right, normal/abnormal, etc., return False directly.
    2. Otherwise use standard BERTScore.
    """
    if not pred.strip() or not gt.strip():
        return False

    # ── Step 1: Direction / location / medical opposites check ──────────────
    direction_keywords = [
        "left", "right", "bilateral", "unilateral",
        "anterior", "posterior", "superior", "inferior",
        "upper", "lower", "proximal", "distal",
    ]

    medical_opposites = [
        ("yes", "no"),
        ("normal", "abnormal"),
        ("enlarged", "normal"),
        ("enlarged", "small"),
    ]

    pred_lower = pred.lower()
    gt_lower = gt.lower()

    # Directional keyword check
    for word in direction_keywords:
        in_pred = word in pred_lower.split()
        in_gt = word in gt_lower.split()
        # If one has it and the other doesn't, and the other side isn't also present
        if in_pred != in_gt:
            return False

    # Medical opposites check
    for a, b in medical_opposites:
        a_in_pred = a in pred_lower.split()
        b_in_pred = b in pred_lower.split()
        a_in_gt = a in gt_lower.split()
        b_in_gt = b in gt_lower.split()
        # If pred says A and gt says B, without the other being present
        if (a_in_pred and b_in_gt) and not (b_in_pred or a_in_gt):
            return False
        if (b_in_pred and a_in_gt) and not (a_in_pred or b_in_gt):
            return False

    # ── Step 2: Standard BERTScore ──────────────────────────────────────────
    results = bertscore.compute(predictions=[pred], references=[gt], lang="en")
    return results["f1"][0] >= threshold

# ============================================================================
# Evaluation loop
# ============================================================================
def evaluate_model(model, processor, samples, model_name="Model"):
    details = []
    stats   = {
        "closed": {"correct": 0, "total": 0},
        "open":   {"correct": 0, "total": 0},
        "organ":  {},
        "question_type": {},
    }
    y_true_closed  = []
    y_pred_closed  = []
    open_f1_scores = []

    for i, sample in enumerate(tqdm(samples, desc=f"Evaluating {model_name}")):
        image         = sample["image"]
        question      = sample["question"]
        ground_truth  = sample["answer"]
        answer_type   = sample["answer_type"]
        image_organ   = sample["image_organ"]
        question_type = sample["question_type"]

        inputs    = build_inputs(processor, image, question, answer_type)
        predicted = generate_answer(model, processor, inputs, answer_type)

        if answer_type == "CLOSED":
            raw_f1 = None
            is_correct = closed_accuracy(predicted, ground_truth)
            y_true_closed.append(ground_truth.lower().strip())
            y_pred_closed.append(extract_closed_pred(predicted))
        else:
            # Compute BERTScore first, then apply direction-aware check
            is_correct = open_accuracy_bertscore(predicted, ground_truth, threshold=BERTSCORE_THRESHOLD)
            # For the "Open F1 mean" statistic, compute plain BERTScore
            # (no direction check, to show raw semantic similarity)
            raw_f1 = bertscore.compute(
                predictions=[predicted], references=[ground_truth], lang="en"
            )["f1"][0]
            open_f1_scores.append(raw_f1)

        bucket = "closed" if answer_type == "CLOSED" else "open"
        stats[bucket]["total"] += 1
        if is_correct:
            stats[bucket]["correct"] += 1

        stats["organ"].setdefault(image_organ, {"correct": 0, "total": 0})
        stats["organ"][image_organ]["total"] += 1
        if is_correct:
            stats["organ"][image_organ]["correct"] += 1

        stats["question_type"].setdefault(question_type, {"correct": 0, "total": 0})
        stats["question_type"][question_type]["total"] += 1
        if is_correct:
            stats["question_type"][question_type]["correct"] += 1

        details.append({
            "index":         i,
            "image_id":      sample.get("image_id", ""),
            "question":      question,
            "ground_truth":  ground_truth,
            "predicted":     predicted,
            "answer_type":   answer_type,
            "image_organ":   image_organ,
            "question_type": question_type,
            "is_correct":    is_correct,
            "bertscore_f1":  round(raw_f1, 4) if raw_f1 is not None else None,
        })

    # ── Aggregate ────────────────────────────────────────────────────────────
    total_correct = stats["closed"]["correct"] + stats["open"]["correct"]
    total_samples = stats["closed"]["total"]   + stats["open"]["total"]
    overall_acc   = total_correct / total_samples if total_samples > 0 else 0.0
    closed_acc    = (stats["closed"]["correct"] / stats["closed"]["total"]
                     if stats["closed"]["total"] > 0 else 0.0)
    open_acc      = (stats["open"]["correct"] / stats["open"]["total"]
                     if stats["open"]["total"] > 0 else 0.0)
    open_f1_mean  = (sum(open_f1_scores) / len(open_f1_scores)
                     if open_f1_scores else 0.0)

    organ_acc = {
        org: cnt["correct"] / cnt["total"]
        for org, cnt in stats["organ"].items() if cnt["total"] > 0
    }
    # Filter out question-type categories with too few samples
    question_type_acc = {
        qt: cnt["correct"] / cnt["total"]
        for qt, cnt in stats["question_type"].items()
        if cnt["total"] >= MIN_QT_SAMPLES
    }

    # ── Closed F1 report ─────────────────────────────────────────────────────
    valid = [(t, p) for t, p in zip(y_true_closed, y_pred_closed)
             if p != "unknown"]
    if valid:
        yt, yp       = zip(*valid)
        report_dict  = classification_report(
            yt, yp, labels=["yes", "no"], target_names=["Yes", "No"],
            output_dict=True, zero_division=0,
        )
        closed_report   = classification_report(
            yt, yp, labels=["yes", "no"], target_names=["Yes", "No"],
            output_dict=False, zero_division=0,
        )
        closed_macro_f1 = report_dict["macro avg"]["f1-score"]
        recall_no       = report_dict["No"]["recall"]
    else:
        closed_report   = "N/A"
        closed_macro_f1 = 0.0
        recall_no       = 0.0

    # ── Console output ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  {model_name}")
    print(f"{'='*60}")
    print(f"  Overall  Accuracy : {overall_acc*100:.1f}%  ({total_correct}/{total_samples})")
    print(f"  Closed   Accuracy : {closed_acc*100:.1f}%")
    print(f"  Open     Accuracy : {open_acc*100:.1f}%  (BERTScore ≥ {BERTSCORE_THRESHOLD})")
    print(f"  Open     F1 (avg) : {open_f1_mean:.4f}")
    print(f"\n  [Closed-ended Classification Report]")
    print(closed_report)
    print(f"  Per-organ Accuracy:")
    for org, acc in organ_acc.items():
        print(f"    {org}: {acc*100:.1f}%")
    print(f"  Per-question-type Accuracy (≥{MIN_QT_SAMPLES} samples):")
    for qt, acc in sorted(question_type_acc.items(), key=lambda x: -x[1]):
        n = stats["question_type"][qt]["total"]
        print(f"    {qt:<15}: {acc*100:.1f}%  (n={n})")

    return {
        "model_name":                 model_name,
        "overall_accuracy":           overall_acc,
        "closed_accuracy":            closed_acc,
        "closed_macro_f1":            closed_macro_f1,
        "closed_recall_no":           recall_no,
        "open_accuracy":              open_acc,
        "open_f1_mean":               open_f1_mean,
        "open_f1_threshold":          BERTSCORE_THRESHOLD,
        "per_organ_accuracy":         organ_acc,
        "per_question_type_accuracy": question_type_acc,
        "details":                    details,
    }

# ============================================================================
# 1. Zero-shot
# ============================================================================
print("=" * 60)
print("1. Zero-shot  (base Qwen3.5-0.8B)")
print("=" * 60)

processor  = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
base_model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
)
base_model.eval()

zero_shot_results = evaluate_model(base_model, processor, test_samples,
                                   model_name="Zero-shot")
del base_model
torch.cuda.empty_cache()

# ============================================================================
# 2. Fine-tuned
# ============================================================================
print("\n" + "=" * 60)
print("2. Fine-tuned  (Unsloth LoRA)")
print("=" * 60)

base_model      = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
)
finetuned_model = PeftModel.from_pretrained(base_model, LORA_PATH)
finetuned_model.eval()

finetuned_results = evaluate_model(finetuned_model, processor, test_samples,
                                   model_name="Unsloth LoRA Fine-tuned")

# ============================================================================
# 3. Save JSON
# ============================================================================
os.makedirs(OUTPUT_DIR, exist_ok=True)
output_path = os.path.join(OUTPUT_DIR, "eval_results_0.8B.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump({"zero_shot": zero_shot_results, "finetuned": finetuned_results},
              f, indent=2, ensure_ascii=False)
print(f"\n✓ Results saved → {output_path}")

# ============================================================================
# 4. Comparison table
# ============================================================================
zs = zero_shot_results
ft = finetuned_results

print("\n" + "=" * 60)
print("Final Comparison")
print("=" * 60)
print(f"{'Metric':<30} {'Zero-shot':>10} {'Fine-tuned':>12} {'Δ':>8}")
print("-" * 60)
rows = [
    ("Overall Accuracy",     zs["overall_accuracy"],   ft["overall_accuracy"]),
    ("Closed Accuracy",      zs["closed_accuracy"],    ft["closed_accuracy"]),
    ("Closed Macro F1",      zs["closed_macro_f1"],    ft["closed_macro_f1"]),
    ("Recall(No)",           zs["closed_recall_no"],   ft["closed_recall_no"]),
    ("Open Accuracy (≥0.85)", zs["open_accuracy"],      ft["open_accuracy"]),
    ("Open F1 mean",         zs["open_f1_mean"],       ft["open_f1_mean"]),
]
for label, z, f in rows:
    delta = f - z
    sign  = "+" if delta >= 0 else ""
    print(f"  {label:<28} {z*100:>9.1f}%  {f*100:>10.1f}%  {sign}{delta*100:>6.1f}%")
print("=" * 60)