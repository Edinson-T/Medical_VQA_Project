"""
evaluate.py - 评估 Qwen3.5-0.8B 的零样本 vs QLoRA 微调性能
"""

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel
from datasets import load_dataset
from tqdm import tqdm

# ==================== 配置 ====================
MODEL_ID = "Qwen/Qwen3.5-0.8B"
LORA_PATH = "./results/Qwen3.5_0.8b/best_model"   # LoRA 适配器路径
TEST_SAMPLES = 100                                # 测试样本数，设大一点但不要太慢

# ==================== 加载数据 ====================
print("Loading test data...")
vqa_rad = load_dataset("flaviagiammarino/vqa-rad")
test_split = vqa_rad["test"].shuffle(seed=42)
test_samples = test_split.select(range(min(TEST_SAMPLES, len(test_split))))
print(f"✓ Loaded {len(test_samples)} test samples\n")

# ==================== 工具函数 ====================
def build_inputs(processor, image, question):
    """构造模型输入"""
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
    # 移动到 GPU（如果存在）
    inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    return inputs

def generate_answer(model, processor, inputs):
    """生成回答"""
    with torch.no_grad():
        # Qwen3.5 需要传递 mm_token_type_ids（如果有）
    
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=False,   # 评估时用贪心解码
        )
    predicted = processor.decode(outputs[0], skip_special_tokens=True)
    # 提取 assistant 回复部分（去掉 prompt）
    if "assistant" in predicted:
        predicted = predicted.split("assistant")[-1].strip()
    return predicted

def closed_accuracy(pred, gt):
    """封闭式问题：直接字符串包含判断"""
    return gt.lower() in pred.lower()

def open_accuracy(pred, gt):
    """开放式问题：宽松匹配（至少一半词重叠）"""
    pred_words = set(pred.lower().split())
    gt_words = set(gt.lower().split())
    if len(gt_words) == 0:
        return False
    overlap = len(pred_words & gt_words) / len(gt_words)
    return overlap >= 0.5

def evaluate_model(model, processor, samples, model_name="Model"):
    """评估一个模型，返回准确率"""
    correct = 0
    total = 0
    for i, sample in enumerate(tqdm(samples, desc=f"Evaluating {model_name}")):
        image = sample["image"]
        question = sample["question"]
        ground_truth = sample["answer"]

        inputs = build_inputs(processor, image, question)
        predicted = generate_answer(model, processor, inputs)

        # 根据问题类型选择合适的匹配方式
        phrase_type = sample.get("phrase_type", "freeform")   # VQA-RAD 中有此字段
        if phrase_type == "para":  # 封闭式
            is_correct = closed_accuracy(predicted, ground_truth)
        else:                      # 开放式
            is_correct = open_accuracy(predicted, ground_truth)

        if is_correct:
            correct += 1
        total += 1

        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{len(samples)}")

    accuracy = correct / total * 100 if total > 0 else 0.0
    print(f"\n✓ {model_name} Accuracy: {accuracy:.1f}% ({correct}/{total})\n")
    return accuracy

# ==================== 1. 零样本评估 ====================
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

zero_shot_acc = evaluate_model(base_model, processor, test_samples, model_name="Zero-shot")

# 释放显存
del base_model
torch.cuda.empty_cache()

# ==================== 2. QLoRA 微调模型评估 ====================
print("=" * 80)
print("2. Fine-tuned Evaluation (QLoRA)")
print("=" * 80)

# 重新加载基础模型并注入 LoRA
base_model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
finetuned_model = PeftModel.from_pretrained(base_model, LORA_PATH)
finetuned_model.eval()

finetuned_acc = evaluate_model(finetuned_model, processor, test_samples, model_name="QLoRA Fine-tuned")

# ==================== 3. 结果对比 ====================
print(f"Zero-shot Accuracy:       {zero_shot_acc:.1f}%")
print(f"QLoRA Fine-tuned Accuracy:{finetuned_acc:.1f}%")
print(f"Improvement:              {finetuned_acc - zero_shot_acc:+.1f}%")