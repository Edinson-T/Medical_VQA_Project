# import torch
# from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
# from peft import PeftModel
# from datasets import load_dataset

# # 加载数据
# print("加载测试数据...")
# vqa_rad = load_dataset("flaviagiammarino/vqa-rad")
# test_samples = vqa_rad['test'].select(range(50))  # 用50个样本对比

# print(f"✓ 加载了{len(test_samples)}个测试样本\n")

# # ========== 第1个模型：Zero-shot（未微调）==========
# print("=" * 80)
# print("模型1：Zero-shot (未微调的原始模型)")
# print("=" * 80)

# model_id = "Qwen/Qwen2-VL-2B-Instruct"
# processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

# zero_shot_model = Qwen2VLForConditionalGeneration.from_pretrained(
#     model_id,
#     torch_dtype=torch.float16,
#     device_map="auto",
#     trust_remote_code=True
# )
# zero_shot_model.eval()

# print("✓ 模型加载完成\n")
# print("正在评估...")

# zero_shot_correct = 0
# zero_shot_total = 0

# for i, sample in enumerate(test_samples):
#     image = sample['image']
#     question = sample['question']
#     ground_truth = sample['answer']
    
#     messages = [
#         {
#             "role": "user",
#             "content": [
#                 {"type": "image", "image": image},
#                 {"type": "text", "text": question},
#             ],
#         }
#     ]
    
#     text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#     inputs = processor(text=[text], images=[image], return_tensors="pt")
#     inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    
#     with torch.no_grad():
#         outputs = zero_shot_model.generate(**inputs, max_new_tokens=100, temperature=0.7)
    
#     predicted = processor.decode(outputs[0], skip_special_tokens=True)
    
#     def check_answer(predicted, ground_truth):
#         pred_words = set(predicted.lower().split())
#         truth_words = set(ground_truth.lower().split())
#         # 如果有50%以上的词重合，就算正确
#         if len(truth_words) > 0:
#             overlap = len(pred_words & truth_words) / len(truth_words)
#             return overlap >= 0.5
#         return False

#     is_correct = check_answer(predicted, ground_truth)

    
#     if is_correct:
#         zero_shot_correct += 1
#     zero_shot_total += 1
    
#     if (i + 1) % 10 == 0:
#         print(f"  已评估 {i+1}/{len(test_samples)}")

# zero_shot_accuracy = zero_shot_correct / zero_shot_total * 100
# print(f"\n✓ Zero-shot 评估完成")
# print(f"  准确率: {zero_shot_accuracy:.1f}% ({zero_shot_correct}/{zero_shot_total})\n")

# # 清空显存
# del zero_shot_model
# torch.cuda.empty_cache()

# # ========== 第2个模型：LoRA微调后 ==========
# print("=" * 80)
# print("模型2：LoRA微调后的模型")
# print("=" * 80)

# base_model = Qwen2VLForConditionalGeneration.from_pretrained(
#     model_id,
#     torch_dtype=torch.float16,
#     device_map="auto",
#     trust_remote_code=True
# )

# finetuned_model = PeftModel.from_pretrained(base_model, "./qwen2vl_lora_epoch3")
# finetuned_model.eval()

# print("✓ 模型加载完成\n")
# print("正在评估...")

# finetuned_correct = 0
# finetuned_total = 0

# for i, sample in enumerate(test_samples):
#     image = sample['image']
#     question = sample['question']
#     ground_truth = sample['answer']
    
#     messages = [
#         {
#             "role": "user",
#             "content": [
#                 {"type": "image", "image": image},
#                 {"type": "text", "text": question},
#             ],
#         }
#     ]
    
#     text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#     inputs = processor(text=[text], images=[image], return_tensors="pt")
#     inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    
#     with torch.no_grad():
#         outputs = finetuned_model.generate(**inputs, max_new_tokens=100, temperature=0.7)
    
#     predicted = processor.decode(outputs[0], skip_special_tokens=True)
#     is_correct = ground_truth.lower() in predicted.lower()
    
#     if is_correct:
#         finetuned_correct += 1
#     finetuned_total += 1
    
#     if (i + 1) % 10 == 0:
#         print(f"  已评估 {i+1}/{len(test_samples)}")

# finetuned_accuracy = finetuned_correct / finetuned_total * 100
# print(f"\n✓ LoRA微调 评估完成")
# print(f"  准确率: {finetuned_accuracy:.1f}% ({finetuned_correct}/{finetuned_total})\n")

# # ========== 对比结果 ==========
# print("=" * 80)
# print("📊 对比结果")
# print("=" * 80)

# improvement = finetuned_accuracy - zero_shot_accuracy
# improvement_percent = (improvement / zero_shot_accuracy * 100) if zero_shot_accuracy > 0 else 0

# print(f"\nZero-shot (未微调):     {zero_shot_accuracy:.1f}%")
# print(f"LoRA微调后:             {finetuned_accuracy:.1f}%")
# print(f"\n改进:                   {improvement:+.1f}% ({improvement_percent:+.1f}%相对改进)")

# if improvement > 0:
#     print(f"\n✅ LoRA微调有效！")
# else:
#     print(f"\n⚠️ 微调后性能下降")

# print("\n" + "=" * 80)



import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from peft import PeftModel
from datasets import load_dataset

# Load data
print("Loading test data...")
vqa_rad = load_dataset("flaviagiammarino/vqa-rad")
test_samples = vqa_rad['test'].select(range(50))  # Use 50 samples for comparison

print(f"✓ Loaded {len(test_samples)} test samples\n")

# ========== Model 1: Zero-shot (No Fine-tuning) ==========
print("=" * 80)
print("Model 1: Zero-shot (Original model without fine-tuning)")
print("=" * 80)

model_id = "Qwen/Qwen2-VL-2B-Instruct"
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

zero_shot_model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
zero_shot_model.eval()

print("✓ Model loading complete\n")
print("Evaluating...")

zero_shot_correct = 0
zero_shot_total = 0

for i, sample in enumerate(test_samples):
    image = sample['image']
    question = sample['question']
    ground_truth = sample['answer']
    
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
    inputs = processor(text=[text], images=[image], return_tensors="pt")
    inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = zero_shot_model.generate(**inputs, max_new_tokens=100, temperature=0.7)
    
    predicted = processor.decode(outputs[0], skip_special_tokens=True)
    
    def check_answer(predicted, ground_truth):
        pred_words = set(predicted.lower().split())
        truth_words = set(ground_truth.lower().split())
        # If more than 50% of the words overlap, consider it correct
        if len(truth_words) > 0:
            overlap = len(pred_words & truth_words) / len(truth_words)
            return overlap >= 0.5
        return False

    is_correct = check_answer(predicted, ground_truth)

    if is_correct:
        zero_shot_correct += 1
    zero_shot_total += 1
    
    if (i + 1) % 10 == 0:
        print(f"  Evaluated {i+1}/{len(test_samples)}")

zero_shot_accuracy = zero_shot_correct / zero_shot_total * 100
print(f"\n✓ Zero-shot evaluation complete")
print(f"  Accuracy: {zero_shot_accuracy:.1f}% ({zero_shot_correct}/{zero_shot_total})\n")

# Clear VRAM
del zero_shot_model
torch.cuda.empty_cache()

# ========== Model 2: After LoRA Fine-tuning ==========
print("=" * 80)
print("Model 2: Fine-tuned model with LoRA")
print("=" * 80)

base_model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

finetuned_model = PeftModel.from_pretrained(base_model, "./qwen2vl_lora_epoch3")
finetuned_model.eval()

print("✓ Model loading complete\n")
print("Evaluating...")

finetuned_correct = 0
finetuned_total = 0

for i, sample in enumerate(test_samples):
    image = sample['image']
    question = sample['question']
    ground_truth = sample['answer']
    
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
    inputs = processor(text=[text], images=[image], return_tensors="pt")
    inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = finetuned_model.generate(**inputs, max_new_tokens=100, temperature=0.7)
    
    predicted = processor.decode(outputs[0], skip_special_tokens=True)
    is_correct = ground_truth.lower() in predicted.lower()
    
    if is_correct:
        finetuned_correct += 1
    finetuned_total += 1
    
    if (i + 1) % 10 == 0:
        print(f"  Evaluated {i+1}/{len(test_samples)}")

finetuned_accuracy = finetuned_correct / finetuned_total * 100
print(f"\n✓ LoRA fine-tuning evaluation complete")
print(f"  Accuracy: {finetuned_accuracy:.1f}% ({finetuned_correct}/{finetuned_total})\n")

# ========== Comparison Results ==========
print("=" * 80)
print("📊 Comparison Results")
print("=" * 80)

improvement = finetuned_accuracy - zero_shot_accuracy
improvement_percent = (improvement / zero_shot_accuracy * 100) if zero_shot_accuracy > 0 else 0

print(f"\nZero-shot (Base):       {zero_shot_accuracy:.1f}%")
print(f"LoRA Fine-tuned:        {finetuned_accuracy:.1f}%")
print(f"\nImprovement:            {improvement:+.1f}% ({improvement_percent:+.1f}% relative improvement)")

if improvement > 0:
    print(f"\n✅ LoRA fine-tuning is effective!")
else:
    print(f"\n⚠️ Performance decreased after fine-tuning")

print("\n" + "=" * 80)