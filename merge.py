import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel

base_model_id = "Qwen/Qwen3.5-2B"
lora_path = "./checkpoints/Qwen3.5_2B/epoch_3"
merged_path = "./results/Qwen3.5_2B/merged_model"

# 加载基础模型到 CPU（避免显存不足）
print("Loading base model...")
model = AutoModelForImageTextToText.from_pretrained(
    base_model_id,
    torch_dtype=torch.bfloat16,
    device_map="cpu"
)

# 加载并合并 LoRA
print("Merging LoRA...")
model = PeftModel.from_pretrained(model, lora_path)
model = model.merge_and_unload()

# 保存合并后的完整模型
print("Saving merged model...")
model.save_pretrained(merged_path)

# 保存 processor（从基础模型复制，但要确保处理参数一致）
processor = AutoProcessor.from_pretrained(base_model_id)
processor.save_pretrained(merged_path)

print(f"Merged model saved to: {merged_path} !")