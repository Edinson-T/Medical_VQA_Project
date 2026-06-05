import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel

base_model_id = "Qwen/Qwen3.5-2B"
lora_path = "./checkpoints/Qwen3.5_2B/epoch_3"
merged_path = "./results/Qwen3.5_2B/merged_model"

# load base model to CPU to avoid GPU memory issues during merging
print("Loading base model...")
model = AutoModelForImageTextToText.from_pretrained(
    base_model_id,
    torch_dtype=torch.bfloat16,
    device_map="cpu"
)

# load and merge LoRA
print("Merging LoRA...")
model = PeftModel.from_pretrained(model, lora_path)
model = model.merge_and_unload()

# save the merged full model
print("Saving merged model...")
model.save_pretrained(merged_path)

# save processor (copy from base model, but ensure processing parameters are consistent)
processor = AutoProcessor.from_pretrained(base_model_id)
processor.save_pretrained(merged_path)

print(f"Merged model saved to: {merged_path} !")