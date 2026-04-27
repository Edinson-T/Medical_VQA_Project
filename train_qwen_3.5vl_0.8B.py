import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoModel,
    BitsAndBytesConfig,
)
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from datasets import load_dataset
from tqdm import tqdm
import time
import matplotlib
import matplotlib.pyplot as plt
import json
matplotlib.use("Agg")

# ==================== Configuration ====================
MODEL_ID = "Qwen/Qwen3.5-0.8B"  
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 1
LEARNING_RATE = 8e-5                   
NUM_EPOCHS = 3
MAX_SAMPLES = 100  # It is recommended to set this to a small number (e.g., 100) for quick testing. Set to None to use the full dataset.                       
LORA_RANK = 8                           
LORA_ALPHA = 16   # alpha = 2*rank
LORA_DROPOUT = 0.1
SEED = 42

OUTPUT_DIR = "./results/Qwen3.5_0.8B"
CHECKPOINT_DIR = "./checkpoints/Qwen3.5_0.8B"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Fix randomness
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

epoch_times = []        # seconds per epoch
epoch_memory_MB = []    # peak VRAM per epoch (MB)

print("=" * 80)
print("🚀 Medical VQA LoRA Training (Manual Loop)")
print("=" * 80)

# ==================== 1. GPU Check ====================
print("\n[1/6] Checking GPU...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✓ Device: {device}")
if device.type == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

# ==================== 2. Load Model & Processor ====================
print("\n[2/6] Loading Model with 4-bit QLoRA...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
model.config.use_cache = False
model.gradient_checkpointing_enable() 
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"], #,"gate_proj", "up_proj", "down_proj"
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
model.gradient_checkpointing_enable()
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
available, total = torch.cuda.mem_get_info()
model_loaded_used_mb = (total - available) / (1024 ** 2)
print(f"✓ Model loaded VRAM usage: {model_loaded_used_mb:.1f} MB")
print(f"✓ Trainable params: {trainable / 1e6:.2f}M")

available, total = torch.cuda.mem_get_info()
model_loaded_used_mb = (total - available) / (1024**2)

# ==================== 3. Dataset ====================
print("\n[3/6] Preparing Dataset...")

dataset = load_dataset("flaviagiammarino/vqa-rad")
train_split = dataset["train"]
test_split = dataset["test"]

# Split: keep train, split test into val/test equally
test_split = test_split.shuffle(seed=SEED)
split_idx = len(test_split) // 2
val_split = test_split.select(range(split_idx))
test_split = test_split.select(range(split_idx, len(test_split)))

print(f"Train: {len(train_split)} | Val: {len(val_split)} | Test: {len(test_split)}")

class MedicalVQADataset(Dataset):
    def __init__(self, hf_dataset, processor, max_samples=None):
        self.data = hf_dataset
        self.processor = processor
        if max_samples is not None:
            self.data = self.data.select(range(min(max_samples, len(hf_dataset))))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = item["image"]
        question = item["question"]
        answer = item["answer"]

        # ---------- Build full conversation ----------
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": answer}],
            },
        ]

        
        full_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        # Tokenize full text + image
        full_inputs = self.processor(
            text=[full_text],
            images=[image],
            return_tensors="pt",
            min_pixels=144 * 28 * 28,
            max_pixels=256 * 28 * 28,
        )
        input_ids = full_inputs["input_ids"].squeeze(0)
        mm_token_type_ids = full_inputs.get("mm_token_type_ids", None)
        if mm_token_type_ids is not None:
            mm_token_type_ids = mm_token_type_ids.squeeze(0)

        # ---------- Get prompt length (user part only) ----------
        prompt_messages = [messages[0]]   # user only
        prompt_text = self.processor.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        prompt_inputs = self.processor(
            text=[prompt_text],
            images=[image],
            return_tensors="pt",
            min_pixels=144 * 28 * 28,
            max_pixels=256 * 28 * 28,
        )
        prompt_len = prompt_inputs["input_ids"].shape[1]

        # ---------- Build labels ----------
        labels = input_ids.clone()
        # ignore loss on prompt part
        labels[:prompt_len] = -100
        # ignore padding tokens as well
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "pixel_values": full_inputs["pixel_values"],
            "image_grid_thw": full_inputs["image_grid_thw"],
            "mm_token_type_ids": mm_token_type_ids,
            "labels": labels,
        }

def collate_fn(batch):
    input_ids_list = [item["input_ids"] for item in batch]
    labels_list = [item["labels"] for item in batch]
    max_len = max(ids.size(0) for ids in input_ids_list)

    padded_input_ids = []
    padded_labels = []
    attention_mask = []
    for ids, labs in zip(input_ids_list, labels_list):
        pad_len = max_len - ids.size(0)
        # 用 tokenizer 的 pad_token_id 填充 input_ids，labels 填 -100
        padded_ids = torch.cat([ids, torch.full((pad_len,), processor.tokenizer.pad_token_id, dtype=ids.dtype)])
        padded_labs = torch.cat([labs, torch.full((pad_len,), -100, dtype=labs.dtype)])
        mask = torch.cat([torch.ones(ids.size(0), dtype=torch.long),
                          torch.zeros(pad_len, dtype=torch.long)])
        padded_input_ids.append(padded_ids)
        padded_labels.append(padded_labs)
        attention_mask.append(mask)

    input_ids = torch.stack(padded_input_ids)          # [B, max_len]
    labels = torch.stack(padded_labels)                # [B, max_len]
    attention_mask = torch.stack(attention_mask)       # [B, max_len]

    pixel_values = torch.cat([item["pixel_values"] for item in batch], dim=0)
    image_grid_thw = torch.cat([item["image_grid_thw"] for item in batch], dim=0)

    mm_token_type_ids = None
    if batch[0]["mm_token_type_ids"] is not None:
        mm_list = [item["mm_token_type_ids"] for item in batch]
        padded_mm = []
        for mm, target_len in zip(mm_list, [ids.size(0) for ids in input_ids_list]):
            pad_len = max_len - mm.size(0)
            padded_mm.append(torch.cat([mm, torch.zeros(pad_len, dtype=mm.dtype)]))  # 也可用 0 填充
        mm_token_type_ids = torch.stack(padded_mm)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
        "mm_token_type_ids": mm_token_type_ids,
    }

train_dataset = MedicalVQADataset(train_split, processor, max_samples=MAX_SAMPLES)
val_dataset = MedicalVQADataset(val_split, processor, max_samples=MAX_SAMPLES)
# test_dataset kept for evaluate.py

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn,
    pin_memory=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn,
    pin_memory=True,
)
print(f"✓ Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

# ==================== 4. Optimizer ====================
print("\n[4/6] Optimizer...")
optimizer = AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=LEARNING_RATE,
    weight_decay=0.01,
)
print("✓ AdamW ready")

# ==================== 5. Training & Validation ====================
def train_one_epoch(model, loader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    step_losses = []
    pbar = tqdm(loader, desc=f"Epoch {epoch+1} [Train]")
    for batch in pbar:
        input_ids = batch["input_ids"].to(device).long()
        pixel_values = batch["pixel_values"].to(device).bfloat16()
        image_grid_thw = batch["image_grid_thw"].to(device)
        labels = batch["labels"].to(device).long()
   
        extra_kwargs = {}
        if batch["mm_token_type_ids"] is not None:
            extra_kwargs["mm_token_type_ids"] = batch["mm_token_type_ids"].to(device)

        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            labels=labels,
            **extra_kwargs,
        )

        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        step_losses.append(loss.item())
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / len(loader)
    return avg_loss, step_losses

@torch.no_grad()
def validate_one_epoch(model, loader, device, epoch):
    model.eval()
    total_loss = 0
    step_losses = []
    pbar = tqdm(loader, desc=f"Epoch {epoch+1} [Val]")
    for batch in pbar:
        input_ids = batch["input_ids"].to(device).long()
        pixel_values = batch["pixel_values"].to(device).bfloat16()
        image_grid_thw = batch["image_grid_thw"].to(device)
        labels = batch["labels"].to(device).long()
        extra_kwargs = {}
        if batch["mm_token_type_ids"] is not None:
            extra_kwargs["mm_token_type_ids"] = batch["mm_token_type_ids"].to(device)

        outputs = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            **extra_kwargs,
            labels=labels,
        )
        loss = outputs.loss
        step_losses.append(loss.item())
        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    avg_loss = total_loss / len(loader)
    return avg_loss, step_losses

# ==================== 6. Start Loop ====================
print("\n[5/6] Training...\n" + "=" * 80)
best_val_loss = float("inf")

train_losses_epoch = []
val_losses_epoch = []
train_step_losses_all = []
val_step_losses_all = []

for epoch in range(NUM_EPOCHS):
    torch.cuda.reset_peak_memory_stats()
    epoch_start = time.time()

    torch.cuda.empty_cache()
    train_avg, train_steps = train_one_epoch(model, train_loader, optimizer, device, epoch)
    val_avg, val_steps = validate_one_epoch(model, val_loader, device, epoch)

    epoch_time = time.time() - epoch_start
    peak_memory_MB = torch.cuda.max_memory_allocated() / (1024 ** 2)

    # save stats for plotting later
    epoch_times.append(epoch_time)
    epoch_memory_MB.append(peak_memory_MB)
    train_losses_epoch.append(train_avg)
    val_losses_epoch.append(val_avg)
    train_step_losses_all.append(train_steps)
    val_step_losses_all.append(val_steps)

    print(f"Epoch {epoch+1}: Train Loss = {train_avg:.4f} | Val Loss = {val_avg:.4f} | Time = {epoch_time:.1f}s | Peak VRAM = {peak_memory_MB:.1f} MB")

    # save best model
    if val_avg < best_val_loss:
        best_val_loss = val_avg
        model.save_pretrained(os.path.join(OUTPUT_DIR, "best_model"))
        processor.save_pretrained(os.path.join(OUTPUT_DIR, "best_model"))
        print(f"  ✓ Best model saved (val_loss={val_avg:.4f})")

    # save checkpoint per epoch
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"epoch_{epoch+1}")
    model.save_pretrained(ckpt_path)
    processor.save_pretrained(ckpt_path)

print("\n" + "=" * 80)
print(f"Training Complete! Best val loss: {best_val_loss:.4f}")
print(f"Results: {OUTPUT_DIR}  |  Checkpoints: {CHECKPOINT_DIR}")


log_path = os.path.join(OUTPUT_DIR, "training_stats.json")
stats = {
    "model_loaded_vram_mb": model_loaded_used_mb,  
    "epoch_times": epoch_times,
    "epoch_memory_MB": epoch_memory_MB,
    "train_losses_epoch": train_losses_epoch,
    "val_losses_epoch": val_losses_epoch,
    "train_step_losses": train_step_losses_all,
    "val_step_losses": val_step_losses_all,
}
with open(log_path, "w") as f:
    json.dump(stats, f, indent=2)
print(f"Training stats saved to: {log_path}")
