# import torch
# from torch.utils.data import Dataset, DataLoader
# from torch.optim import AdamW
# from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, BitsAndBytesConfig
# from peft import get_peft_model, LoraConfig, TaskType
# from datasets import load_dataset
# from PIL import Image
# from tqdm import tqdm
# import io

# # ========== Configuration ==========
# MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
# BATCH_SIZE = 1
# LEARNING_RATE = 5e-4
# NUM_EPOCHS = 3
# MAX_SAMPLES = 2500
# LORA_RANK = 8

# print("=" * 80)
# print("🚀 Medical VQA LoRA Training")
# print("=" * 80)

# # ========== Check GPU ==========
# print("\n[1/7] Checking GPU...")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"✓ Device: {device}")
# if device.type == "cuda":
#     print(f"  GPU: {torch.cuda.get_device_name(0)}")
#     print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

# # ========== Load Model ==========
# print("\n[2/7] Loading Qwen2-VL Model with QLoRA...")

# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16 
# )

# processor = AutoProcessor.from_pretrained(
#     MODEL_ID,
#     trust_remote_code=True
# )
# model = Qwen2VLForConditionalGeneration.from_pretrained(
#     MODEL_ID,
#     quantization_config=bnb_config,
#     torch_dtype=torch.float16,
#     device_map="auto",
#     trust_remote_code=True
# )
# model.config.use_cache = False  # Disable cache for training
# print("✓ Model loaded successfully")

# # ========== Apply QLoRA ==========
# print("\n[3/7] Applying QLoRA...")
# lora_config = LoraConfig(
#     r=LORA_RANK,
#     lora_alpha=LORA_RANK * 2,
#     target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
#     lora_dropout=0.1,
#     bias="none",
#     task_type=TaskType.CAUSAL_LM,
# )
# model = get_peft_model(model, lora_config)
# model.gradient_checkpointing_enable()  # Save VRAM
# trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print(f"✓ LoRA applied successfully")
# print(f"  Trainable Parameters: {trainable_params / 1e6:.2f}M")

# # ========== Dataset Preparation ==========
# print("\n[4/7] Preparing Dataset...")

# class MedicalVQADataset(Dataset):
#     def __init__(self, dataset, processor, split='train', max_samples=None):
#         self.data = dataset[split]
#         self.processor = processor
#         if max_samples:
#             self.data = self.data.select(range(min(max_samples, len(self.data))))
    
#     def __len__(self):
#         return len(self.data)
    
#     def __getitem__(self, idx):
#         item = self.data[idx]
#         image = item['image']
#         question = item['question']
#         answer = item['answer']
        
#         # Format conversation using chat template
#         messages = [
#             {
#                 "role": "user",
#                 "content": [
#                     {"type": "image", "image": image},
#                     {"type": "text", "text": question},
#                 ],
#             }
#         ]

#         text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

#         # Process multimodal inputs
#         inputs = self.processor(
#             text=[text],
#             images=[image],
#             return_tensors="pt",
#             min_pixels=256*28*28,
#             max_pixels=512*28*28,
#             padding=True,
#         )
        
#         # Tokenize labels (answers)
#         labels = self.processor.tokenizer(
#             answer,
#             return_tensors="pt",
#             padding=True,
#             return_token_type_ids=False
#         ).input_ids
        
#         # Important: Qwen2-VL needs 'image_grid_thw' for vision encoding
#         return {
#             'input_ids': inputs['input_ids'].squeeze(0),
#             'pixel_values': inputs['pixel_values'],
#             'image_grid_thw': inputs['image_grid_thw'],
#             'labels': labels.squeeze(0),
#         }

# def collate_fn(batch):
#     """Custom collate function to handle variable sized tensors"""
#     return {
#         'input_ids': torch.stack([item['input_ids'] for item in batch]),
#         'pixel_values': torch.cat([item['pixel_values'] for item in batch], dim=0),
#         'image_grid_thw': torch.cat([item['image_grid_thw'] for item in batch], dim=0),
#         'labels': torch.stack([item['labels'] for item in batch]),
#     }

# # Load the medical VQA dataset
# vqa_rad = load_dataset("flaviagiammarino/vqa-rad")
# train_dataset = MedicalVQADataset(vqa_rad, 
#                                  processor, 
#                                  split='train', 
#                                  max_samples=MAX_SAMPLES)

# train_loader = DataLoader(train_dataset, 
#                           batch_size=BATCH_SIZE, 
#                           shuffle=True, 
#                           pin_memory=True,
#                           collate_fn=collate_fn)
# print(f"✓ Dataset ready: {len(train_loader)} batches")

# # ========== Optimizer ==========
# print("\n[5/7] Setting up Optimizer...")
# optimizer = AdamW(
#     [p for p in model.parameters() if p.requires_grad],
#     lr=LEARNING_RATE,
#     weight_decay=0.01,
# )
# print(f"✓ Optimizer ready")

# # ========== Training Function ==========
# def train_epoch(model, train_loader, optimizer, device, epoch):
#     model.train()
#     total_loss = 0
#     progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=True)
#     num_batches = 0

#     for batch_idx, batch in enumerate(progress_bar):
#         try:
#             if batch is None:
#                 continue
            
#             # Move data to device
#             input_ids = batch['input_ids'].to(device).long()
#             pixel_values = batch['pixel_values'].to(device).half()
#             image_grid_thw = batch['image_grid_thw'].to(device)
#             labels = batch['labels'].to(device).long()
            
#             # Zero gradients
#             optimizer.zero_grad()
            
#             # Forward pass
#             outputs = model(
#                 input_ids=input_ids,
#                 pixel_values=pixel_values,
#                 image_grid_thw=image_grid_thw,
#                 labels=labels
#             )
            
#             loss = outputs.loss
            
#             # Backward pass
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#             optimizer.step()
            
#             # Logging
#             total_loss += loss.item()
#             num_batches += 1
#             avg_loss = total_loss / num_batches
            
#             progress_bar.set_postfix({
#                 'loss': f'{loss.item():.4f}',
#                 'avg': f'{avg_loss:.4f}'
#             })
            
#             # Periodic cache clearing
#             if (batch_idx + 1) % 50 == 0:
#                 torch.cuda.empty_cache()
        
#         except RuntimeError as e:
#             if 'out of memory' in str(e):
#                 print(f"\n❌ OOM Error! Consider reducing MAX_SAMPLES or Image Resolution")
#                 torch.cuda.empty_cache()
#                 continue
#             else:
#                 raise
#         except Exception as e:
#             print(f"\n❌ Unexpected Error: {e}")
#             continue
    
#     return total_loss / num_batches if num_batches > 0 else 0


# # ========== Start Training ==========
# print("\n[6/7] Starting Training Phase...\n")
# print("=" * 80)

# for epoch in range(NUM_EPOCHS):
#     print(f"\nEPOCH {epoch + 1}/{NUM_EPOCHS}")
#     torch.cuda.reset_peak_memory_stats()
#     torch.cuda.empty_cache()
    
#     train_loss = train_epoch(model, train_loader, optimizer, device, epoch)
    
#     print(f"✓ Average Loss: {train_loss:.4f}")
    
#     peak_memory = torch.cuda.max_memory_allocated() / 1e9
#     print(f"✓ Peak VRAM Usage: {peak_memory:.2f}GB")
    
#     # Save checkpoint after every epoch
#     checkpoint_dir = f"./qwen2vl_lora_epoch{epoch+1}"
#     model.save_pretrained(checkpoint_dir)
#     processor.save_pretrained(checkpoint_dir)
#     print(f"✓ Checkpoint saved to: {checkpoint_dir}")

# print("\n" + "=" * 80)
# print("✅ Training Complete!")
# print("=" * 80)

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
    BitsAndBytesConfig,
)
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset
from tqdm import tqdm
import time
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")

# ==================== Configuration ====================
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"   # 可换成 Qwen/Qwen3-VL-0.8B-Instruct
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 2e-4                     # 原 5e-4 太高，LoRA 推荐 2e-4
NUM_EPOCHS = 3
MAX_SAMPLES = None                       # 不限制，用全部训练数据
LORA_RANK = 16                           # 16 效果更好
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
SEED = 42

OUTPUT_DIR = "./results/qwen2vl_2b"
CHECKPOINT_DIR = "./checkpoints/qwen2vl_2b"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# 固定随机性
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

epoch_times = []        # 每个 epoch 的秒数
epoch_memory_MB = []    # 每个 epoch 的峰值显存 (MB)

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
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,          # 统一 bfloat16
    device_map="auto",
    trust_remote_code=True,
)
model.config.use_cache = False
model.gradient_checkpointing_enable()

lora_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"✓ Trainable params: {trainable / 1e6:.2f}M")

# ==================== 3. Dataset ====================
print("\n[3/6] Preparing Dataset...")

dataset = load_dataset("flaviagiammarino/vqa-rad")
train_split = dataset["train"]
test_split = dataset["test"]

# 划分：保留 train，test 对半为 val / test
test_split = test_split.shuffle(seed=SEED)
split_idx = len(test_split) // 2
val_split = test_split.select(range(split_idx))
test_split = test_split.select(range(split_idx, len(test_split)))

print(f"Train: {len(train_split)} | Val: {len(val_split)} | Test: {len(test_split)}")

class MedicalVQADataset(Dataset):
    def __init__(self, hf_dataset, processor):
        self.data = hf_dataset
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = item["image"]
        question = item["question"]
        answer = item["answer"]

        # ---------- 构造完整对话 ----------
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

        # 完整文本（含答案，不加生成提示）
        full_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        # Tokenize 完整文本 + 图像
        full_inputs = self.processor(
            text=[full_text],
            images=[image],
            return_tensors="pt",
            min_pixels=256 * 28 * 28,
            max_pixels=512 * 28 * 28,
        )
        input_ids = full_inputs["input_ids"].squeeze(0)
        mm_token_type_ids = full_inputs.get("mm_token_type_ids", None)
        if mm_token_type_ids is not None:
            mm_token_type_ids = mm_token_type_ids.squeeze(0)

        # ---------- 获取 prompt 长度（只含问题部分） ----------
        prompt_messages = [messages[0]]   # 仅 user 部分
        prompt_text = self.processor.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        prompt_inputs = self.processor(
            text=[prompt_text],
            images=[image],
            return_tensors="pt",
            min_pixels=256 * 28 * 28,
            max_pixels=512 * 28 * 28,
        )
        prompt_len = prompt_inputs["input_ids"].shape[1]

        # ---------- 构建 labels ----------
        labels = input_ids.clone()
        # prompt 部分不计算损失
        labels[:prompt_len] = -100
        # padding 部分也忽略
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "pixel_values": full_inputs["pixel_values"],
            "image_grid_thw": full_inputs["image_grid_thw"],
            "mm_token_type_ids": mm_token_type_ids,
            "labels": labels,
        }

def collate_fn(batch):
    input_ids = torch.stack([item["input_ids"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    pixel_values = torch.cat([item["pixel_values"] for item in batch], dim=0)
    image_grid_thw = torch.cat([item["image_grid_thw"] for item in batch], dim=0)
    
    mm_token_type_ids = None
    if batch[0]["mm_token_type_ids"] is not None:
        mm_token_type_ids = torch.stack([item["mm_token_type_ids"] for item in batch])
    
    return {
        "input_ids": input_ids,
        "labels": labels,
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
        "mm_token_type_ids": mm_token_type_ids,
    }

train_dataset = MedicalVQADataset(train_split, processor)
val_dataset = MedicalVQADataset(val_split, processor)
# test_dataset 留给 evaluate.py

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
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    return total_loss / len(loader)

@torch.no_grad()
def validate_one_epoch(model, loader, device, epoch):
    model.eval()
    total_loss = 0
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
        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    return total_loss / len(loader)

# ==================== 6. Start Loop ====================
print("\n[5/6] Training...\n" + "=" * 80)
best_val_loss = float("inf")
for epoch in range(NUM_EPOCHS):
    torch.cuda.empty_cache()
    train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
    val_loss = validate_one_epoch(model, val_loader, device, epoch)

    print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f} | Val Loss = {val_loss:.4f}")

    # 保存最优模型
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        model.save_pretrained(os.path.join(OUTPUT_DIR, "best_model"))
        processor.save_pretrained(os.path.join(OUTPUT_DIR, "best_model"))
        print(f"  ✓ Best model saved (val_loss={val_loss:.4f})")

    # 每 epoch 保存检查点
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"epoch_{epoch+1}")
    model.save_pretrained(ckpt_path)
    processor.save_pretrained(ckpt_path)

print("\n" + "=" * 80)
print(f"✅ Training Complete! Best val loss: {best_val_loss:.4f}")
print(f"Results: {OUTPUT_DIR}  |  Checkpoints: {CHECKPOINT_DIR}")

fig_dir = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(fig_dir, exist_ok=True)

# ---------- 图1：训练时间柱状图 + 表格 ----------
fig, ax = plt.subplots(figsize=(8, 5))
epochs = list(range(1, NUM_EPOCHS + 1))
ax.bar(epochs, epoch_times, color="skyblue", edgecolor="black")
ax.set_xlabel("Epoch")
ax.set_ylabel("Time (seconds)")
ax.set_title("Training Time per Epoch")
ax.set_xticks(epochs)

# 在柱子上标注秒数
for i, v in enumerate(epoch_times):
    ax.text(i + 1, v + 2, f"{v:.1f}s", ha="center", fontsize=9)

# 在图表下方添加表格（用 matplotlib table）
table_data = [["Epoch", "Time (s)", "VRAM Peak (MB)"]]
for ep, t, m in zip(epochs, epoch_times, epoch_memory_MB):
    table_data.append([str(ep), f"{t:.1f}", f"{m:.0f}"])
table = ax.table(cellText=table_data, cellLoc="center",
                 colWidths=[0.15, 0.2, 0.2],
                 bbox=[0.1, -0.5, 0.8, 0.4])  # 放在图下方
ax.set_ylim(0, max(epoch_times) * 1.3)  # 留空间给标签
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "training_time.png"), dpi=150, bbox_inches="tight")
plt.close()

# ---------- 图2：显存使用饼图（用最后一次峰值） ----------
total_vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)  # MB
used_vram = epoch_memory_MB[-1]  # 取最后一个 epoch 的峰值，或整个训练过程的最大值
free_vram = total_vram - used_vram

fig, ax = plt.subplots()
labels = ["Used (peak)", "Free"]
sizes = [used_vram, free_vram]
colors = ["#ff9999", "#c2c2f0"]
explode = (0.05, 0)
ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
       shadow=True, startangle=90)
ax.set_title(f"GPU VRAM Usage (Total: {total_vram:.0f} MB)")
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "vram_pie.png"), dpi=150)
plt.close()

print(f"Charts saved to: {fig_dir}")