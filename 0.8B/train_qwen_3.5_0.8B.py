"""
train_qwen3.5_0.8B_unsloth.py
"""

from unsloth import FastVisionModel
from functools import partial
import gc
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from datasets import load_dataset
from tqdm import tqdm
import time
import matplotlib
import json

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
matplotlib.use("Agg")

# ==================== Configuration ====================
MODEL_ID   = "Qwen/Qwen3.5-0.8B"
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4

LEARNING_RATE  = 2e-5
NUM_EPOCHS     = 3
MAX_SAMPLES    = None   # set e.g. 100 for a smoke-test run

LORA_RANK    = 8
LORA_ALPHA   = 2 * LORA_RANK
LORA_DROPOUT = 0
WARMUP_RATIO = 0.05

# ── Image resolution ─────────────────────────────────────────────────────────
MAX_PIXELS = 112 * 28 * 28

# Unsloth needs an explicit max_seq_length.
MAX_SEQ_LENGTH = 1024

SEED = 42

LOCAL_JSON_PATH = r"D:\1MA2Semester\ML&BDP\Medical_vqa_project\VQA_RAD_Dataset_Public.json"
OUTPUT_DIR     = "./results/Qwen3.5_0.8B"
CHECKPOINT_DIR = "./checkpoints/Qwen3.5_0.8B"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

epoch_times    = []
epoch_memory_MB = []

print("=" * 80)
print("Medical VQA  —  Unsloth LoRA Training  (Qwen3.5-0.8B)")
print("=" * 80)

# ==================== 1. GPU Check ====================
print("\n[1/6] Checking GPU...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✓ Device: {device}")
if device.type == "cuda":
    print(f"  GPU : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ==================== 2. Load Model with Unsloth ====================
print("\n[2/6] Loading model via Unsloth (4-bit QLoRA)...")

model, processor = FastVisionModel.from_pretrained(
    model_name     = MODEL_ID,
    max_seq_length = MAX_SEQ_LENGTH,
    load_in_4bit   = True,          # 4-bit NF4 quantisation (same as QLoRA)
    dtype          = torch.bfloat16,
    # Unsloth automatically handles gradient checkpointing and
    # prepare_model_for_kbit_training — no extra steps needed.
)

# Apply LoRA via Unsloth's wrapper.
# use_gradient_checkpointing="unsloth" activates Unsloth's custom
# activation-recompute kernel, which uses ~30% less VRAM than the
# standard HF gradient checkpointing.
model = FastVisionModel.get_peft_model(
    model,
    r              = LORA_RANK,
    lora_alpha     = LORA_ALPHA,
    lora_dropout   = LORA_DROPOUT,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
    # finetune_vision_layers=False keeps the vision encoder frozen.
    
    finetune_vision_layers      = False,
    finetune_language_layers    = True,
    finetune_attention_modules  = True,
    finetune_mlp_modules        = False,
    bias           = "none",
    random_state   = SEED,
    use_gradient_checkpointing  = "unsloth",
)


import torch.nn as nn
for name, module in model.named_modules():
    if isinstance(module, nn.LayerNorm):
        if hasattr(module, 'weight') and module.weight is not None:
            module.weight.data = module.weight.data.to(torch.bfloat16)
        if hasattr(module, 'bias') and module.bias is not None:
            module.bias.data = module.bias.data.to(torch.bfloat16)
print("✓ LayerNorm weights cast to BFloat16")

# Ensure pad token is set (same as QLoRA version)
processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
model.config.use_cache = False

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
available, total_mem = torch.cuda.mem_get_info()
model_loaded_used_mb = (total_mem - available) / (1024 ** 2)
print(f"✓ Model loaded  —  VRAM: {model_loaded_used_mb:.1f} MB  |  "
      f"Trainable params: {trainable / 1e6:.2f}M")

# ==================== 3. Dataset ====================
print("\n[3/6] Preparing dataset...")

dataset    = load_dataset("flaviagiammarino/vqa-rad")
train_split = dataset["train"]
test_split  = dataset["test"]

def load_local_metadata(json_path):
    print(f"  Loading metadata: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    print(f"  ✓ {len(metadata)} entries")
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
    example["answer_type"] = meta.get("answer_type", "OPEN")
    return example

test_split = test_split.shuffle(seed=SEED)
split_idx  = len(test_split) // 2
val_split  = test_split.select(range(split_idx))
test_split = test_split.select(range(split_idx, len(test_split)))
print(f"Train: {len(train_split)} | Val: {len(val_split)} | Test: {len(test_split)}")

full_metadata = load_local_metadata(LOCAL_JSON_PATH)
meta_map      = build_metadata_map(full_metadata)
val_split     = val_split.map(partial(add_metadata, meta_map=meta_map))
print("✓ Metadata merged into validation set")


class MedicalVQADataset(Dataset):
    def __init__(self, hf_dataset, processor, max_samples=None):
        self.data      = hf_dataset
        self.processor = processor
        if max_samples is not None:
            self.data = self.data.select(range(min(max_samples, len(hf_dataset))))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item     = self.data[idx]
        image    = item["image"]
        question = item["question"]
        answer   = item["answer"]

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text",  "text":  question},
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
        full_inputs = self.processor(
            text=[full_text], images=[image], return_tensors="pt",
            max_pixels=MAX_PIXELS,          # ← unified constant
        )
        input_ids = full_inputs["input_ids"].squeeze(0)
        mm_token_type_ids = full_inputs.get("mm_token_type_ids", None)
        if mm_token_type_ids is not None:
            mm_token_type_ids = mm_token_type_ids.squeeze(0)

        # prompt_len: must use the SAME max_pixels as full_inputs
        # so image-token counts match and label masking is correct.
        prompt_text = self.processor.apply_chat_template(
            [messages[0]], tokenize=False, add_generation_prompt=True
        )
        prompt_inputs = self.processor(
            text=[prompt_text], images=[image], return_tensors="pt",
            max_pixels=MAX_PIXELS,          # ← same constant, bug fix
        )
        prompt_len = prompt_inputs["input_ids"].shape[1]

        labels = input_ids.clone()
        labels[:prompt_len] = -100
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {
            "input_ids":         input_ids,
            "pixel_values":      full_inputs["pixel_values"],
            "image_grid_thw":    full_inputs["image_grid_thw"],
            "mm_token_type_ids": mm_token_type_ids,
            "labels":            labels,
        }


def collate_fn(batch):
    input_ids_list = [item["input_ids"] for item in batch]
    labels_list    = [item["labels"]    for item in batch]
    max_len        = max(ids.size(0) for ids in input_ids_list)

    padded_input_ids, padded_labels, attention_mask = [], [], []
    for ids, labs in zip(input_ids_list, labels_list):
        pad_len = max_len - ids.size(0)
        padded_input_ids.append(torch.cat([ids,  torch.full((pad_len,), processor.tokenizer.pad_token_id, dtype=ids.dtype)]))
        padded_labels.append(   torch.cat([labs, torch.full((pad_len,), -100, dtype=labs.dtype)]))
        attention_mask.append(  torch.cat([torch.ones(ids.size(0), dtype=torch.long),
                                           torch.zeros(pad_len,    dtype=torch.long)]))

    pixel_values   = torch.cat([item["pixel_values"]  for item in batch], dim=0)
    image_grid_thw = torch.cat([item["image_grid_thw"] for item in batch], dim=0)

    mm_token_type_ids = None
    if batch[0]["mm_token_type_ids"] is not None:
        padded_mm = []
        for item in batch:
            mm      = item["mm_token_type_ids"]
            pad_len = max_len - mm.size(0)
            padded_mm.append(torch.cat([mm, torch.zeros(pad_len, dtype=mm.dtype)]))
        mm_token_type_ids = torch.stack(padded_mm)

    return {
        "input_ids":         torch.stack(padded_input_ids),
        "attention_mask":    torch.stack(attention_mask),
        "labels":            torch.stack(padded_labels),
        "pixel_values":      pixel_values,
        "image_grid_thw":    image_grid_thw,
        "mm_token_type_ids": mm_token_type_ids,
    }


train_dataset = MedicalVQADataset(train_split, processor, max_samples=MAX_SAMPLES)
val_dataset   = MedicalVQADataset(val_split,   processor, max_samples=MAX_SAMPLES)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          collate_fn=collate_fn, pin_memory=False)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                          collate_fn=collate_fn, pin_memory=False)

# ==================== 4. Optimizer & Scheduler ====================
print("\n[4/6] Optimizer & Scheduler...")
optimizer    = AdamW([p for p in model.parameters() if p.requires_grad],
                     lr=LEARNING_RATE, weight_decay=0.01)
total_steps  = (len(train_loader) // GRADIENT_ACCUMULATION_STEPS) * NUM_EPOCHS
warmup_steps = int(total_steps * WARMUP_RATIO)
scheduler    = get_cosine_schedule_with_warmup(optimizer,
                   num_warmup_steps=warmup_steps,
                   num_training_steps=total_steps)
print(f"✓ AdamW + cosine schedule  (total_steps={total_steps}, warmup={warmup_steps})")

# ==================== 5. Train / Val functions ====================
def train_one_epoch(model, loader, optimizer, scheduler, device, epoch):
    model.train()
    total_loss, step_losses = 0, []
    pbar = tqdm(loader, desc=f"Epoch {epoch+1} [Train]")
    optimizer.zero_grad()

    for step, batch in enumerate(pbar):
        input_ids      = batch["input_ids"].to(device).long()
        pixel_values   = batch["pixel_values"].to(device).bfloat16()
        image_grid_thw = batch["image_grid_thw"].to(device)
        labels         = batch["labels"].to(device).long()

        extra_kwargs = {}
        if batch["mm_token_type_ids"] is not None:
            extra_kwargs["mm_token_type_ids"] = batch["mm_token_type_ids"].to(device)

        outputs = model(input_ids=input_ids, pixel_values=pixel_values,
                        image_grid_thw=image_grid_thw, labels=labels, **extra_kwargs)

        loss = outputs.loss / GRADIENT_ACCUMULATION_STEPS
        loss.backward()

        raw_loss = outputs.loss.item()
        total_loss += raw_loss
        step_losses.append(raw_loss)
        pbar.set_postfix({"loss": f"{raw_loss:.4f}"})

        is_last = (step + 1) == len(loader)
        if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0 or is_last:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    return total_loss / len(loader), step_losses


@torch.no_grad()
def validate_one_epoch(model, loader, device, epoch):
    model.eval()
    total_loss, step_losses = 0, []
    pbar = tqdm(loader, desc=f"Epoch {epoch+1} [Val]")
    for batch in pbar:
        input_ids      = batch["input_ids"].to(device).long()
        pixel_values   = batch["pixel_values"].to(device).bfloat16()
        image_grid_thw = batch["image_grid_thw"].to(device)
        labels         = batch["labels"].to(device).long()
        extra_kwargs   = {}
        if batch["mm_token_type_ids"] is not None:
            extra_kwargs["mm_token_type_ids"] = batch["mm_token_type_ids"].to(device)
        outputs = model(input_ids=input_ids, pixel_values=pixel_values,
                        image_grid_thw=image_grid_thw, labels=labels, **extra_kwargs)
        step_losses.append(outputs.loss.item())
        total_loss += outputs.loss.item()
        pbar.set_postfix({"loss": f"{outputs.loss.item():.4f}"})
    return total_loss / len(loader), step_losses


@torch.no_grad()
def compute_val_closed_accuracy(model, processor, val_split, device):
    model.eval()
    correct, total = 0, 0
    for sample in tqdm(val_split, desc="Val Closed Acc", leave=False):
        if sample.get("answer_type", "OPEN") != "CLOSED":
            continue
        image        = sample["image"]
        question     = sample["question"]
        ground_truth = sample["answer"].strip().lower()

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text":  question + "\nAnswer with only 'yes' or 'no':"},
            ],
        }]
        text   = processor.apply_chat_template(messages, tokenize=False,
                                               add_generation_prompt=True,
                                               enable_thinking   = False, )  # suppress Qwen3.5 chain-of-thought tokens
        inputs = processor(text=[text], images=[image], return_tensors="pt",
                           max_pixels=MAX_PIXELS)
        inputs = {
            k: v.to(device, dtype=torch.bfloat16) if (isinstance(v, torch.Tensor) and v.is_floating_point())
            else (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in inputs.items()
        }
        input_len = inputs["input_ids"].shape[1]

        outputs = model.generate(
            **inputs,
            max_new_tokens    = 5,
            do_sample         = False,
            pad_token_id      = processor.tokenizer.eos_token_id
        )

        generated_ids = outputs[0][input_len:]
        predicted = processor.decode(generated_ids, skip_special_tokens=True).strip().lower()

        matched = ""
        for token in predicted.split():
            if token in ("yes", "no"):
                matched = token
                break
        if not matched:
            if predicted.startswith("yes"):
                matched = "yes"
            elif predicted.startswith("no"):
                matched = "no"

        if matched == ground_truth:
            correct += 1
        total += 1

    gc.collect()
    torch.cuda.empty_cache()
    model.train()
    return correct / total if total > 0 else 0.0

# ==================== 6. Training Loop ====================
print("\n[5/6] Training...\n" + "=" * 80)

train_losses_epoch  = []
val_losses_epoch    = []
val_closed_acc_list = []

for epoch in range(NUM_EPOCHS):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    time.sleep(5)
    epoch_start = time.time()

    train_avg, _ = train_one_epoch(model, train_loader, optimizer, scheduler, device, epoch)
    val_avg,   _ = validate_one_epoch(model, val_loader, device, epoch)
    val_acc       = compute_val_closed_accuracy(model, processor, val_split, device)

    epoch_time   = time.time() - epoch_start
    peak_mem_mb  = torch.cuda.max_memory_allocated() / (1024 ** 2)

    epoch_times.append(epoch_time)
    epoch_memory_MB.append(peak_mem_mb)
    train_losses_epoch.append(train_avg)
    val_losses_epoch.append(val_avg)
    val_closed_acc_list.append(val_acc)

    print(f"Epoch {epoch+1}: Train={train_avg:.4f} | Val={val_avg:.4f} | "
          f"Closed Acc={val_acc:.4f} | {epoch_time:.1f}s | VRAM={peak_mem_mb:.1f}MB")

    ckpt_path = os.path.join(CHECKPOINT_DIR, f"epoch_{epoch+1}")
    model.save_pretrained(ckpt_path)
    processor.save_pretrained(ckpt_path)
    torch.save({
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch":     epoch,
        "train_losses_epoch":  train_losses_epoch,
        "val_losses_epoch":    val_losses_epoch,
        "val_closed_acc_list": val_closed_acc_list,
        "epoch_times":         epoch_times,
        "epoch_memory_MB":     epoch_memory_MB,
    }, os.path.join(ckpt_path, "train_state.pt"))

print("\n" + "=" * 80)
print(f"Training complete.  Results: {OUTPUT_DIR}  |  Checkpoints: {CHECKPOINT_DIR}")

stats_path = os.path.join(OUTPUT_DIR, "training_stats.json")
with open(stats_path, "w") as f:
    json.dump({
        "model_loaded_vram_mb": model_loaded_used_mb,
        "epoch_times":          epoch_times,
        "epoch_memory_MB":      epoch_memory_MB,
        "train_losses_epoch":   train_losses_epoch,
        "val_losses_epoch":     val_losses_epoch,
        "val_closed_acc":       val_closed_acc_list,
    }, f, indent=2)
print(f"Stats saved → {stats_path}")