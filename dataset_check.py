from collections import Counter
import gc
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

LOCAL_JSON_PATH = r"D:\1MA2Semester\ML&BDP\Medical_vqa_project\VQA_RAD_Dataset_Public.json"
dataset = load_dataset("flaviagiammarino/vqa-rad")
train_split = dataset["train"]
test_split = dataset["test"]

# Split: keep train, split test into val/test equally
test_split = test_split.shuffle(seed=42)
split_idx = len(test_split) // 2
val_split = test_split.select(range(split_idx))
test_split = test_split.select(range(split_idx, len(test_split)))

print(f"Train: {len(train_split)} | Val: {len(val_split)} | Test: {len(test_split)}")
with open(LOCAL_JSON_PATH, "r") as f:
    full_metadata = json.load(f)
meta_map = {}
for item in full_metadata:
    q = item["question"].strip().lower()
    a = str(item["answer"]).strip().lower()
    meta_map[(q, a)] = item["answer_type"]

def add_answer_type(example):
    q = example["question"].strip().lower()
    a = str(example["answer"]).strip().lower()
    example["answer_type"] = meta_map.get((q, a), "unknown")
    return example

val_split = val_split.map(add_answer_type)
test_split = test_split.map(add_answer_type)

print("Val answer_type:", Counter(s["answer_type"] for s in val_split))
print("Test answer_type:", Counter(s["answer_type"] for s in test_split))

val_yes = val_no = 0
for s in val_split:
    if s.get("answer_type", "OPEN") == "CLOSED":
        if s["answer"].strip().lower() == "yes":
            val_yes += 1
        elif s["answer"].strip().lower() == "no":
            val_no += 1
print(f"Val closed yes: {val_yes}, no: {val_no}")