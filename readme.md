# Medical VQA with QLoRA

Fine-tuning Qwen 3.5 (0.8B and 2B) for radiology Visual Question Answering using QLoRA and unsloth.  

---

## Models

Qwen3.5 (0.8B and 2B)

All models are fine-tuned on [VQA-RAD](https://huggingface.co/datasets/flaviagiammarino/vqa-rad) using QLoRA (4-bit NF4, LoRA r=8, α=16).

---

## Repository Structure

```text
├── 0.8B/
│   ├── train_qwen_3_5_0.8B.py       # Training script for Qwen3.5-0.8B
│   ├── evaluate_qwen_3_5_0.8B.py    # Evaluation script for Qwen3.5-0.8B
│   └── plot.py                      # Generate training curves and result plots
├── 2B/
│   ├── train_qwen_3_5_2B.py         # Training script for Qwen3.5-2B
│   ├── evaluate_qwen_3_5_2B.py      # Evaluation script for Qwen3.5-2B
│   └── plot.py                      # Generate training curves and result plots
├── checkpoints/
│   ├── Qwen3.5_0.8B/
│   └── Qwen3.5_2B/
├── results/
│   ├── Qwen3.5_0.8B/
│   └── Qwen3.5_2B/
├── unsloth_compiled_cache/
├── .gitignore
├── check_env.py                     # Environment and dependency sanity check
├── dataset_check.py                 # Dataset split and label distribution check
├── merge.py                         # Merge LoRA adapters into base model
├── readme.md
├── requirements.txt
├── VQA_RAD_Dataset_Public.json      # Additional metadata for VQA-RAD
└── web_demo.py                      # Gradio web demo for local deployment

---

## Setup

**Requirements:** Python 3.10+, CUDA-capable GPU (≥ 8 GB VRAM recommended)

```bash
pip install torch transformers peft trl bitsandbytes datasets unsloth gradio
```

Verify your environment before training:

```bash
python check_env.py
```

---

## Training

### Qwen3.5 models (via Unsloth)

```bash
python train_qwen_3_5_2B.py
```

Checkpoints are saved under `./checkpoints/Qwen3.5_2B/`. Training stats (loss, accuracy, VRAM) are logged to `./results/Qwen3.5_2B/training_stats.json`.

The 0.8B script follows the same structure.

---

## Evaluation

```bash
python evaluate_qwen_3_5_2B.py
```

Results are saved to `./results/Qwen3.5_2B/eval_results_2B.json`.  
To generate plots from saved results:

```bash
python plot.py
```

---

## Local Deployment

First merge the LoRA adapter into the base model:

```bash
python merge.py
```

Then launch the Gradio web demo:

```bash
#Port is customized
python web_demo.py --backend hf --checkpoint-path "Qwen/Qwen3.5-0.8B" --server-port 7861 
python web_demo.py --backend hf --checkpoint-path "./results/Qwen3.5_0.8B/merged_model" --server-port 7860

python web_demo.py --backend hf --checkpoint-path "./results/Qwen3.5_2B/merged_model" --server-port 7862
python web_demo.py --backend hf --checkpoint-path "Qwen/Qwen3.5-2B" --server-port 7863
```
---

## Dataset

We use the [VQA-RAD dataset](https://huggingface.co/datasets/flaviagiammarino/vqa-rad) 
Additional metadata (organ, answer type, question type) is sourced from `VQA_RAD_Dataset_Public.json`, available from the [official VQA-RAD release](https://osf.io/89kps/).

Split used for Qwen experiments:
- Train: 1,793 pairs (official)
- Val: 225 pairs (first half of HF test set)
- Test: 226 pairs (second half of HF test set)

---
