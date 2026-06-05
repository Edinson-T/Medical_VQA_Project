# Medical VQA with QLoRA

https://github.com/Edinson-T/Medical_VQA_Project/n
Fine-tuning vision-language models for radiology Visual Question Answering using QLoRA.  

---

## Models

| Model  | LM Params |
| Qwen3.5-0.8B | 0.8B |
| Qwen3.5-2B  | 2.0B |

All models are fine-tuned on [VQA-RAD](https://huggingface.co/datasets/flaviagiammarino/vqa-rad) using QLoRA (4-bit NF4, LoRA r=8, α=16).

---

## Repository Structure
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

The 0.8B script follows the same structure with adjusted batch size (batch=2, grad-accum=4).

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
python web_demo.py
```

The demo runs on a single 8 GB GPU and accepts a radiology image + free-text question as input.

---

## Dataset

We use the [VQA-RAD dataset](https://huggingface.co/datasets/flaviagiammarino/vqa-rad) 
Additional metadata (organ, answer type, question type) is sourced from `VQA_RAD_Dataset_Public.json`, available from the [official VQA-RAD release](https://osf.io/89kps/).

Split used for Qwen experiments:
- Train: 1,793 pairs (official)
- Val: 225 pairs (first half of HF test set)
- Test: 226 pairs (second half of HF test set)

---
