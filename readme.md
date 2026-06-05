# Medical VQA with QLoRA

Fine-tuning Qwen 3.5 (0.8B and 2B) for radiology Visual Question Answering using QLoRA and unsloth.  

---

## Models

Qwen3.5 (0.8B and 2B)

All models are fine-tuned on [VQA-RAD](https://huggingface.co/datasets/flaviagiammarino/vqa-rad) using QLoRA (4-bit NF4, LoRA r=8, α=16).

---
## Key Results

The fine-tuned models demonstrate significant improvements over zero-shot baselines, particularly in clinical reliability.

| Model | Method | Overall Accuracy | Closed-Ended Acc | "No" Recall | Open-Ended Acc |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Qwen3.5-0.8B** | Zero-Shot | 54.0% | 59.9% | 72.1% | 44.9% |
| **Qwen3.5-0.8B** | **Fine-tuned** | **58.8%** (+4.9%) | 62.0% | 58.8% | 53.9% |
| **Qwen3.5-2B** | Zero-Shot | 57.5% | 65.7% | 57.4% | 44.9% |
| **Qwen3.5-2B** | **Fine-tuned** | **64.2%** (+6.6%) | 67.9% | **80.9%** (+23.5%) | **58.4%** |
<img width="320" height="240" alt="0 8B_accuracy_comparison" src="https://github.com/user-attachments/assets/cff52abf-5594-4e82-a2b4-f4425a77dc29" />

<img width="320" height="240" alt="2B_accuracy_comparison" src="https://github.com/user-attachments/assets/2d6f4370-91aa-4628-8f5f-548ed5b45a62" />


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
