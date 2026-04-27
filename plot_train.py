"""
plot_results.py - Generate training curves, loss, time, and VRAM charts
                  from saved training_stats.json and eval_results.json.
"""

import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from matplotlib.lines import Line2D

def load_json(path):
    """Safely load a JSON file, return None if not found."""
    if not os.path.exists(path):
        print(f"Warning: {path} not found, skipping related plots.")
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_loss_curves(train_losses, val_losses, fig_dir):
    """Plot average training and validation loss per epoch."""
    epochs = list(range(1, len(train_losses) + 1))
    fig, ax = plt.subplots()
    ax.plot(epochs, train_losses, marker='o', label='Train Loss', color='tab:blue')
    ax.plot(epochs, val_losses, marker='s', label='Val Loss', color='tab:red')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and Validation Loss per Epoch")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(fig_dir, "loss_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path.replace(os.sep, '/')}")


def plot_time_bar(epoch_times, epoch_memory_MB, fig_dir):
    """Plot training time per epoch with attached data table."""
    epochs = list(range(1, len(epoch_times) + 1))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(epochs, epoch_times, color="skyblue", edgecolor="black")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Training Time per Epoch")
    ax.set_xticks(epochs)
    for i, v in enumerate(epoch_times):
        ax.text(i + 1, v + 2, f"{v:.1f}s", ha="center", fontsize=9)

    table_data = [["Epoch", "Time (s)", "VRAM Peak (MB)"]]
    for ep, t, m in zip(epochs, epoch_times, epoch_memory_MB):
        table_data.append([str(ep), f"{t:.1f}", f"{m:.0f}"])
    ax.table(cellText=table_data, cellLoc="center",
             colWidths=[0.15, 0.2, 0.2],
             bbox=[0.1, -0.5, 0.8, 0.4])
    ax.set_ylim(0, max(epoch_times) * 1.3)
    plt.tight_layout()
    path = os.path.join(fig_dir, "training_time.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path.replace(os.sep, '/')}")


def plot_vram_breakdown(model_loaded_mb, train_peak_mb, fig_dir):
    """Draw a single pie chart: Free, Model Loaded, Training Overhead."""
    total_mb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
    overhead = max(0, train_peak_mb - model_loaded_mb)
    free_mb = total_mb - model_loaded_mb - overhead

    labels = ["Model Loaded", "Training Overhead", "Free"]
    sizes = [model_loaded_mb, overhead, free_mb]
    colors = ["#ffcc99", "#ff9999", "#c2c2f0"]
    explode = (0.02, 0.05, 0)

    fig, ax = plt.subplots()
    ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
           shadow=True, startangle=90)
    ax.set_title(f"GPU VRAM Breakdown (Total: {total_mb:.0f} MB)")
    plt.tight_layout()
    path = os.path.join(fig_dir, "vram_breakdown.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path.replace(os.sep, '/')}")


def plot_step_losses(train_step_losses, val_step_losses, fig_dir):
    """Plot fine-grained step-level loss curves with improved colors and legend."""
    fig, ax = plt.subplots(figsize=(10, 5))

    for ep_idx, steps in enumerate(train_step_losses):
        xs = [ep_idx + (i / (len(steps) or 1)) for i in range(len(steps))]
        ax.plot(xs, steps, alpha=0.8, linewidth=0.5, color='tab:blue')

    for ep_idx, steps in enumerate(val_step_losses):
        xs = [ep_idx + (i / (len(steps) or 1)) for i in range(len(steps))]
        ax.plot(xs, steps, alpha=0.8, linewidth=0.5, color='tab:red')

    legend_elements = [
        Line2D([0], [0], color='tab:blue', lw=1, label='Training loss (per step)'),
        Line2D([0], [0], color='tab:red', lw=1, label='Validation loss (per step)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Step-level Training and Validation Loss (All Epochs)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(fig_dir, "loss_steps.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path.replace(os.sep, '/')}")


def plot_accuracy_comparison(eval_data, fig_dir):
    """Grouped bar chart: Zero-shot vs Fine-tuned accuracy (overall, closed, open)."""
    if not eval_data:
        return
    zero = eval_data.get("zero_shot", {})
    fine = eval_data.get("finetuned", {})

    categories = ["Overall", "Closed-ended", "Open-ended"]
    zero_vals = [
        zero.get("overall_accuracy", 0) * 100,
        zero.get("closed_accuracy", 0) * 100,
        zero.get("open_accuracy", 0) * 100,
    ]
    fine_vals = [
        fine.get("overall_accuracy", 0) * 100,
        fine.get("closed_accuracy", 0) * 100,
        fine.get("open_accuracy", 0) * 100,
    ]

    x = range(len(categories))
    width = 0.35
    fig, ax = plt.subplots()
    bars1 = ax.bar([i - width/2 for i in x], zero_vals, width, label='Zero-shot', color='#8da0cb')
    bars2 = ax.bar([i + width/2 for i in x], fine_vals, width, label='Fine-tuned', color='#fc8d62')
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Zero-shot vs Fine-tuned Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    # Value labels
    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    path = os.path.join(fig_dir, "accuracy_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path.replace(os.sep, '/')}")


def plot_modality_heatmap(eval_data, fig_dir):
    """Plot per-modality accuracy as a horizontal bar chart for comparison."""
    if not eval_data:
        return
    zero_mod = eval_data.get("zero_shot", {}).get("per_modality_accuracy", {})
    fine_mod = eval_data.get("finetuned", {}).get("per_modality_accuracy", {})
    modalities = sorted(set(list(zero_mod.keys()) + list(fine_mod.keys())))
    if not modalities:
        return
    zero_vals = [zero_mod.get(m, 0) * 100 for m in modalities]
    fine_vals = [fine_mod.get(m, 0) * 100 for m in modalities]

    y = range(len(modalities))
    height = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh([i - height/2 for i in y], zero_vals, height, label='Zero-shot', color='#8da0cb')
    ax.barh([i + height/2 for i in y], fine_vals, height, label='Fine-tuned', color='#fc8d62')
    ax.set_yticks(y)
    ax.set_yticklabels(modalities)
    ax.set_xlabel("Accuracy (%)")
    ax.set_title("Per-Modality Accuracy Comparison")
    ax.legend()
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    path = os.path.join(fig_dir, "modality_accuracy.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path.replace(os.sep, '/')}")


def main():
    # Adjust to your actual result folder
    result_dir = "./results/Qwen3.5_0.8b"
    fig_dir = os.path.join(result_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # Load training stats
    stats_path = os.path.join(result_dir, "training_stats.json")
    stats = load_json(stats_path)

    if stats:
        # Extract data safely
        epoch_times = stats.get("epoch_times", [])
        epoch_memory_MB = stats.get("epoch_memory_MB", [])
        train_losses = stats.get("train_losses_epoch", [])
        val_losses = stats.get("val_losses_epoch", [])
        train_step_losses = stats.get("train_step_losses", [])
        val_step_losses = stats.get("val_step_losses", [])
        model_loaded_vram = stats.get("model_loaded_vram_mb", None)

        if train_losses and val_losses:
            plot_loss_curves(train_losses, val_losses, fig_dir)

        if epoch_times:
            plot_time_bar(epoch_times, epoch_memory_MB, fig_dir)

        if model_loaded_vram is not None and epoch_memory_MB:
            plot_vram_breakdown(model_loaded_vram, epoch_memory_MB[-1], fig_dir)
        elif epoch_memory_MB:
            # Fallback: only training peak (old format)
            total_mb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
            used = epoch_memory_MB[-1]
            free = total_mb - used
            fig, ax = plt.subplots()
            ax.pie([used, free], labels=["Used (peak)", "Free"], autopct='%1.1f%%',
                   startangle=90, colors=["#ff9999", "#c2c2f0"])
            ax.set_title(f"GPU VRAM Usage (Total: {total_mb:.0f} MB)")
            plt.tight_layout()
            path = os.path.join(fig_dir, "vram_legacy.png")
            plt.savefig(path, dpi=150)
            plt.close()
            print(f"Saved {path.replace(os.sep, '/')} (legacy VRAM pie)")

        if train_step_losses and val_step_losses:
            plot_step_losses(train_step_losses, val_step_losses, fig_dir)

    # Load evaluation results (optional)
    eval_path = os.path.join(result_dir, "eval_results.json")
    eval_data = load_json(eval_path)
    if eval_data:
        plot_accuracy_comparison(eval_data, fig_dir)
        plot_modality_heatmap(eval_data, fig_dir)

    print("All done.")


if __name__ == "__main__":
    main()