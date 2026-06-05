"""
plot_results.py - Generate training curves, loss, time, and VRAM charts
                  from saved training_stats.json and eval_results.json.
"""

import os
import json
import matplotlib
matplotlib.use("Agg")
from matplotlib import ticker
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


def plot_loss_and_acc(train_losses, val_losses, val_acc, fig_dir):
    """Combined chart: training & validation loss (left y-axis)
       and validation closed-ended accuracy (right y-axis)."""
    epochs = list(range(1, len(train_losses) + 1))
    fig, ax1 = plt.subplots()

    # Loss curves on left y-axis
    ax1.plot(epochs, train_losses, marker='o', label='Train Loss', color='tab:blue')
    ax1.plot(epochs, val_losses, marker='s', label='Val Loss', color='tab:red')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.grid(True, alpha=0.3)

    # Accuracy curve on right y-axis
    ax2 = ax1.twinx()
    ax2.plot(epochs, [v * 100 for v in val_acc], marker='D', color='green',
             linestyle='--', label='Val Closed Acc')
    ax2.set_ylabel("Accuracy (%)", color='green')
    ax2.tick_params(axis='y', labelcolor='green')

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='center right')

    ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    plt.title("Training Loss, Validation Loss and Closed Accuracy")
    plt.tight_layout()
    path = os.path.join(fig_dir, "2B_loss_and_acc.png")
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

    ax.set_ylim(0, max(epoch_times) * 1.3)
    plt.tight_layout()
    path = os.path.join(fig_dir, "2B_training_time.png")
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
    path = os.path.join(fig_dir, "2B_vram_breakdown.png")
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
    path = os.path.join(fig_dir, "2B_accuracy_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path.replace(os.sep, '/')}")

def plot_organ_comparison(eval_data, fig_dir):
    """Horizontal bar chart: per-organ accuracy (Zero-shot vs Fine-tuned)."""
    if not eval_data:
        return
    zero_org = eval_data.get("zero_shot", {}).get("per_organ_accuracy", {})
    fine_org = eval_data.get("finetuned", {}).get("per_organ_accuracy", {})
    organs = sorted(set(list(zero_org.keys()) + list(fine_org.keys())))
    if not organs:
        return

    zero_vals = [zero_org.get(o, 0) * 100 for o in organs]
    fine_vals = [fine_org.get(o, 0) * 100 for o in organs]

    y = range(len(organs))
    height = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh([i - height/2 for i in y], zero_vals, height,
            label='Zero-shot', color='#8da0cb')
    ax.barh([i + height/2 for i in y], fine_vals, height,
            label='Fine-tuned', color='#fc8d62')
    ax.set_yticks(y)
    ax.set_yticklabels(organs)
    ax.set_xlabel("Accuracy (%)")
    ax.set_title("Per-Organ Accuracy Comparison")
    ax.legend()
    ax.grid(axis='x', alpha=0.3)

    # Value labels on bars
    for i, (zv, fv) in enumerate(zip(zero_vals, fine_vals)):
        ax.text(zv + 1, i - height/2, f'{zv:.1f}', va='center', fontsize=8)
        ax.text(fv + 1, i + height/2, f'{fv:.1f}', va='center', fontsize=8)

    plt.tight_layout()
    path = os.path.join(fig_dir, "2B_organ_accuracy.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path.replace(os.sep, '/')}")

def plot_question_type_comparison(eval_data, fig_dir, min_samples=5):
    """Horizontal bar chart: per-question-type accuracy (Zero-shot vs Fine-tuned).
       Filters out categories with too few samples."""
    if not eval_data:
        return
    zero_qt = eval_data.get("zero_shot", {}).get("per_question_type_accuracy", {})
    fine_qt = eval_data.get("finetuned", {}).get("per_question_type_accuracy", {})

    details_ft = eval_data.get("finetuned", {}).get("details", [])
    qt_counts = {}
    for d in details_ft:
        qt = d.get("question_type", "other")
        qt_counts[qt] = qt_counts.get(qt, 0) + 1

    qt_list = [qt for qt in set(list(zero_qt.keys()) + list(fine_qt.keys()))
               if qt_counts.get(qt, 0) >= min_samples]
    qt_list.sort(key=lambda qt: fine_qt.get(qt, 0), reverse=True)

    if not qt_list:
        return

    zero_vals = [zero_qt.get(qt, 0) * 100 for qt in qt_list]
    fine_vals = [fine_qt.get(qt, 0) * 100 for qt in qt_list]

    y = range(len(qt_list))
    height = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh([i - height/2 for i in y], zero_vals, height,
            label='Zero-shot', color='#8da0cb')
    ax.barh([i + height/2 for i in y], fine_vals, height,
            label='Fine-tuned', color='#fc8d62')
    ax.set_yticks(y)
    ax.set_yticklabels(qt_list)
    ax.set_xlabel("Accuracy (%)")
    ax.set_title("Per-Question-Type Accuracy Comparison")
    ax.legend()
    ax.grid(axis='x', alpha=0.3)

    # Value labels on bars
    for i, (zv, fv) in enumerate(zip(zero_vals, fine_vals)):
        ax.text(zv + 1, i - height/2, f'{zv:.1f}', va='center', fontsize=8)
        ax.text(fv + 1, i + height/2, f'{fv:.1f}', va='center', fontsize=8)

    plt.tight_layout()
    path = os.path.join(fig_dir, "2B_question_type_accuracy.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path.replace(os.sep, '/')}")




def main():
    # Adjust to your actual result folder
    result_dir = "./results/Qwen3.5_2B"
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
        val_closed_acc = stats.get("val_closed_acc", None)
        model_loaded_vram = stats.get("model_loaded_vram_mb", None)

        if train_losses and val_losses and val_closed_acc:
            plot_loss_and_acc(train_losses, val_losses, val_closed_acc, fig_dir)

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


    # Load evaluation results (optional)
    eval_path = os.path.join(result_dir, "eval_results_2B.json")
    eval_data = load_json(eval_path)
    if eval_data:
        plot_accuracy_comparison(eval_data, fig_dir)
        plot_organ_comparison(eval_data, fig_dir)                  
        plot_question_type_comparison(eval_data, fig_dir, min_samples=15) 

    print("All done.")


if __name__ == "__main__":
    main()