# -*- coding: utf-8 -*-
"""
Generate comparative gradient descent convergence plots across the ResNet family (ResNet-18, ResNet-34, ResNet-50).
"""
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
TUNING_DIR = REPO_ROOT / "artifacts" / "tuning"
FIGURES_DIR = REPO_ROOT / "artifacts" / "figures"


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    models = ["resnet18", "resnet34", "resnet50"]
    data = {}

    for m in models:
        fpath = TUNING_DIR / f"tuning_results_{m}.json"
        if not fpath.exists():
            print(f"[!] File not found: {fpath}")
            return
        with open(fpath, "r", encoding="utf-8") as f:
            data[m] = json.load(f)

    fig, axes = plt.subplots(2, 2, figsize=(15, 11), dpi=200)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    arch_labels = {
        "resnet18": "ResNet-18 (11.3M params)",
        "resnet34": "ResNet-34 (21.4M params)",
        "resnet50": "ResNet-50 (24.0M params)",
    }
    colors = {"resnet18": "#1f77b4", "resnet34": "#2ca02c", "resnet50": "#d62728"}
    markers = {"resnet18": "o", "resnet34": "s", "resnet50": "^"}

    REPORT_FIGURES_DIR = REPO_ROOT / "report" / "figures"
    REPORT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Best Training Loss Curve
    ax = axes[0, 0]
    for m in models:
        h = data[m]["best_trial"]["history"]
        eps = list(range(1, len(h["train_loss"]) + 1))
        ax.plot(eps, h["train_loss"], label=f"{arch_labels[m]}", color=colors[m], marker=markers[m], linewidth=2.2, markersize=6.5)
    ax.set_title("Training Loss Convergence (Selected Baseline)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=11.5)
    ax.set_ylabel("Cross-Entropy Loss", fontsize=11.5)
    ax.legend(fontsize=11, framealpha=0.9)
    ax.tick_params(labelsize=10.5)
    ax.grid(True, linestyle="--", alpha=0.6)

    # 2. Best Validation Loss Curve
    ax = axes[0, 1]
    for m in models:
        h = data[m]["best_trial"]["history"]
        v_loss_seq = h.get("val_unsmoothed_loss", h["val_loss"])
        eps = list(range(1, len(v_loss_seq) + 1))
        best_v = data[m]["best_trial"].get("best_val_unsmoothed_loss", data[m]["best_trial"].get("best_val_loss", 0.0))
        ax.plot(eps, v_loss_seq, label=f"{arch_labels[m]} (Best: {best_v:.4f})", color=colors[m], marker=markers[m], linewidth=2.2, markersize=6.5)
    ax.set_title("Validation Loss Generalization Dynamics (Unsmoothed CE)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=11.5)
    ax.set_ylabel("Validation Loss (Unsmoothed CE)", fontsize=11.5)
    ax.legend(fontsize=11, framealpha=0.9)
    ax.tick_params(labelsize=10.5)
    ax.grid(True, linestyle="--", alpha=0.6)

    # 3. Validation Accuracy Curve
    ax = axes[1, 0]
    for m in models:
        h = data[m]["best_trial"]["history"]
        eps = list(range(1, len(h["val_acc"]) + 1))
        ax.plot(eps, h["val_acc"], label=f"{arch_labels[m]} (Selected: {data[m]['best_trial']['best_val_acc']:.2f}%)", color=colors[m], marker=markers[m], linewidth=2.2, markersize=6.5)
    ax.set_title("Top-1 Validation Accuracy (%) Trajectory", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=11.5)
    ax.set_ylabel("Validation Accuracy (%)", fontsize=11.5)
    ax.legend(fontsize=11, framealpha=0.9)
    ax.tick_params(labelsize=10.5)
    ax.grid(True, linestyle="--", alpha=0.6)

    # 4. Validation Macro-F1 Curve
    ax = axes[1, 1]
    for m in models:
        h = data[m]["best_trial"]["history"]
        eps = list(range(1, len(h["val_macro_f1"]) + 1))
        ax.plot(eps, h["val_macro_f1"], label=f"{arch_labels[m]} (Selected: {data[m]['best_trial']['best_val_macro_f1']:.2f}%)", color=colors[m], marker=markers[m], linewidth=2.2, markersize=6.5)
    ax.set_title("Validation Macro-F1 (%) Trajectory", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=11.5)
    ax.set_ylabel("Validation Macro-F1 (%)", fontsize=11.5)
    ax.legend(fontsize=11, framealpha=0.9)
    ax.tick_params(labelsize=10.5)
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    out_file = FIGURES_DIR / "resnet_family_gradient_descent_comparison.png"
    plt.savefig(out_file)
    report_out_file = REPORT_FIGURES_DIR / "resnet_family_gradient_descent_comparison.png"
    plt.savefig(report_out_file)
    plt.close()
    print(f"[+] Successfully saved ResNet family comparison figure to: {out_file} and {report_out_file}")


if __name__ == "__main__":
    main()
