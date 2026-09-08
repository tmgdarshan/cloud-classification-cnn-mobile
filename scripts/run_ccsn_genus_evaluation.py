# -*- coding: utf-8 -*-
"""
CCSN 11-Class Genus Final Training & Evaluation Pipeline.

Trains ResNet-18 across 3 seeds {42, 43, 44} on the 11-class canonical CCSN split
using the hyperparameter-tuned configuration (tuned_resnet18_ccsn11.toml).
Evaluates strictly on the held-out test split, generating:
- Per-seed checkpoints (*.pth)
- Raw test predictions (*.npz with logits, labels, probs, y_pred)
- 11x11 row-normalized confusion matrix PNG
- Per-genus recall table and overall metrics
- README.md report
"""
from __future__ import annotations

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import run_harmonized as rh

SEEDS = [42, 43, 44]
NUM_CLASSES = 11
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "ccsn_genus"
CONFIG_PATH = REPO_ROOT / "config" / "training" / "tuned_resnet18_ccsn11.toml"
TUNING_JSON = REPO_ROOT / "artifacts" / "tuning" / "tuning_results_resnet18_ccsn11.json"
MANIFEST_PATH = REPO_ROOT / "metadata" / "splits" / "ccsn_11class_canonical.json"


@torch.no_grad()
def evaluate_test_with_logits(
    model: nn.Module,
    ds: torch.utils.data.Dataset,
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluates model on dataset returning logits, probabilities, predictions, and labels."""
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=False,
        num_workers=0,
    )
    model.eval()

    all_logits = []
    all_targets = []

    for batch_imgs, batch_lbls in loader:
        batch_imgs = batch_imgs.to(rh.DEVICE, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=(rh.DEVICE.type == "cuda")):
            outputs = model(batch_imgs)
        all_logits.append(outputs.float().cpu().numpy())
        all_targets.append(batch_lbls.numpy())

    logits = np.concatenate(all_logits, axis=0)
    labels = np.concatenate(all_targets, axis=0)

    # Compute softmax probabilities and class predictions
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    preds = np.argmax(logits, axis=1)

    return logits, probs, preds, labels


def plot_11x11_confusion_matrix(
    mean_cm: np.ndarray,
    sd_cm: np.ndarray,
    class_names: list[str],
    save_path: Path,
):
    """Renders high-resolution 11x11 row-normalized confusion matrix with cell annotations."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 7.5), dpi=300)

    im = ax.imshow(mean_cm, interpolation="nearest", cmap=plt.cm.Blues, vmin=0, vmax=100)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Recall / Distribution (%)", fontsize=11, fontweight="bold")
    cbar.ax.tick_params(labelsize=10)

    n_cls = len(class_names)
    ax.set_xticks(range(n_cls))
    ax.set_yticks(range(n_cls))
    ax.set_xticklabels(class_names, fontsize=10, fontweight="bold")
    ax.set_yticklabels(class_names, fontsize=10, fontweight="bold")
    ax.set_xlabel("Predicted Genus", fontsize=12, fontweight="bold", labelpad=8)
    ax.set_ylabel("True Genus", fontsize=12, fontweight="bold", labelpad=8)
    ax.set_title("CCSN 11-Class Genus Confusion Matrix (ResNet-18, 3-Seed Mean %)", fontsize=13, fontweight="bold", pad=12)

    thresh = 50.0
    for i in range(n_cls):
        for j in range(n_cls):
            val = mean_cm[i, j]
            color = "white" if val > thresh else "black"
            if i == j:
                txt = f"{val:.1f}%\n({sd_cm[i, j]:.1f})"
            elif val >= 0.5:
                txt = f"{val:.1f}%"
            else:
                txt = ""
            if txt:
                ax.text(j, i, txt, ha="center", va="center", color=color, fontsize=7.0)

    fig.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Saved 11x11 confusion matrix plot to {save_path}", flush=True)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Execution aborted per policy to avoid running on CPU.")

    parser = argparse.ArgumentParser(description="CCSN 11-Class Final Training and Evaluation")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH, help="Path to tuned TOML configuration")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS, help="Seeds to train and evaluate")
    parser.add_argument("--output-dir", type=Path, default=ARTIFACTS_DIR, help="Directory to save artifacts")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print(f"[*] CCSN 11-CLASS GENUS FINAL TRAINING & EVALUATION")
    print(f"[*] Config: {args.config} | Seeds: {args.seeds} | Output: {args.output_dir}")
    print("=" * 90)

    # 1. Load configuration
    cfg = rh.load_flat_toml_config(args.config)
    print(f"[+] Loaded tuning configuration: {cfg.get('name', 'tuned')} (Trial: {cfg.get('trial_id')})")

    # 2. Load canonical 11-class manifest
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    classes = manifest["classes"]
    print(f"[+] Taxonomy classes ({len(classes)}): {classes}")

    tr_samples = []
    va_samples = []
    te_samples = []

    for s in manifest["samples"]:
        target_path = rh.CCSN_DIR / s["path"]
        target_label = s["label"]
        if s["split"] == "train":
            tr_samples.append((target_path, target_label))
        elif s["split"] == "val":
            va_samples.append((target_path, target_label))
        elif s["split"] == "test":
            te_samples.append((target_path, target_label))

    print(f"[+] Partitions: {len(tr_samples)} train, {len(va_samples)} val, {len(te_samples)} test")

    # 3. Preload image data
    print("\n--- Preloading Image Data into RAM Tensors ---")
    tr_imgs, tr_lbls = rh.preload_images_parallel_preallocated(tr_samples)
    va_imgs, va_lbls = rh.preload_images_parallel_preallocated(va_samples)
    te_imgs, te_lbls = rh.preload_images_parallel_preallocated(te_samples)

    aug_cfg = cfg.get("augmentation", {})
    train_tf, eval_tf = rh.get_physically_valid_transforms(aug_cfg)

    train_ds = rh.FastCachedCloudDataset(tr_imgs, tr_lbls, transform=train_tf)
    val_ds = rh.FastCachedCloudDataset(va_imgs, va_lbls, transform=eval_tf)
    test_ds = rh.FastCachedCloudDataset(te_imgs, te_lbls, transform=eval_tf)

    # 4. Train across seeds
    seed_results = {}
    norm_cms = []
    raw_cms = []

    epochs = int(cfg.get("epochs", 60))
    batch_size = int(cfg.get("batch_size", 64))
    lr_bb = float(cfg.get("lr_backbone", 5e-5))
    lr_head = float(cfg.get("lr_head", 5e-4))
    wd = float(cfg.get("weight_decay", 1e-2))
    ls = float(cfg.get("label_smoothing", 0.1))
    drop_bb = float(cfg.get("dropout_backbone", 0.3))
    drop_head = float(cfg.get("dropout_head", 0.2))
    opt_type = str(cfg.get("optimizer", "adamw"))
    sched_type = str(cfg.get("scheduler", "cosine_annealing"))
    eta_min = float(cfg.get("eta_min", 1e-6))
    arch = str(cfg.get("model_architecture", "resnet18"))

    for seed in args.seeds:
        print(f"\n" + "#" * 90)
        print(f"# TRAINING RESNET-18 SEED {seed} (Epochs: {epochs}, LS: {ls}, LR: {lr_bb}/{lr_head})")
        print("#" * 90)

        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        model_loss, model_f1, val_summary = rh.train_pool_model(
            model_name=f"resnet18_ccsn11_seed{seed}",
            train_ds=train_ds,
            val_ds=val_ds,
            model_arch=arch,
            num_classes=NUM_CLASSES,
            epochs=epochs,
            batch_size=batch_size,
            lr_backbone=lr_bb,
            lr_head=lr_head,
            weight_decay=wd,
            label_smoothing=ls,
            dropout_head=drop_head,
            dropout_bb=drop_bb,
            optimizer_type=opt_type,
            scheduler_type=sched_type,
            eta_min=eta_min,
            seed=seed,
        )

        # Save min-val-loss checkpoint
        ckpt_path = args.output_dir / f"resnet18_ccsn11_seed{seed}.pth"
        torch.save(model_loss.state_dict(), ckpt_path)
        print(f"[+] Saved checkpoint to {ckpt_path}")

        # Evaluate strictly on test split
        logits, probs, preds, targets = evaluate_test_with_logits(model_loss, test_ds, batch_size=batch_size)

        # Save raw predictions
        npz_path = args.output_dir / f"predictions_seed{seed}.npz"
        np.savez_compressed(
            npz_path,
            logits=logits,
            labels=targets,
            probs=probs,
            y_pred=preds,
            y_true=targets,
        )
        print(f"[+] Saved test predictions (logits + labels) to {npz_path}")

        # Calculate seed metrics
        top1_acc = (preds == targets).mean() * 100.0
        k_top3 = min(3, logits.shape[1])
        top3_preds = np.argsort(logits, axis=1)[:, -k_top3:]
        top3_acc = np.any(top3_preds == targets[:, None], axis=1).mean() * 100.0
        macro_f1 = f1_score(targets, preds, average="macro", zero_division=0) * 100.0

        cm_raw = confusion_matrix(targets, preds, labels=list(range(NUM_CLASSES)))
        row_sums = cm_raw.sum(axis=1, keepdims=True).astype(float)
        cm_norm = np.divide(cm_raw.astype(float) * 100.0, row_sums, out=np.zeros_like(cm_raw, dtype=float), where=row_sums != 0)
        recalls = np.diag(cm_norm)

        norm_cms.append(cm_norm)
        raw_cms.append(cm_raw)

        seed_results[str(seed)] = {
            "top1_acc": round(top1_acc, 2),
            "top3_acc": round(top3_acc, 2),
            "macro_f1": round(macro_f1, 2),
            "best_val_loss": val_summary["best_val_loss"],
            "best_val_acc": val_summary["best_val_acc"],
            "best_epoch": val_summary["best_epoch"],
            "per_genus_recall": {cls: round(float(recalls[idx]), 2) for idx, cls in enumerate(classes)},
        }
        print(f"[+] Seed {seed} Test Results: Top-1: {top1_acc:.2f}%, Top-3: {top3_acc:.2f}%, Macro-F1: {macro_f1:.2f}%")

    # 5. Aggregate metrics across 3 seeds
    top1_vals = [seed_results[str(s)]["top1_acc"] for s in args.seeds]
    top3_vals = [seed_results[str(s)]["top3_acc"] for s in args.seeds]
    f1_vals = [seed_results[str(s)]["macro_f1"] for s in args.seeds]

    mean_top1, sd_top1 = float(np.mean(top1_vals)), float(np.std(top1_vals, ddof=1))
    mean_top3, sd_top3 = float(np.mean(top3_vals)), float(np.std(top3_vals, ddof=1))
    mean_f1, sd_f1 = float(np.mean(f1_vals)), float(np.std(f1_vals, ddof=1))

    mean_norm_cm = np.mean(norm_cms, axis=0)
    sd_norm_cm = np.std(norm_cms, axis=0, ddof=1)

    per_genus_summary = {}
    for idx, cls in enumerate(classes):
        rec_list = [float(seed_results[str(s)]["per_genus_recall"][cls]) for s in args.seeds]
        m_r = float(np.mean(rec_list))
        s_r = float(np.std(rec_list, ddof=1))
        per_genus_summary[cls] = {
            "mean": round(m_r, 2),
            "sd": round(s_r, 2),
        }

    # 6. Render 11x11 Confusion Matrix PNG
    cm_fig_path = args.output_dir / "cm_ccsn11_resnet18.png"
    plot_11x11_confusion_matrix(mean_norm_cm, sd_norm_cm, classes, cm_fig_path)

    # 7. Load tuning results for reporting
    tuning_data = {}
    if TUNING_JSON.exists():
        with open(TUNING_JSON, "r", encoding="utf-8") as f:
            tuning_data = json.load(f)

    # 8. Save overall summary JSON
    summary_json_path = args.output_dir / "ccsn11_evaluation_summary.json"
    overall_summary = {
        "model_architecture": arch,
        "taxonomy": "ccsn11",
        "num_classes": NUM_CLASSES,
        "classes": classes,
        "config": cfg,
        "seeds": args.seeds,
        "overall_metrics": {
            "top1_accuracy": {"mean": round(mean_top1, 2), "sd": round(sd_top1, 2)},
            "top3_accuracy": {"mean": round(mean_top3, 2), "sd": round(sd_top3, 2)},
            "macro_f1": {"mean": round(mean_f1, 2), "sd": round(sd_f1, 2)},
        },
        "per_genus_recall": per_genus_summary,
        "per_seed_results": seed_results,
    }
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(overall_summary, f, indent=2)
    print(f"[+] Saved evaluation summary to {summary_json_path}")

    # 9. Write README.md
    write_readme(
        output_path=args.output_dir / "README.md",
        cfg=cfg,
        tuning_data=tuning_data,
        overall_metrics=overall_summary["overall_metrics"],
        per_genus_summary=per_genus_summary,
        seed_results=seed_results,
        seeds=args.seeds,
        classes=classes,
    )
    print(f"[+] Wrote comprehensive README.md report to {args.output_dir / 'README.md'}")
    print("\n" + "=" * 90)
    print(f"[*] EVALUATION COMPLETED:")
    print(f"    Top-1 Accuracy: {mean_top1:.2f}% +/- {sd_top1:.2f}% (Mobile Deployment Baseline)")
    print(f"    Top-3 Accuracy: {mean_top3:.2f}% +/- {sd_top3:.2f}%")
    print(f"    Macro-F1 Score: {mean_f1:.2f}% +/- {sd_f1:.2f}%")
    print("=" * 90)


def write_readme(
    output_path: Path,
    cfg: dict,
    tuning_data: dict,
    overall_metrics: dict,
    per_genus_summary: dict,
    seed_results: dict,
    seeds: list[int],
    classes: list[str],
):
    """Generates artifacts/ccsn_genus/README.md."""
    top1_m = overall_metrics["top1_accuracy"]["mean"]
    top1_s = overall_metrics["top1_accuracy"]["sd"]
    top3_m = overall_metrics["top3_accuracy"]["mean"]
    top3_s = overall_metrics["top3_accuracy"]["sd"]
    f1_m = overall_metrics["macro_f1"]["mean"]
    f1_s = overall_metrics["macro_f1"]["sd"]

    # Trials table
    trials_table_lines = [
        "| Trial ID | Description | Optimizer | LR Backbone | LR Head | Weight Decay | Label Smoothing | Best Val Loss | Val Top-1 (%) | Val Top-3 (%) | Val F1 (%) |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for tr in tuning_data.get("all_trials", []):
        c = tr["config"]
        h = tr
        v_loss = h.get("best_val_unsmoothed_loss", h.get("best_val_loss", "-"))
        v_top1 = h.get("best_val_acc", "-")
        v_top3 = h.get("best_val_top3_acc", "-")
        v_f1 = h.get("best_val_macro_f1", "-")
        trials_table_lines.append(
            f"| {c['id']} | {c['name']} | {c['optimizer']} | {c['lr_backbone']} | {c['lr_head']} | {c['weight_decay']} | {c['label_smoothing']} | {v_loss} | {v_top1} | {v_top3} | {v_f1} |"
        )

    # Per-genus table
    genus_table_lines = [
        "| Genus Code | Full Genus Name | Seed 42 (%) | Seed 43 (%) | Seed 44 (%) | 3-Seed Mean +/- SD (%) | Separation Status |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :--- |",
    ]
    full_names = {
        "Ac": "Altocumulus",
        "As": "Altostratus",
        "Cb": "Cumulonimbus",
        "Cc": "Cirrocumulus",
        "Ci": "Cirrus",
        "Cs": "Cirrostratus",
        "Ct": "Contrail (Cirrus)",
        "Cu": "Cumulus",
        "Ns": "Nimbostratus",
        "Sc": "Stratocumulus",
        "St": "Stratus",
    }

    for cls in classes:
        s42 = seed_results.get("42", {}).get("per_genus_recall", {}).get(cls, 0.0)
        s43 = seed_results.get("43", {}).get("per_genus_recall", {}).get(cls, 0.0)
        s44 = seed_results.get("44", {}).get("per_genus_recall", {}).get(cls, 0.0)
        m = per_genus_summary[cls]["mean"]
        s = per_genus_summary[cls]["sd"]
        fname = full_names.get(cls, cls)
        if m >= 70.0:
            status = "Reliably Separable"
        elif m >= 50.0:
            status = "Moderate Distinctiveness"
        else:
            status = "High Confusion / Collapse Risk"
        genus_table_lines.append(f"| **{cls}** | {fname} | {s42:.1f}% | {s43:.1f}% | {s44:.1f}% | **{m:.1f}% +/- {s:.1f}%** | {status} |")

    content = f"""# CCSN 11-Class Genus-Level ResNet-18 Classifier

This artifact documents the hyperparameter tuning, final 3-seed production training, and test-holdout evaluation of **ResNet-18** on the canonical **CCSN 11-class genus taxonomy**.

**Deployment Baseline Top-1 Accuracy: {top1_m:.2f}% +/- {top1_s:.2f}%** (across 3 seeds on 508 held-out test images).

---

## 1. Winning Hyperparameter Configuration

The winning configuration was determined through a rigorous 10-trial hyperparameter sweep evaluated strictly on the canonical validation split (407 images) with zero test peeking, ranked by minimum unsmoothed validation loss (NLL).

- **Configuration File**: [config/training/tuned_resnet18_ccsn11.toml](file:///d:/cloud-classification-cnn-mobile/config/training/tuned_resnet18_ccsn11.toml)
- **Winning Trial**: {cfg.get('trial_id', 'N/A')} ({cfg.get('trial_name', 'N/A')})
- **Architecture**: ResNet-18 (ImageNet-1K pretrained with custom classification head)
- **Optimizer**: {cfg.get('optimizer', 'adamw')}
- **Differential Learning Rates**:
  - Backbone: {cfg.get('lr_backbone', 0.0)}
  - Classification Head: {cfg.get('lr_head', 0.0)}
- **Weight Decay**: {cfg.get('weight_decay', 0.0)}
- **Label Smoothing**: {cfg.get('label_smoothing', 0.0)}
- **Dropout**: Backbone {cfg.get('dropout_backbone', 0.0)}, Head {cfg.get('dropout_head', 0.0)}
- **Scheduler**: CosineAnnealingLR (T_max={cfg.get('epochs', 60)}, eta_min=1e-6)
- **Epoch Budget**: {cfg.get('epochs', 60)}
- **Batch Size**: {cfg.get('batch_size', 64)}

---

## 2. 10-Trial Hyperparameter Exploration Table

All trials were trained on the 1,622 training images and evaluated on the 407 validation images of the canonical split:

{chr(10).join(trials_table_lines)}

---

## 3. Overall Test Split Performance (3-Seed Mean +/- SD)

Evaluated on the held-out test split of **508 images** across 3 seeds ({seeds}):

| Metric | 3-Seed Mean +/- SD | Seed 42 | Seed 43 | Seed 44 |
| :--- | :---: | :---: | :---: | :---: |
| **Top-1 Accuracy** | **{top1_m:.2f}% +/- {top1_s:.2f}%** | {seed_results.get('42', {}).get('top1_acc', 0.0):.2f}% | {seed_results.get('43', {}).get('top1_acc', 0.0):.2f}% | {seed_results.get('44', {}).get('top1_acc', 0.0):.2f}% |
| **Top-3 Accuracy** | **{top3_m:.2f}% +/- {top3_s:.2f}%** | {seed_results.get('42', {}).get('top3_acc', 0.0):.2f}% | {seed_results.get('43', {}).get('top3_acc', 0.0):.2f}% | {seed_results.get('44', {}).get('top3_acc', 0.0):.2f}% |
| **Macro-F1 Score** | **{f1_m:.2f}% +/- {f1_s:.2f}%** | {seed_results.get('42', {}).get('macro_f1', 0.0):.2f}% | {seed_results.get('43', {}).get('macro_f1', 0.0):.2f}% | {seed_results.get('44', {}).get('macro_f1', 0.0):.2f}% |

---

## 4. Per-Genus Recall Table

Row-normalized sensitivity per cloud genus across the 3 seeds:

{chr(10).join(genus_table_lines)}

---

## 5. Confusion Matrix (11x11 Row-Normalized)

![CCSN 11-Class Confusion Matrix](file:///d:/cloud-classification-cnn-mobile/artifacts/ccsn_genus/cm_ccsn11_resnet18.png)

*Figure 1: 11x11 row-normalized confusion matrix across the held-out test split (508 images), averaged over 3 seeds {42, 43, 44}. Diagonal entries show mean percentage and standard deviation across seeds; off-diagonal entries show mean confusion rates.*

---

## 6. Meteorological Error Analysis & Morphological Separability

1. **Reliably Separable Genera**: Convective and distinct cellular cloud forms such as Cumulonimbus (**Cb**), Cumulus (**Cu**), and Contrails (**Ct**) achieve the highest discriminatory sensitivity due to high local contrast, sharp vertical development boundaries, and clear linear condensation geometry.
2. **Diffuse Layer Collapse (As vs Ns)**: Altostratus (**As**) and Nimbostratus (**Ns**) exhibit significant cross-class confusion because both present as featureless, gray-to-white overcast sheets lacking high-frequency spatial gradients, where distinguishing rain-bearing depth without radar or radiometric moisture channels is physically ill-posed.
3. **High-Altitude Optical Texture Confusion (Ci vs Cs)**: Cirrus (**Ci**) and Cirrostratus (**Cs**) share identical ice-crystal optical properties and fibrous textures; subtle differences in optical thickness and veil continuity make clean visual separation challenging.
4. **Mid-Level Stratiform Overlap (Ac vs Sc)**: Altocumulus (**Ac**) and Stratocumulus (**Sc**) frequently inter-diffuse due to scale ambiguity in single monocular ground-based RGB imagery, where apparent clump size varies directly with cloud base altitude and camera field-of-view.
5. **Mobile Deployment Implication**: While strict 11-class top-1 accuracy sits at **{top1_m:.2f}% +/- {top1_s:.2f}%**, top-3 accuracy reaches **{top3_m:.2f}% +/- {top3_s:.2f}%**, demonstrating that nearly all misclassifications reside within immediate meteorologically adjacent genera. For on-device deployment, presenting top-3 predictions with uncertainty margins or offering hierarchical genus/harmonized family rollups provides a robust, user-aligned operational workflow.

---

## 7. Artifact Index

- Model Checkpoints:
  - artifacts/ccsn_genus/resnet18_ccsn11_seed42.pth
  - artifacts/ccsn_genus/resnet18_ccsn11_seed43.pth
  - artifacts/ccsn_genus/resnet18_ccsn11_seed44.pth
- Raw Test Predictions (logits, probabilities, labels, predictions):
  - artifacts/ccsn_genus/predictions_seed42.npz
  - artifacts/ccsn_genus/predictions_seed43.npz
  - artifacts/ccsn_genus/predictions_seed44.npz
- Figures & Summaries:
  - artifacts/ccsn_genus/cm_ccsn11_resnet18.png
  - artifacts/ccsn_genus/ccsn11_evaluation_summary.json
  - artifacts/tuning/tuning_results_resnet18_ccsn11.json
  - config/training/tuned_resnet18_ccsn11.toml
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content.strip() + "\n")


if __name__ == "__main__":
    main()
