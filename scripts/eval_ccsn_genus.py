# -*- coding: utf-8 -*-
"""
CCSN 11-Class Genus Dedicated Evaluation Runner.

Loads frozen checkpoints (resnet18_ccsn11_seed{42,43,44}.pth) and evaluates
strictly on the held-out test split (508 images), generating:
- Raw test predictions (*.npz with logits, labels, probs, y_pred)
- 11x11 row-normalized confusion matrix PNG (cm_ccsn11_resnet18.png)
- Per-genus recall table and overall multi-seed metrics (mean +/- SD)
- Summary JSON (ccsn11_evaluation_summary.json)
- Formatted report in artifacts/ccsn_genus/README.md
"""
from __future__ import annotations

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import hashlib
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


def sha256_file(path: Path) -> str:
    """Computes hex SHA-256 digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


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


def evaluate_ccsn_genus(
    config_path: Path = CONFIG_PATH,
    seeds: list[int] = SEEDS,
    checkpoints_dir: Path = ARTIFACTS_DIR,
    output_dir: Path = ARTIFACTS_DIR,
) -> dict:
    """Evaluates frozen checkpoints on the CCSN held-out test split."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print(f"[*] CCSN 11-CLASS GENUS DEDICATED EVALUATION PIPELINE")
    print(f"[*] Checkpoints: {checkpoints_dir} | Seeds: {seeds} | Output: {output_dir}")
    print("=" * 90)

    cfg = rh.load_flat_toml_config(config_path)
    arch = str(cfg.get("model_architecture", "resnet18"))
    drop_bb = float(cfg.get("dropout_backbone", 0.3))
    drop_head = float(cfg.get("dropout_head", 0.2))
    batch_size = int(cfg.get("batch_size", 64))

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    classes = manifest["classes"]
    print(f"[+] Taxonomy classes ({len(classes)}): {classes}")

    te_samples = []
    for s in manifest["samples"]:
        if s["split"] == "test":
            te_samples.append((rh.CCSN_DIR / s["path"], s["label"]))

    print(f"[+] Evaluation Test Samples: {len(te_samples)}")
    assert len(te_samples) == 508, f"Expected 508 test samples, got {len(te_samples)}"

    print("\n--- Preloading Held-out Test Data into RAM ---")
    te_imgs, te_lbls = rh.preload_images_parallel_preallocated(te_samples)
    _, eval_tf = rh.get_physically_valid_transforms(cfg.get("augmentation", {}))
    test_ds = rh.FastCachedCloudDataset(te_imgs, te_lbls, transform=eval_tf)

    seed_results = {}
    norm_cms = []
    raw_cms = []

    for seed in seeds:
        ckpt_path = checkpoints_dir / f"resnet18_ccsn11_seed{seed}.pth"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing required checkpoint: {ckpt_path}. Run training first.")

        ckpt_sha256 = sha256_file(ckpt_path)
        print(f"\n[*] Evaluating Seed {seed} Checkpoint: {ckpt_path.name} (SHA-256: {ckpt_sha256[:16]}...)")

        model = rh.build_resnet_model(
            model_name=arch,
            num_classes=NUM_CLASSES,
            dropout_head=drop_head,
            dropout_bb=drop_bb,
        ).to(rh.DEVICE)

        state_dict = torch.load(ckpt_path, map_location=rh.DEVICE, weights_only=True)
        model.load_state_dict(state_dict)

        logits, probs, preds, targets = evaluate_test_with_logits(model, test_ds, batch_size=batch_size)

        npz_path = output_dir / f"predictions_seed{seed}.npz"
        np.savez_compressed(
            npz_path,
            logits=logits,
            labels=targets,
            probs=probs,
            y_pred=preds,
            y_true=targets,
        )
        npz_sha256 = sha256_file(npz_path)
        print(f"[+] Saved test predictions to {npz_path.name} (SHA-256: {npz_sha256[:16]}...)")

        top1_acc = (preds == targets).mean() * 100.0
        k_top3 = min(3, logits.shape[1])
        top3_preds = np.argsort(logits, axis=1)[:, -k_top3:]
        top3_acc = np.any(top3_preds == targets[:, None], axis=1).mean() * 100.0
        macro_f1 = f1_score(targets, preds, average="macro", zero_division=0) * 100.0

        cm_raw = confusion_matrix(targets, preds, labels=list(range(NUM_CLASSES)))
        row_sums = cm_raw.sum(axis=1, keepdims=True).astype(float)
        cm_norm = np.divide(cm_raw.astype(float) * 100.0, row_sums, out=np.zeros_like(cm_raw, dtype=float), where=row_sums != 0)

        per_genus_recall = np.diag(cm_norm).tolist()

        seed_results[seed] = {
            "checkpoint_sha256": ckpt_sha256,
            "predictions_sha256": npz_sha256,
            "top1_accuracy": top1_acc,
            "top3_accuracy": top3_acc,
            "macro_f1": macro_f1,
            "per_genus_recall": {cls_name: rec for cls_name, rec in zip(classes, per_genus_recall)},
            "confusion_matrix_raw": cm_raw.tolist(),
            "confusion_matrix_norm": cm_norm.tolist(),
        }

        norm_cms.append(cm_norm)
        raw_cms.append(cm_raw)

        print(f"    - Top-1 Accuracy: {top1_acc:.2f}%")
        print(f"    - Top-3 Accuracy: {top3_acc:.2f}%")
        print(f"    - Macro-F1:       {macro_f1:.2f}%")

    # Multi-seed aggregate statistics
    top1_vals = [res["top1_accuracy"] for res in seed_results.values()]
    top3_vals = [res["top3_accuracy"] for res in seed_results.values()]
    f1_vals = [res["macro_f1"] for res in seed_results.values()]

    mean_top1, sd_top1 = float(np.mean(top1_vals)), float(np.std(top1_vals, ddof=1))
    mean_top3, sd_top3 = float(np.mean(top3_vals)), float(np.std(top3_vals, ddof=1))
    mean_f1, sd_f1 = float(np.mean(f1_vals)), float(np.std(f1_vals, ddof=1))

    mean_cm = np.mean(norm_cms, axis=0)
    sd_cm = np.std(norm_cms, axis=0, ddof=1)
    mean_per_genus_recall = np.diag(mean_cm).tolist()
    sd_per_genus_recall = np.diag(sd_cm).tolist()

    print("\n" + "=" * 90)
    print("[*] MULTI-SEED TEST SET PERFORMANCE (N=508):")
    print(f"    - Top-1 Accuracy: {mean_top1:.2f}% +/- {sd_top1:.2f}%")
    print(f"    - Top-3 Accuracy: {mean_top3:.2f}% +/- {sd_top3:.2f}%")
    print(f"    - Macro-F1:       {mean_f1:.2f}% +/- {sd_f1:.2f}%")
    print("=" * 90)

    # Plot confusion matrix
    cm_path = output_dir / "cm_ccsn11_resnet18.png"
    plot_11x11_confusion_matrix(mean_cm, sd_cm, classes, cm_path)

    # Save summary JSON
    summary_data = {
        "taxonomy": "ccsn11",
        "num_classes": NUM_CLASSES,
        "classes": classes,
        "test_samples": len(te_samples),
        "seeds": seeds,
        "model_architecture": arch,
        "config": cfg,
        "metrics_mean_sd": {
            "top1_accuracy": {"mean": mean_top1, "sd": sd_top1, "formatted": f"{mean_top1:.2f}% +/- {sd_top1:.2f}%"},
            "top3_accuracy": {"mean": mean_top3, "sd": sd_top3, "formatted": f"{mean_top3:.2f}% +/- {sd_top3:.2f}%"},
            "macro_f1": {"mean": mean_f1, "sd": sd_f1, "formatted": f"{mean_f1:.2f}% +/- {sd_f1:.2f}%"},
        },
        "per_genus_recall": {
            cls_name: {"mean": m, "sd": s, "formatted": f"{m:.2f}% +/- {s:.2f}%"}
            for cls_name, m, s in zip(classes, mean_per_genus_recall, sd_per_genus_recall)
        },
        "confusion_matrix_mean": mean_cm.tolist(),
        "confusion_matrix_sd": sd_cm.tolist(),
        "seed_results": {str(s): res for s, res in seed_results.items()},
    }

    summary_json_path = output_dir / "ccsn11_evaluation_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"[+] Saved evaluation summary JSON to {summary_json_path}")

    # Generate README report
    generate_readme_report(summary_data, output_dir / "README.md")
    return summary_data


def generate_readme_report(summary: dict, output_path: Path):
    """Writes standardized markdown report for artifacts/ccsn_genus/README.md."""
    classes = summary["classes"]
    metrics = summary["metrics_mean_sd"]
    recalls = summary["per_genus_recall"]
    seeds = summary["seeds"]
    seed_res = summary["seed_results"]

    # Load 10-trial sweep data if available
    tuning_info = ""
    sweep_table_md = ""
    if TUNING_JSON.exists():
        with open(TUNING_JSON, "r", encoding="utf-8") as f:
            tj = json.load(f)
        trials = tj.get("all_trials", [])
        best_t = tj.get("best_trial", {})
        best_cfg = best_t.get("config", {})
        tuning_info = (
            f"- **Tuning Sweep Trials**: {len(trials)} trials (60 epochs each on Seed {tj.get('seed', 42)})\n"
            f"- **Winning Configuration**: Trial `{best_cfg.get('id', 'N/A')}` ({best_cfg.get('name', 'N/A')})\n"
            f"- **Winning Hyperparameters**: LR Backbone: `{best_cfg.get('lr_backbone')}`, "
            f"LR Head: `{best_cfg.get('lr_head')}`, "
            f"Weight Decay: `{best_cfg.get('weight_decay')}`, "
            f"Label Smoothing: `{best_cfg.get('label_smoothing')}`\n"
            f"- **Validation Performance (Winner)**: Val Loss: `{best_t.get('best_val_loss', 0):.4f}`, "
            f"Top-1 Acc: `{best_t.get('best_val_acc', 0):.2f}%`, "
            f"Top-3 Acc: `{best_t.get('best_val_top3_acc', 0):.2f}%`, "
            f"Macro-F1: `{best_t.get('best_val_macro_f1', 0):.2f}%`\n"
        )
        if trials:
            table_lines = [
                "| Trial ID | Configuration Name | Val Loss | Val Top-1 (%) | Val Top-3 (%) | Val Macro-F1 (%) |",
                "| :--- | :--- | :---: | :---: | :---: | :---: |",
            ]
            for tr in trials:
                c = tr.get("config", {})
                is_win = " **(Winner)**" if c.get("id") == best_cfg.get("id") else ""
                table_lines.append(
                    f"| `{c.get('id')}` | {c.get('name')}{is_win} | {tr.get('best_val_loss', 0):.4f} | "
                    f"{tr.get('best_val_acc', 0):.2f}% | {tr.get('best_val_top3_acc', 0):.2f}% | "
                    f"{tr.get('best_val_macro_f1', 0):.2f}% |"
                )
            sweep_table_md = "\n### 10-Trial Empirical Exploration Summary\n\n" + "\n".join(table_lines) + "\n"

    lines = [
        "# CCSN 11-Class Genus Evaluation Report",
        "",
        "## 1. Executive Summary",
        "",
        f"This report details the final 3-seed production training and evaluation of **ResNet-18** on the canonical **11-class CCSN genus taxonomy** (`metadata/splits/ccsn_11class_canonical.json`).",
        "",
        "The evaluation adheres strictly to hold-out evaluation protocol: models are trained across seeds `{42, 43, 44}` on train (1,622 images) and validation (407 images) partitions, restoring the minimum validation loss checkpoint. The 508-image test partition is strictly isolated and evaluated only once per seed checkpoint.",
        "",
        "> [!NOTE]",
        "> **Note**: These re-audited metrics supersede an earlier contaminated-structure run.",
        "",
        "## 2. Overall Performance Metrics (Held-out Test Split, N=508)",
        "",
        "| Metric | 3-Seed Mean +/- SD | Seed 42 | Seed 43 | Seed 44 |",
        "| :--- | :---: | :---: | :---: | :---: |",
        f"| **Top-1 Accuracy** | **{metrics['top1_accuracy']['formatted']}** | {seed_res['42']['top1_accuracy']:.2f}% | {seed_res['43']['top1_accuracy']:.2f}% | {seed_res['44']['top1_accuracy']:.2f}% |",
        f"| **Top-3 Accuracy** | **{metrics['top3_accuracy']['formatted']}** | {seed_res['42']['top3_accuracy']:.2f}% | {seed_res['43']['top3_accuracy']:.2f}% | {seed_res['44']['top3_accuracy']:.2f}% |",
        f"| **Macro-F1** | **{metrics['macro_f1']['formatted']}** | {seed_res['42']['macro_f1']:.2f}% | {seed_res['43']['macro_f1']:.2f}% | {seed_res['44']['macro_f1']:.2f}% |",
        "",
        "## 3. Hyperparameter Sweep & Tuning Provenance",
        "",
        tuning_info,
        "- **Config TOML**: `config/training/tuned_resnet18_ccsn11.toml`",
        "- **Tuning Trajectory Log**: `artifacts/tuning/tuning_results_resnet18_ccsn11.json`",
        sweep_table_md,
        "",
        "## 4. Per-Genus Sensitivity / Recall Table",
        "",
        "| Genus Code | Cloud Genus Name | Recall (3-Seed Mean +/- SD) | Seed 42 | Seed 43 | Seed 44 |",
        "| :--- | :--- | :---: | :---: | :---: | :---: |",
    ]

    genus_names = {
        "Ac": "Altocumulus",
        "As": "Altostratus",
        "Cb": "Cumulonimbus",
        "Cc": "Cirrocumulus",
        "Ci": "Cirrus",
        "Cs": "Cirrostratus",
        "Ct": "Contraila",
        "Cu": "Cumulus",
        "Ns": "Nimbostratus",
        "Sc": "Stratocumulus",
        "St": "Stratus",
    }

    for cls in classes:
        rec_data = recalls[cls]
        r42 = seed_res["42"]["per_genus_recall"][cls]
        r43 = seed_res["43"]["per_genus_recall"][cls]
        r44 = seed_res["44"]["per_genus_recall"][cls]
        gname = genus_names.get(cls, cls)
        lines.append(f"| `{cls}` | {gname} | **{rec_data['formatted']}** | {r42:.1f}% | {r43:.1f}% | {r44:.1f}% |")

    lines.extend([
        "",
        "## 5. 11x11 Genus Confusion Matrix",
        "",
        "The row-normalized confusion matrix (mean % across 3 seeds with sample standard deviation in parentheses) is rendered below:",
        "",
        "![CCSN 11-Class Genus Confusion Matrix](cm_ccsn11_resnet18.png)",
        "",
        "## 6. Artifact Inventory & Cryptographic Checksums",
        "",
        "| File | Description | SHA-256 Checksum |",
        "| :--- | :--- | :--- |",
    ])

    for s in seeds:
        ckpt_sha = seed_res[str(s)]["checkpoint_sha256"]
        npz_sha = seed_res[str(s)]["predictions_sha256"]
        lines.append(f"| `resnet18_ccsn11_seed{s}.pth` | Frozen model checkpoint (Seed {s}) | `{ckpt_sha}` |")
        lines.append(f"| `predictions_seed{s}.npz` | Raw test predictions & logits (Seed {s}) | `{npz_sha}` |")

    lines.extend([
        f"| `cm_ccsn11_resnet18.png` | 11x11 Confusion Matrix Figure | `{sha256_file(output_path.parent / 'cm_ccsn11_resnet18.png')}` |",
        f"| `ccsn11_evaluation_summary.json` | Complete Machine-readable Evaluation Results | `{sha256_file(output_path.parent / 'ccsn11_evaluation_summary.json')}` |",
        "",
        "## 7. Exact Reproduction Commands",
        "",
        "```powershell",
        "# 1. Run 10-trial hyperparameter sweep (Seed 42, 60 epochs/trial)",
        "python src/tune_resnet_family.py --model resnet18 --taxonomy ccsn11 --seed 42 --epochs 60",
        "",
        "# 2. Train final model across 3 seeds with strict test split isolation",
        "python scripts/train_ccsn_genus.py --config config/training/tuned_resnet18_ccsn11.toml --seeds 42 43 44",
        "",
        "# 3. Evaluate frozen checkpoints on held-out test partition",
        "python scripts/eval_ccsn_genus.py --config config/training/tuned_resnet18_ccsn11.toml --seeds 42 43 44",
        "```",
        "",
    ])

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[+] Written report to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="CCSN 11-Class Genus Evaluation")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH, help="Path to TOML config")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS, help="Random seeds to evaluate")
    parser.add_argument("--checkpoints-dir", type=Path, default=ARTIFACTS_DIR, help="Directory containing .pth files")
    parser.add_argument("--output-dir", type=Path, default=ARTIFACTS_DIR, help="Output directory for predictions and figures")
    args = parser.parse_args()

    evaluate_ccsn_genus(
        config_path=args.config,
        seeds=args.seeds,
        checkpoints_dir=args.checkpoints_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()

