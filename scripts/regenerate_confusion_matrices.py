# -*- coding: utf-8 -*-
"""
Regenerate confusion-matrix figures and raw predictions for ResNet-18.

Evaluates minimum-validation-loss checkpoints for ResNet-18 across seeds 42, 43, 44:
  Condition A:  Joint model      -> Pooled 5-class test set (N=3,330)
  Condition B1: CCSN-15 model   -> CCSN test component (N=468)
  Condition B2: Joint model      -> CCSN test component (N=468)

Saves:
  - artifacts/harmonized_thorough/predictions/resnet18_{arm}_seed{S}_{evalset}.npz (y_true, y_pred)
  - artifacts/harmonized_thorough/predictions/cm_resnet18_{arm}_seed{S}_{evalset}.json (raw 5x5 counts)
  - report/figures/cm_joint_pooled_resnet18.png (Condition A, 3-seed mean +/- SD)
  - report/figures/cm_ccsn_indomain_vs_joint_resnet18.png (Condition B1 vs B2, 3-seed mean +/- SD)
"""
from __future__ import annotations

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import run_harmonized as rh

SEEDS = [42, 43, 44]
CLASSES = ["cumulus", "altocumulus", "cirrus", "stratocumulus", "cumulonimbus"]
TICK_LABELS = ["Cu", "Ac", "Ci", "Sc", "Cb"]

MANIFEST_PATH = REPO_ROOT / "metadata" / "splits" / "harmonized_5bin_canonical.json"
MASTER_RESULTS_PATH = REPO_ROOT / "artifacts" / "harmonized_thorough" / "master_multi_seed_results.json"
PREDICTIONS_DIR = REPO_ROOT / "artifacts" / "harmonized_thorough" / "predictions"
FIGURES_DIR = REPO_ROOT / "report" / "figures"


def load_model(ckpt_path: Path, device: torch.device) -> torch.nn.Module:
    """Instantiate ResNet-18 and load checkpoint weights strictly."""
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
    model = rh.build_resnet_model(model_name="resnet18", num_classes=5)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def prepare_datasets() -> tuple[rh.FastCachedCloudDataset, rh.FastCachedCloudDataset, torch.utils.data.ConcatDataset]:
    """Builds identical test datasets and transforms as run_harmonized.py."""
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    ccsn_samples_test = []
    gcd_samples_test = []

    for s in manifest["samples"]:
        if s["split"] != "test":
            continue
        src = s.get("dataset", s.get("source_dataset"))
        target_path = (rh.CCSN_DIR if src == "ccsn" else rh.GCD_DIR) / s["path"]
        target_label = s["label"]
        if src == "ccsn":
            ccsn_samples_test.append((target_path, target_label))
        else:
            gcd_samples_test.append((target_path, target_label))

    print(f"[*] Manifest test partition: {len(ccsn_samples_test)} CCSN samples, {len(gcd_samples_test)} GCD samples.")

    ccsn_te_imgs, ccsn_te_lbls = rh.preload_images_parallel_preallocated(ccsn_samples_test)
    gcd_te_imgs, gcd_te_lbls = rh.preload_images_parallel_preallocated(gcd_samples_test)

    _, val_tf = rh.get_physically_valid_transforms()

    ccsn_te_ds = rh.FastCachedCloudDataset(ccsn_te_imgs, ccsn_te_lbls, transform=val_tf)
    gcd_te_ds = rh.FastCachedCloudDataset(gcd_te_imgs, gcd_te_lbls, transform=val_tf)
    joint_te_ds = torch.utils.data.ConcatDataset([ccsn_te_ds, gcd_te_ds])

    return ccsn_te_ds, gcd_te_ds, joint_te_ds


def evaluate_and_record(
    model: torch.nn.Module,
    ds: torch.utils.data.Dataset,
    arm: str,
    seed: int,
    evalset: str,
    expected_recalls: list[float],
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Runs model inference, writes prediction npz and raw confusion matrix JSON,
    and asserts per-class recall matches recorded master benchmarks within 0.1 pp.
    """
    y_pred, y_true = rh.evaluate_on_dataset(model, ds)

    # Save raw predictions (npz)
    npz_path = PREDICTIONS_DIR / f"resnet18_{arm}_seed{seed}_{evalset}.npz"
    np.savez_compressed(npz_path, y_true=y_true.astype(int), y_pred=y_pred.astype(int))

    # Also save alias if evalset is pooled -> joint
    if evalset == "pooled":
        npz_alias = PREDICTIONS_DIR / f"resnet18_{arm}_seed{seed}_joint.npz"
        np.savez_compressed(npz_alias, y_true=y_true.astype(int), y_pred=y_pred.astype(int))

    # Raw 5x5 confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4])
    json_path = PREDICTIONS_DIR / f"cm_resnet18_{arm}_seed{seed}_{evalset}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(cm.tolist(), f, indent=2)

    if evalset == "pooled":
        json_alias = PREDICTIONS_DIR / f"cm_resnet18_{arm}_seed{seed}_joint.json"
        with open(json_alias, "w", encoding="utf-8") as f:
            json.dump(cm.tolist(), f, indent=2)

    # Row-normalized CM (%)
    row_sums = cm.sum(axis=1, keepdims=True).astype(float)
    cm_norm = np.divide(cm.astype(float) * 100.0, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)
    recalls = np.diag(cm_norm)

    # Cross-check against master results
    diffs = np.abs(recalls - np.array(expected_recalls))
    max_diff = float(np.max(diffs))

    print(f"  [{arm.upper()} | Seed {seed} | {evalset}] Max recall deviation: {max_diff:.4f} pp")
    if max_diff > 0.1:
        raise AssertionError(
            f"FATAL: Recall mismatch exceeds 0.1 pp threshold for {arm} seed {seed} on {evalset}!\n"
            f"Regenerated: {np.round(recalls, 3).tolist()}\n"
            f"Recorded:    {expected_recalls}\n"
            f"Max diff:    {max_diff:.4f} pp"
        )

    return cm, cm_norm, max_diff


def plot_single_cm(mean_cm: np.ndarray, sd_cm: np.ndarray, save_path: Path):
    """Plots a single row-normalized confusion matrix (Condition A)."""
    fig, ax = plt.subplots(figsize=(4.8, 4.0), dpi=300)

    im = ax.imshow(mean_cm, interpolation="nearest", cmap=plt.cm.Blues, vmin=0, vmax=100)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Recall / Distribution (%)", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    ax.set_xticks(range(len(TICK_LABELS)))
    ax.set_yticks(range(len(TICK_LABELS)))
    ax.set_xticklabels(TICK_LABELS, fontsize=10)
    ax.set_yticklabels(TICK_LABELS, fontsize=10)
    ax.set_xlabel("Predicted Class", fontsize=10)
    ax.set_ylabel("True Class", fontsize=10)

    thresh = 50.0
    for i in range(5):
        for j in range(5):
            val = mean_cm[i, j]
            color = "white" if val > thresh else "black"
            if i == j:
                txt = f"{val:.1f}%\n({sd_cm[i, j]:.1f})"
            else:
                txt = f"{val:.1f}%"
            ax.text(j, i, txt, ha="center", va="center", color=color, fontsize=8.5)

    fig.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Saved single-panel figure: {save_path}")


def plot_side_by_side_cm(
    mean_cm_left: np.ndarray,
    sd_cm_left: np.ndarray,
    mean_cm_right: np.ndarray,
    sd_cm_right: np.ndarray,
    save_path: Path,
):
    """Plots side-by-side row-normalized confusion matrices (Condition B1 vs B2)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 4.0), dpi=300)

    im1 = ax1.imshow(mean_cm_left, interpolation="nearest", cmap=plt.cm.Blues, vmin=0, vmax=100)
    im2 = ax2.imshow(mean_cm_right, interpolation="nearest", cmap=plt.cm.Blues, vmin=0, vmax=100)

    cbar = fig.colorbar(im2, ax=[ax1, ax2], fraction=0.025, pad=0.03)
    cbar.set_label("Recall / Distribution (%)", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    for ax, mean_cm, sd_cm, is_left in [(ax1, mean_cm_left, sd_cm_left, True), (ax2, mean_cm_right, sd_cm_right, False)]:
        ax.set_xticks(range(len(TICK_LABELS)))
        ax.set_yticks(range(len(TICK_LABELS)))
        ax.set_xticklabels(TICK_LABELS, fontsize=10)
        ax.set_yticklabels(TICK_LABELS, fontsize=10)
        ax.set_xlabel("Predicted Class", fontsize=10)
        if is_left:
            ax.set_ylabel("True Class", fontsize=10)

        thresh = 50.0
        for i in range(5):
            for j in range(5):
                val = mean_cm[i, j]
                color = "white" if val > thresh else "black"
                if i == j:
                    txt = f"{val:.1f}%\n({sd_cm[i, j]:.1f})"
                else:
                    txt = f"{val:.1f}%"
                ax.text(j, i, txt, ha="center", va="center", color=color, fontsize=8.5)

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Saved two-panel figure: {save_path}")


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[*] Starting confusion matrix regeneration on {device}...")

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    with open(MASTER_RESULTS_PATH, "r", encoding="utf-8") as f:
        master = json.load(f)
    r18_loss = master["resnet18"]["criterion_loss"]

    ccsn_te_ds, gcd_te_ds, joint_te_ds = prepare_datasets()

    norm_cms_A = []
    norm_cms_B1 = []
    norm_cms_B2 = []
    global_max_diff = 0.0

    print("\n--- Running Inference & Cross-Checks ---")
    for s in SEEDS:
        s_str = str(s)
        ccsn_ckpt = REPO_ROOT / f"artifacts/harmonized_thorough/resnet18/ccsn15_seed{s}/ccsn_model_resnet18.pth"
        joint_ckpt = REPO_ROOT / f"artifacts/harmonized_thorough/resnet18/joint_seed{s}/harmonized_joint_resnet18.pth"

        # Load models
        model_ccsn = load_model(ccsn_ckpt, device)
        model_joint = load_model(joint_ckpt, device)

        # Expected recalls from master results
        rec_A = [x["Recall / Sensitivity (%)"] for x in r18_loss["joint"][s_str]["joint_model"]["test_on_joint"]["per_class"]]
        rec_B1 = [x["Recall / Sensitivity (%)"] for x in r18_loss["ccsn15"][s_str]["ccsn_in_domain"]["test_holdout"]["per_class"]]
        rec_B2 = [x["Recall / Sensitivity (%)"] for x in r18_loss["joint"][s_str]["joint_model"]["test_on_ccsn"]["per_class"]]

        # Condition A: Joint checkpoint -> pooled test set (N=3,330)
        _, norm_A, diff_A = evaluate_and_record(model_joint, joint_te_ds, "joint", s, "pooled", rec_A)
        norm_cms_A.append(norm_A)
        global_max_diff = max(global_max_diff, diff_A)

        # Condition B1: CCSN-15 checkpoint -> CCSN test component (N=468)
        _, norm_B1, diff_B1 = evaluate_and_record(model_ccsn, ccsn_te_ds, "ccsn15", s, "ccsn", rec_B1)
        norm_cms_B1.append(norm_B1)
        global_max_diff = max(global_max_diff, diff_B1)

        # Condition B2: Joint checkpoint -> CCSN test component (N=468)
        _, norm_B2, diff_B2 = evaluate_and_record(model_joint, ccsn_te_ds, "joint", s, "ccsn", rec_B2)
        norm_cms_B2.append(norm_B2)
        global_max_diff = max(global_max_diff, diff_B2)

    print(f"\n[+] All 9 inference runs verified successfully!")
    print(f"[+] Global max recall deviation: {global_max_diff:.4f} pp (Threshold <= 0.1000 pp)")

    # Aggregate across 3 seeds (mean and sample SD)
    mean_A = np.mean(norm_cms_A, axis=0)
    sd_A = np.std(norm_cms_A, axis=0, ddof=1)

    mean_B1 = np.mean(norm_cms_B1, axis=0)
    sd_B1 = np.std(norm_cms_B1, axis=0, ddof=1)

    mean_B2 = np.mean(norm_cms_B2, axis=0)
    sd_B2 = np.std(norm_cms_B2, axis=0, ddof=1)

    print("\n--- Generating Publication Figures ---")
    fig_A_path = FIGURES_DIR / "cm_joint_pooled_resnet18.png"
    plot_single_cm(mean_A, sd_A, fig_A_path)

    fig_B_path = FIGURES_DIR / "cm_ccsn_indomain_vs_joint_resnet18.png"
    plot_side_by_side_cm(mean_B1, sd_B1, mean_B2, sd_B2, fig_B_path)

    print("\n[+] Done! Raw predictions saved, matrices cross-checked, and figures generated.")


if __name__ == "__main__":
    main()
