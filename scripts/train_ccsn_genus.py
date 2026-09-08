# -*- coding: utf-8 -*-
"""
CCSN 11-Class Genus Dedicated Training Runner.

Trains ResNet-18 across 3 seeds {42, 43, 44} on the 11-class canonical CCSN split
using the winning hyperparameter configuration (tuned_resnet18_ccsn11.toml).

STRICT TEST SPLIT ISOLATION:
The 508-image test split is strictly not loaded, preloaded, or referenced during training.
Only train (1,622 images) and validation (407 images) partitions are accessed.
Checkpoints are selected by minimum validation loss and frozen to disk.
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

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import run_harmonized as rh

SEEDS = [42, 43, 44]
NUM_CLASSES = 11
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "ccsn_genus"
CONFIG_PATH = REPO_ROOT / "config" / "training" / "tuned_resnet18_ccsn11.toml"
MANIFEST_PATH = REPO_ROOT / "metadata" / "splits" / "ccsn_11class_canonical.json"


def sha256_file(path: Path) -> str:
    """Computes hex SHA-256 digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def train_ccsn_genus(
    config_path: Path = CONFIG_PATH,
    seeds: list[int] = SEEDS,
    output_dir: Path = ARTIFACTS_DIR,
) -> dict[int, dict]:
    """
    Trains CCSN 11-class genus models across specified seeds with strict test split isolation.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Execution aborted per policy to avoid running on CPU.")

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print(f"[*] CCSN 11-CLASS GENUS DEDICATED TRAINING PIPELINE")
    print(f"[*] Config: {config_path} | Seeds: {seeds} | Output: {output_dir}")
    print(f"[*] Device: {rh.DEVICE} ({torch.cuda.get_device_name(0)})")
    print("=" * 90)

    # 1. Load configuration
    cfg = rh.load_flat_toml_config(config_path)
    print(f"[+] Loaded tuning configuration: {cfg.get('name', 'tuned')} (Trial: {cfg.get('trial_id')})")

    # 2. Load canonical 11-class manifest (strictly train + val ONLY)
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    classes = manifest["classes"]
    assert len(classes) == NUM_CLASSES, f"Expected {NUM_CLASSES} classes, found {len(classes)}"
    print(f"[+] Taxonomy classes ({len(classes)}): {classes}")

    tr_samples = []
    va_samples = []

    for s in manifest["samples"]:
        target_path = rh.CCSN_DIR / s["path"]
        target_label = s["label"]
        split = s["split"]
        if split == "train":
            tr_samples.append((target_path, target_label))
        elif split == "val":
            va_samples.append((target_path, target_label))
        # Zero access to test split during training

    print(f"[+] Isolated Partitions: {len(tr_samples)} train, {len(va_samples)} val (test split strictly excluded)")
    assert len(tr_samples) == 1622, f"Expected 1622 train samples, got {len(tr_samples)}"
    assert len(va_samples) == 407, f"Expected 407 val samples, got {len(va_samples)}"

    # 3. Preload ONLY train and validation image data into RAM
    print("\n--- Preloading Train & Validation Data into RAM ---")
    tr_imgs, tr_lbls = rh.preload_images_parallel_preallocated(tr_samples)
    va_imgs, va_lbls = rh.preload_images_parallel_preallocated(va_samples)

    aug_cfg = cfg.get("augmentation", {})
    train_tf, eval_tf = rh.get_physically_valid_transforms(aug_cfg)

    train_ds = rh.FastCachedCloudDataset(tr_imgs, tr_lbls, transform=train_tf)
    val_ds = rh.FastCachedCloudDataset(va_imgs, va_lbls, transform=eval_tf)

    # 4. Hyperparameters
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

    results = {}

    for seed in seeds:
        print(f"\n" + "#" * 90)
        print(f"# TRAINING RESNET-18 SEED {seed} (Epochs: {epochs}, LS: {ls}, LR: {lr_bb}/{lr_head}, WD: {wd})")
        print("#" * 90)

        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        model, val_summary = rh.train_pool_model(
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
        )

        # Save min-val-loss checkpoint
        ckpt_path = output_dir / f"resnet18_ccsn11_seed{seed}.pth"
        torch.save(model.state_dict(), ckpt_path)
        ckpt_hash = sha256_file(ckpt_path)

        print(f"[+] Saved checkpoint: {ckpt_path}")
        print(f"[+] Checkpoint SHA-256: {ckpt_hash}")
        print(f"[+] Best Validation: Loss={val_summary['best_val_loss']:.4f}, Acc={val_summary['best_val_acc']:.2f}%, F1={val_summary['best_val_macro_f1']:.2f}% (Epoch {val_summary['best_epoch']})")

        results[seed] = {
            "checkpoint_path": str(ckpt_path),
            "checkpoint_sha256": ckpt_hash,
            "val_summary": val_summary,
        }

    print("\n" + "=" * 90)
    print("[*] TRAINING COMPLETE: All seed checkpoints saved and frozen.")
    print("=" * 90)
    for s, res in results.items():
        print(f"  - Seed {s}: {res['checkpoint_sha256']} -> {res['checkpoint_path']}")

    return results


def main():
    parser = argparse.ArgumentParser(description="CCSN 11-Class Genus Training (Test Isolated)")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH, help="Path to TOML config")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS, help="Random seeds to train")
    parser.add_argument("--output-dir", type=Path, default=ARTIFACTS_DIR, help="Output directory for checkpoints")
    args = parser.parse_args()

    train_ccsn_genus(
        config_path=args.config,
        seeds=args.seeds,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()

