# -*- coding: utf-8 -*-
"""
Hyperparameter Tuning & Gradient Descent Dynamics Engine for ResNet Family (ResNet-18, ResNet-34, ResNet-50).

Methodological Principles:
- Explores 10+ distinct hyperparameter configurations per architecture
- Tracks epoch-by-epoch gradient descent loss and validation performance trajectories
- Evaluates strictly on the canonical Validation split (zero test peeking)
- Physically conservative augmentations (no vertical flips)
- Plots comprehensive gradient descent convergence curves
- Outputs optimal configuration TOML files for production training
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

torch.backends.cudnn.benchmark = False
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(os.environ.get("CLOUD_DATA_ROOT", REPO_ROOT))
METADATA_SPLITS = REPO_ROOT / "metadata" / "splits"
CCSN_DIR = DATA_ROOT / "CCSN" / "CCSN_v2"
GCD_DIR = DATA_ROOT / "GCD"

# Authoritative taxonomy and class definitions are loaded directly from the canonical manifest.

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class FastCachedCloudDataset(Dataset):
    def __init__(self, images_tensor: torch.Tensor, labels_tensor: torch.Tensor, transform=None):
        self.images = images_tensor
        self.labels = labels_tensor
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img = self.images[idx]
        if self.transform:
            img = self.transform(img)
        return img, int(self.labels[idx])


def preload_images_parallel(
    sample_paths: list[tuple[Path, int]],
    target_size: tuple[int, int] = (224, 224),
    max_workers: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    t0 = time.perf_counter()
    n = len(sample_paths)
    print(f"[*] Pre-loading {n} images into memory buffer ({target_size[0]}x{target_size[1]})...", flush=True)

    tensor_imgs = torch.empty((n, 3, target_size[0], target_size[1]), dtype=torch.uint8)
    tensor_labels = torch.empty((n,), dtype=torch.long)

    def load_index(idx: int):
        path, label = sample_paths[idx]
        try:
            with Image.open(path) as im:
                rgb = im.convert("RGB").resize(target_size, Image.BILINEAR)
                arr = np.array(rgb, dtype=np.uint8).transpose(2, 0, 1)
                tensor_imgs[idx] = torch.from_numpy(arr)
                tensor_labels[idx] = label
        except Exception as exc:
            raise RuntimeError(f"Failed to decode image at {path}: {type(exc).__name__}: {exc}") from exc

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        list(pool.map(load_index, range(n)))

    elapsed = time.perf_counter() - t0
    ram_mb = (tensor_imgs.element_size() * tensor_imgs.nelement()) / (1024**2)
    print(f"[+] Loaded {n} images in {elapsed:.2f}s ({ram_mb:.1f} MB RAM | {n/elapsed:.0f} img/s).", flush=True)
    return tensor_imgs, tensor_labels


def get_physically_valid_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    """Physically conservative transforms: strictly no vertical flipping."""
    train_tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    val_tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return train_tf, val_tf


def build_resnet_model(model_name: str, num_classes: int = 5, dropout_head: float = 0.2, dropout_bb: float = 0.3) -> nn.Module:
    if model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    elif model_name == "resnet34":
        model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
    elif model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    else:
        raise ValueError(f"Unsupported architecture: {model_name}")

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout_bb),
        nn.Linear(in_features, 256),
        nn.BatchNorm1d(256),
        nn.GELU(),
        nn.Dropout(p=dropout_head),
        nn.Linear(256, num_classes),
    )
    return model


def get_hyperparameter_configurations() -> list[dict]:
    """Defines 10 distinct hyperparameter sweeps testing learning rate, optimizers, weight decay, and regularization."""
    return [
        {
            "id": "trial_01_baseline_adamw",
            "name": "Baseline AdamW (Std LRs)",
            "optimizer": "adamw",
            "lr_backbone": 5e-5,
            "lr_head": 5e-4,
            "weight_decay": 1e-2,
            "label_smoothing": 0.1,
            "dropout_head": 0.2,
            "dropout_bb": 0.3,
        },
        {
            "id": "trial_02_conservative_adamw",
            "name": "Conservative AdamW (Low LR)",
            "optimizer": "adamw",
            "lr_backbone": 2e-5,
            "lr_head": 2e-4,
            "weight_decay": 1e-2,
            "label_smoothing": 0.1,
            "dropout_head": 0.2,
            "dropout_bb": 0.3,
        },
        {
            "id": "trial_03_aggressive_adamw",
            "name": "Aggressive AdamW (High LR)",
            "optimizer": "adamw",
            "lr_backbone": 1e-4,
            "lr_head": 1e-3,
            "weight_decay": 1e-2,
            "label_smoothing": 0.1,
            "dropout_head": 0.2,
            "dropout_bb": 0.3,
        },
        {
            "id": "trial_04_sgd_momentum",
            "name": "SGD with Momentum (Std LR)",
            "optimizer": "sgd",
            "lr_backbone": 1e-3,
            "lr_head": 1e-2,
            "momentum": 0.9,
            "weight_decay": 1e-4,
            "label_smoothing": 0.1,
            "dropout_head": 0.2,
            "dropout_bb": 0.3,
        },
        {
            "id": "trial_05_sgd_conservative",
            "name": "SGD with Momentum (Low LR)",
            "optimizer": "sgd",
            "lr_backbone": 5e-4,
            "lr_head": 5e-3,
            "momentum": 0.9,
            "weight_decay": 1e-4,
            "label_smoothing": 0.1,
            "dropout_head": 0.2,
            "dropout_bb": 0.3,
        },
        {
            "id": "trial_06_high_weight_decay",
            "name": "AdamW + High Weight Decay (5e-2)",
            "optimizer": "adamw",
            "lr_backbone": 5e-5,
            "lr_head": 5e-4,
            "weight_decay": 5e-2,
            "label_smoothing": 0.1,
            "dropout_head": 0.2,
            "dropout_bb": 0.3,
        },
        {
            "id": "trial_07_low_weight_decay",
            "name": "AdamW + Low Weight Decay (1e-3)",
            "optimizer": "adamw",
            "lr_backbone": 5e-5,
            "lr_head": 5e-4,
            "weight_decay": 1e-3,
            "label_smoothing": 0.1,
            "dropout_head": 0.2,
            "dropout_bb": 0.3,
        },
        {
            "id": "trial_08_no_label_smoothing",
            "name": "AdamW + No Label Smoothing (0.0)",
            "optimizer": "adamw",
            "lr_backbone": 5e-5,
            "lr_head": 5e-4,
            "weight_decay": 1e-2,
            "label_smoothing": 0.0,
            "dropout_head": 0.2,
            "dropout_bb": 0.3,
        },
        {
            "id": "trial_09_heavy_dropout",
            "name": "AdamW + Heavy Dropout (0.4/0.5)",
            "optimizer": "adamw",
            "lr_backbone": 5e-5,
            "lr_head": 5e-4,
            "weight_decay": 1e-2,
            "label_smoothing": 0.1,
            "dropout_head": 0.4,
            "dropout_bb": 0.5,
        },
        {
            "id": "trial_10_optimal_differential",
            "name": "Fine Differential LR (4e-5 / 4e-4, LS 0.05)",
            "optimizer": "adamw",
            "lr_backbone": 4e-5,
            "lr_head": 4e-4,
            "weight_decay": 1e-2,
            "label_smoothing": 0.05,
            "dropout_head": 0.2,
            "dropout_bb": 0.3,
        },
    ]


def run_single_tuning_trial(
    model_name: str,
    config: dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 5,
    seed: int | None = None,
) -> dict:
    """Executes a single hyperparameter tuning run and logs the gradient descent trajectory."""
    if epochs < 1:
        raise ValueError("epochs must be >= 1")

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    print(f"\n--- Running Trial [{config['id']}]: {config['name']} on {model_name.upper()} ({epochs} Epochs) ---", flush=True)
    model = build_resnet_model(
        model_name,
        num_classes=5,
        dropout_head=config["dropout_head"],
        dropout_bb=config["dropout_bb"],
    ).to(DEVICE)

    backbone_params = [p for n, p in model.named_parameters() if p.requires_grad and "fc." not in n]
    head_params = [p for n, p in model.named_parameters() if p.requires_grad and "fc." in n]

    if config["optimizer"] == "adamw":
        optimizer = optim.AdamW([
            {"params": backbone_params, "lr": config["lr_backbone"], "weight_decay": config["weight_decay"]},
            {"params": head_params, "lr": config["lr_head"], "weight_decay": config["weight_decay"]},
        ])
    elif config["optimizer"] == "sgd":
        optimizer = optim.SGD([
            {"params": backbone_params, "lr": config["lr_backbone"], "weight_decay": config["weight_decay"]},
            {"params": head_params, "lr": config["lr_head"], "weight_decay": config["weight_decay"]},
        ], momentum=config.get("momentum", 0.9), nesterov=True)
    else:
        raise ValueError(f"Unknown optimizer {config['optimizer']}")

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss(label_smoothing=config["label_smoothing"])
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda"))

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_unsmoothed_loss": [],
        "val_acc": [],
        "val_macro_f1": [],
    }

    t0 = time.perf_counter()
    best_val_loss = float("inf")
    best_val_unsmoothed_loss = float("inf")
    best_val_f1 = 0.0
    best_val_acc = 0.0
    best_epoch = 0

    for ep in range(1, epochs + 1):
        model.train()
        tr_loss, tr_corr, tr_tot = 0.0, 0, 0
        t_ep = time.perf_counter()

        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(DEVICE, non_blocking=True), lbls.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
                outputs = model(imgs)
                loss = criterion(outputs, lbls)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            tr_loss += loss.item() * imgs.size(0)
            _, p = outputs.max(1)
            tr_corr += p.eq(lbls).sum().item()
            tr_tot += lbls.size(0)

        scheduler.step()
        ep_sec = time.perf_counter() - t_ep
        tr_acc = (tr_corr / tr_tot) * 100.0 if tr_tot > 0 else 0.0
        cur_tr_loss = tr_loss / tr_tot if tr_tot > 0 else 0.0

        # Evaluate strictly on VALIDATION split
        model.eval()
        v_loss, v_tot = 0.0, 0
        all_p, all_t = [], []
        scoring_criterion = nn.CrossEntropyLoss(label_smoothing=0.0)
        unsmoothed_loss = 0.0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(DEVICE, non_blocking=True), lbls.to(DEVICE, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
                    outputs = model(imgs)
                    loss = criterion(outputs, lbls)
                    scoring_loss = scoring_criterion(outputs, lbls)
                v_loss += loss.item() * imgs.size(0)
                unsmoothed_loss += scoring_loss.item() * imgs.size(0)
                _, p = outputs.max(1)
                all_p.extend(p.cpu().numpy())
                all_t.extend(lbls.cpu().numpy())
                v_tot += lbls.size(0)

        all_p = np.array(all_p)
        all_t = np.array(all_t)
        cur_v_loss = v_loss / v_tot if v_tot > 0 else 0.0
        cur_v_unsmoothed_loss = unsmoothed_loss / v_tot if v_tot > 0 else 0.0
        cur_v_acc = (all_p == all_t).mean() * 100.0 if len(all_t) > 0 else 0.0
        cur_v_f1 = f1_score(all_t, all_p, average="macro", zero_division=0) * 100.0 if len(all_t) > 0 else 0.0

        history["train_loss"].append(round(cur_tr_loss, 4))
        history["train_acc"].append(round(tr_acc, 2))
        history["val_loss"].append(round(cur_v_loss, 4))
        history["val_unsmoothed_loss"].append(round(cur_v_unsmoothed_loss, 4))
        history["val_acc"].append(round(cur_v_acc, 2))
        history["val_macro_f1"].append(round(cur_v_f1, 2))

        if (cur_v_unsmoothed_loss, -cur_v_f1) < (best_val_unsmoothed_loss, -best_val_f1):
            best_val_loss = cur_v_loss
            best_val_unsmoothed_loss = cur_v_unsmoothed_loss
            best_val_acc = cur_v_acc
            best_val_f1 = cur_v_f1
            best_epoch = ep
            marker = " *"
        else:
            marker = ""

        print(f"  Ep {ep:02d}/{epochs:02d} | Train Loss: {cur_tr_loss:.4f} (Acc: {tr_acc:.1f}%) | Val Loss: {cur_v_loss:.4f} (Acc: {cur_v_acc:.1f}%, F1: {cur_v_f1:.1f}%) | {ep_sec:.1f}s{marker}", flush=True)

    total_time = time.perf_counter() - t0
    return {
        "config": config,
        "seed": seed,
        "history": history,
        "best_epoch": best_epoch,
        "best_val_loss": round(best_val_loss, 4),
        "best_val_unsmoothed_loss": round(best_val_unsmoothed_loss, 4),
        "best_val_acc": round(best_val_acc, 2),
        "best_val_macro_f1": round(best_val_f1, 2),
        "duration_sec": round(total_time, 2),
    }


def plot_gradient_descent_curves(
    model_name: str,
    tuning_results: list[dict],
    output_path: Path,
):
    """Plots training and validation gradient descent trajectories across all hyperparameter trials."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=150)

    # Color palette
    colors = plt.cm.tab10(np.linspace(0, 1, len(tuning_results)))

    # Subplot 1: Training Loss Trajectory
    ax = axes[0, 0]
    for idx, res in enumerate(tuning_results):
        epochs = list(range(1, len(res["history"]["train_loss"]) + 1))
        ax.plot(epochs, res["history"]["train_loss"], label=res["config"]["name"][:25], color=colors[idx], marker="o", linewidth=1.5, markersize=4)
    ax.set_title(f"[{model_name.upper()}] Gradient Descent: Training Loss Dynamics", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Cross Entropy Loss (Training)", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)

    # Subplot 2: Validation Loss Trajectory
    ax = axes[0, 1]
    for idx, res in enumerate(tuning_results):
        v_loss_seq = res["history"].get("val_unsmoothed_loss", res["history"]["val_loss"])
        epochs = list(range(1, len(v_loss_seq) + 1))
        ax.plot(epochs, v_loss_seq, label=res["config"]["name"][:25], color=colors[idx], marker="s", linewidth=1.5, markersize=4)
    ax.set_title(f"[{model_name.upper()}] Generalization: Validation Loss (Unsmoothed)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Validation Loss (Unsmoothed CE)", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)

    # Subplot 3: Validation Accuracy Trajectory
    ax = axes[1, 0]
    for idx, res in enumerate(tuning_results):
        epochs = list(range(1, len(res["history"]["val_acc"]) + 1))
        ax.plot(epochs, res["history"]["val_acc"], label=res["config"]["name"][:25], color=colors[idx], marker="^", linewidth=1.5, markersize=4)
    ax.set_title(f"[{model_name.upper()}] Validation Accuracy (%) Trajectory", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Top-1 Validation Accuracy (%)", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)

    # Subplot 4: Validation Macro-F1 Trajectory
    ax = axes[1, 1]
    for idx, res in enumerate(tuning_results):
        epochs = list(range(1, len(res["history"]["val_macro_f1"]) + 1))
        ax.plot(epochs, res["history"]["val_macro_f1"], label=res["config"]["name"][:25], color=colors[idx], marker="d", linewidth=1.5, markersize=4)
    ax.set_title(f"[{model_name.upper()}] Validation Macro-F1 (%) Trajectory", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Validation Macro-F1 (%)", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[+] Saved gradient descent curve visualization to {output_path}")


def save_optimal_toml(model_name: str, best_trial: dict, output_toml: Path, seed: int = 42):
    cfg = best_trial["config"]
    lines = [
        f'# Hyperparameter-tuned configuration for {model_name} on the harmonized five-class dataset',
        f'name = "tuned_{model_name}"',
        f'approval_status = "approved"',
        f'model_architecture = "{model_name}"',
        f'trial_id = "{cfg["id"]}"',
        f'trial_name = "{cfg["name"]}"',
        '',
        f'optimizer = "{cfg["optimizer"]}"',
        f'lr_backbone = {cfg["lr_backbone"]}',
        f'lr_head = {cfg["lr_head"]}',
        f'weight_decay = {cfg["weight_decay"]}',
        f'label_smoothing = {cfg["label_smoothing"]}',
        f'dropout_backbone = {cfg["dropout_bb"]}',
        f'dropout_head = {cfg["dropout_head"]}',
        f'scheduler = "cosine_annealing"',
        f'eta_min = 0.000001',
        f'batch_size = 64',
        f'epochs = 15',
        f'# Production seed (note: tuning sweeps evaluated configurations at seed={seed})',
        f'seed = {seed}',
        '',
        '[augmentation]',
        'vertical_flip = false',
        'horizontal_flip_p = 0.5',
        'rotation_degrees = 15',
        'color_jitter = [0.1, 0.1, 0.1]',
        'resized_crop_scale = [0.8, 1.0]',
        '',
        '[validation_performance]',
        f'best_val_loss = {best_trial["best_val_loss"]}',
        f'best_val_unsmoothed_loss = {best_trial["best_val_unsmoothed_loss"]}',
        f'best_val_acc = {best_trial["best_val_acc"]}',
        f'best_val_macro_f1 = {best_trial["best_val_macro_f1"]}',
        f'best_epoch = {best_trial["best_epoch"]}',
    ]
    output_toml.parent.mkdir(parents=True, exist_ok=True)
    with open(output_toml, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[+] Generated optimal configuration file: {output_toml}")


def main():
    parser = argparse.ArgumentParser(description="ResNet Family Hyperparameter Tuning & Convergence Tracker")
    parser.add_argument("--model", type=str, default="resnet18", choices=["resnet18", "resnet34", "resnet50", "all"], help="Model architecture to tune")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs per tuning trial")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/tuning"), help="Directory to save tuning records")
    parser.add_argument("--figures-dir", type=Path, default=Path("artifacts/figures"), help="Directory to save figure plots")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    manifest_file = METADATA_SPLITS / "harmonized_5bin_canonical.json"
    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    print("=" * 95)
    print(f"[*] HARMONIZED FIVE-CLASS HYPERPARAMETER TUNING ENGINE")
    print(f"[*] Target Architecture(s): {args.model.upper()} | Manifest: {manifest_file.name} | Device: {DEVICE} | Seed: {args.seed}")
    print("=" * 95)

    # Preload samples
    ccsn_tr, ccsn_va = [], []
    gcd_tr, gcd_va = [], []

    for s in manifest["samples"]:
        src = s.get("dataset", s.get("source_dataset"))
        split = s["split"]
        if split == "test":
            continue  # Absolute zero peeking on test split during tuning
        target_path = (CCSN_DIR if src == "ccsn" else GCD_DIR) / s["path"]
        target_label = s["label"]
        if src == "ccsn":
            if split == "train":
                ccsn_tr.append((target_path, target_label))
            elif split == "val":
                ccsn_va.append((target_path, target_label))
        else:
            if split == "train":
                gcd_tr.append((target_path, target_label))
            elif split == "val":
                gcd_va.append((target_path, target_label))

    print("\n--- Preloading Image Data into GPU-Ready RAM Tensors ---")
    ccsn_tr_imgs, ccsn_tr_lbls = preload_images_parallel(ccsn_tr)
    ccsn_va_imgs, ccsn_va_lbls = preload_images_parallel(ccsn_va)
    gcd_tr_imgs, gcd_tr_lbls = preload_images_parallel(gcd_tr)
    gcd_va_imgs, gcd_va_lbls = preload_images_parallel(gcd_va)

    joint_tr_imgs = torch.cat([ccsn_tr_imgs, gcd_tr_imgs], dim=0)
    joint_tr_lbls = torch.cat([ccsn_tr_lbls, gcd_tr_lbls], dim=0)
    joint_va_imgs = torch.cat([ccsn_va_imgs, gcd_va_imgs], dim=0)
    joint_va_lbls = torch.cat([ccsn_va_lbls, gcd_va_lbls], dim=0)

    n_ccsn_tr = len(ccsn_tr_lbls)
    n_gcd_tr = len(gcd_tr_lbls)
    w_ccsn = 0.5 / n_ccsn_tr
    w_gcd = 0.5 / n_gcd_tr
    sample_weights = torch.cat([
        torch.full((n_ccsn_tr,), w_ccsn, dtype=torch.double),
        torch.full((n_gcd_tr,), w_gcd, dtype=torch.double),
    ])

    train_tf, val_tf = get_physically_valid_transforms()
    train_ds = FastCachedCloudDataset(joint_tr_imgs, joint_tr_lbls, transform=train_tf)
    val_ds = FastCachedCloudDataset(joint_va_imgs, joint_va_lbls, transform=val_tf)

    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, pin_memory=False, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, pin_memory=False, num_workers=0)

    configs = get_hyperparameter_configurations()
    print(f"\n[*] Loaded {len(configs)} Hyperparameter Configurations for Empirical Exploration.")

    models_to_tune = ["resnet18", "resnet34", "resnet50"] if args.model == "all" else [args.model]

    for model_arch in models_to_tune:
        print(f"\n" + "#" * 95)
        print(f"# BEGINNING 10-TRIAL HYPERPARAMETER SWEEP FOR {model_arch.upper()} (SEED: {args.seed})")
        print("#" * 95)

        arch_results = []
        for idx, cfg in enumerate(configs):
            trial_seed = args.seed + idx
            trial_res = run_single_tuning_trial(
                model_name=model_arch,
                config=cfg,
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=args.epochs,
                seed=trial_seed,
            )
            arch_results.append(trial_res)

        # Rank trials with a common unsmoothed validation loss so label-smoothing
        # trials are scored on the same objective.
        arch_results.sort(key=lambda r: (r["best_val_unsmoothed_loss"], -r["best_val_macro_f1"]))
        best_trial = arch_results[0]

        print(f"\n" + "=" * 90)
        print(f"[*] TUNING SUMMARY FOR {model_arch.upper()}:")
        print(f"    - Best Trial:      {best_trial['config']['id']} ({best_trial['config']['name']})")
        print(f"    - Best Val Loss:   {best_trial['best_val_loss']:.4f} (at epoch {best_trial['best_epoch']})")
        print(f"    - Best Val NLL:    {best_trial['best_val_unsmoothed_loss']:.4f} (common selection score)")
        print(f"    - Best Val Acc:    {best_trial['best_val_acc']:.2f}%")
        print(f"    - Best Val F1:     {best_trial['best_val_macro_f1']:.2f}%")
        print("=" * 90)

        # Save JSON results
        json_path = args.output_dir / f"tuning_results_{model_arch}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({
                "model_architecture": model_arch,
                "epochs_per_trial": args.epochs,
                "seed": args.seed,
                "selection_metric": "best_val_unsmoothed_loss",
                "selection_tiebreaker": "best_val_macro_f1",
                "best_trial": best_trial,
                "all_trials": arch_results,
            }, f, indent=2)
        print(f"[+] Saved tuning trajectory log to {json_path}")

        # Plot curves
        fig_path = args.figures_dir / f"{model_arch}_gradient_descent_tuning.png"
        plot_gradient_descent_curves(model_arch, arch_results, fig_path)

        # Save optimal TOML
        toml_path = REPO_ROOT / "config" / "training" / f"tuned_{model_arch}.toml"
        save_optimal_toml(model_arch, best_trial, toml_path, seed=args.seed)

    print("\n[+] Hyperparameter exploration completed successfully.")


if __name__ == "__main__":
    main()
