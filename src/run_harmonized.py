# -*- coding: utf-8 -*-
"""
Harmonized cross-source cloud classification and transfer benchmark engine.

Harmonizes ground-based cloud imagery across CCSN (narrow-angle telephoto)
and GCD (all-sky sensor) into a shared five-class compatibility taxonomy:
  1. cumulus       <- CCSN: Cu          | GCD: 1_cumulus
  2. altocumulus   <- CCSN: Ac, Cc      | GCD: 2_altocumulus
  3. cirrus        <- CCSN: Ci, Cs      | GCD: 3_cirrus
  4. stratocumulus <- CCSN: Sc, St, As  | GCD: 5_stratocumulus
  5. cumulonimbus  <- CCSN: Cb, Ns      | GCD: 6_cumulonimbus

Methodological specifications:
- Driven exclusively by canonical manifest: metadata/splits/harmonized_5bin_canonical.json
- Grouped stratified holdout: exact-duplicate clusters stay atomic (StratifiedGroupKFold, seed 42)
- Fixed train (64%) / validation (16%) / test holdout (20%) partition
- Model selection governed strictly by validation loss; test sets evaluated strictly once
- Physically conservative augmentations: strictly no vertical flipping
- Confusion matrices and per-class classification reports
- Non-parametric 95% percentile bootstrap confidence intervals (B=1000)
- Source-balanced batch sampling mitigating GCD volume dominance (85.96% GCD vs 14.04% CCSN)
- Full support for the ResNet family: ResNet-18, ResNet-34, ResNet-50
- Zero-copy virtual dataset concatenation (ConcatDataset) avoiding memory exhaustion
"""
from __future__ import annotations

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
from concurrent.futures import ThreadPoolExecutor
import functools
import gc
import json
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision import models, transforms

from evaluation import generate_evaluation_report
from training_state import snapshot_state_dict_cpu

torch.backends.cudnn.benchmark = False
DEVICE = torch.device("cuda:0")


def worker_init_fn(worker_id: int, seed: int = 0):
    worker_seed = seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def get_worker_count() -> int:
    env_nw = os.environ.get("NUM_WORKERS")
    if env_nw:
        return min(int(env_nw), os.cpu_count() or 1)
    default_nw = 2 if sys.platform == "win32" else 8
    return min(default_nw, os.cpu_count() or 1)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(os.environ.get("CLOUD_DATA_ROOT", REPO_ROOT))
METADATA_SPLITS = REPO_ROOT / "metadata" / "splits"
CCSN_DIR = DATA_ROOT / "CCSN" / "CCSN_v2"
GCD_DIR = DATA_ROOT / "GCD"

HARMONIZED_CLASSES = ["cumulus", "altocumulus", "cirrus", "stratocumulus", "cumulonimbus"]
# Authoritative taxonomy and class definitions are loaded directly from the canonical manifest.

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class FastCachedCloudDataset(Dataset):
    """Memory-resident tensor dataset providing fast random access during training."""

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


def preload_images_parallel_preallocated(
    sample_paths: list[tuple[Path, int]],
    target_size: tuple[int, int] = (224, 224),
    max_workers: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decodes JPEG images into preallocated contiguous RAM tensors using thread pool workers."""
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


def get_physically_valid_transforms(aug_cfg: dict | None = None) -> tuple[transforms.Compose, transforms.Compose]:
    """Physically valid transforms dynamically driven by configuration: strictly no vertical flipping."""
    if aug_cfg is None:
        aug_cfg = {}

    hflip_p = float(aug_cfg.get("horizontal_flip_p", 0.5))
    rot_degrees = float(aug_cfg.get("rotation_degrees", 15))
    crop_scale = tuple(aug_cfg.get("resized_crop_scale", [0.8, 1.0]))
    jitter = aug_cfg.get("color_jitter", [0.1, 0.1, 0.1])
    if isinstance(jitter, (list, tuple)) and len(jitter) == 3:
        b, c, s = jitter
    else:
        b, c, s = 0.1, 0.1, 0.1

    if aug_cfg.get("vertical_flip", False):
        print("[!] Warning: vertical_flip requested in config, but physically invalid for cloud imagery; enforcing vertical_flip = False.", flush=True)

    train_tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(224, scale=crop_scale),
        transforms.RandomHorizontalFlip(p=hflip_p),
        transforms.RandomRotation(degrees=rot_degrees),
        transforms.ColorJitter(brightness=b, contrast=c, saturation=s),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    val_tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return train_tf, val_tf


def build_resnet_model(
    model_name: str = "resnet18",
    num_classes: int = 5,
    dropout_head: float = 0.2,
    dropout_bb: float = 0.3,
) -> nn.Module:
    """Instantiates a ResNet architecture with pretrained weights and customized classification head."""
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


def get_tuned_optimizer(
    model: nn.Module,
    optimizer_type: str = "adamw",
    lr_backbone: float = 5e-5,
    lr_head: float = 5e-4,
    weight_decay: float = 1e-2,
    momentum: float = 0.9,
) -> optim.Optimizer:
    """Constructs an optimizer with layer-differentiated learning rates."""
    backbone_params = [p for n, p in model.named_parameters() if p.requires_grad and "fc." not in n]
    head_params = [p for n, p in model.named_parameters() if p.requires_grad and "fc." in n]

    if optimizer_type == "adamw":
        return optim.AdamW([
            {"params": backbone_params, "lr": lr_backbone, "weight_decay": weight_decay},
            {"params": head_params, "lr": lr_head, "weight_decay": weight_decay},
        ])
    elif optimizer_type == "sgd":
        return optim.SGD([
            {"params": backbone_params, "lr": lr_backbone, "weight_decay": weight_decay},
            {"params": head_params, "lr": lr_head, "weight_decay": weight_decay},
        ], momentum=momentum, nesterov=True)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_type}")


def train_pool_model(
    model_name: str,
    train_ds: Dataset,
    val_ds: Dataset,
    model_arch: str = "resnet18",
    epochs: int = 15,
    batch_size: int = 64,
    lr_backbone: float = 5e-5,
    lr_head: float = 5e-4,
    weight_decay: float = 1e-2,
    label_smoothing: float = 0.0,
    dropout_head: float = 0.2,
    dropout_bb: float = 0.3,
    optimizer_type: str = "adamw",
    sample_weights: torch.Tensor | None = None,
    scheduler_type: str = "cosine_annealing",
    eta_min: float = 1e-6,
    momentum: float = 0.9,
    seed: int = 42,
) -> tuple[nn.Module, nn.Module, dict]:
    """Trains a model for a fixed epoch budget, tracking dual snapshots driven by min val loss and max val macro-F1."""
    if epochs < 1:
        raise ValueError("epochs must be >= 1")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    torch.cuda.empty_cache()

    num_workers = get_worker_count()
    if sys.platform == "win32":
        num_workers = 0
    pref_factor = 2 if sys.platform == "win32" else 4
    generator = torch.Generator().manual_seed(seed)
    init_fn = functools.partial(worker_init_fn, seed=seed)

    pin_mem = (num_workers > 0)
    if sample_weights is not None:
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
            generator=generator,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=sampler,
            pin_memory=pin_mem,
            num_workers=num_workers,
            persistent_workers=(num_workers > 0),
            prefetch_factor=pref_factor if num_workers > 0 else None,
            worker_init_fn=init_fn,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=pin_mem,
            num_workers=num_workers,
            persistent_workers=(num_workers > 0),
            prefetch_factor=pref_factor if num_workers > 0 else None,
            generator=generator,
            worker_init_fn=init_fn,
        )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=False,
        num_workers=0,
    )

    model = build_resnet_model(model_name=model_arch, num_classes=5, dropout_head=dropout_head, dropout_bb=dropout_bb).to(DEVICE)
    optimizer = get_tuned_optimizer(
        model,
        optimizer_type=optimizer_type,
        lr_backbone=lr_backbone,
        lr_head=lr_head,
        weight_decay=weight_decay,
        momentum=momentum,
    )
    if scheduler_type == "cosine_annealing":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=eta_min)
    elif scheduler_type == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=max(1, epochs // 3), gamma=0.1)
    elif scheduler_type in ("none", "constant"):
        scheduler = None
    else:
        raise ValueError(
            f"Unsupported scheduler '{scheduler_type}'. "
            "Supported values are: cosine_annealing, step, none, constant."
        )

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda"))

    best_val_loss = float("inf")
    best_val_loss_acc = 0.0
    best_val_loss_f1 = 0.0
    best_epoch_loss = 0
    best_state_loss = None

    best_val_f1 = -1.0
    best_val_f1_loss = float("inf")
    best_val_f1_acc = 0.0
    best_epoch_f1 = 0
    best_state_f1 = None

    print(f"\n[*] Training {model_name} ({model_arch.upper()}) on {len(train_ds)} images (Val: {len(val_ds)}) for {epochs} epochs (Batch: {batch_size}, Workers: {num_workers})...", flush=True)

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

        if scheduler is not None:
            scheduler.step()
        ep_sec = time.perf_counter() - t_ep
        tr_acc = (tr_corr / tr_tot) * 100.0 if tr_tot > 0 else 0.0

        # Evaluate strictly on validation split
        model.eval()
        v_loss, v_tot = 0.0, 0
        all_p, all_t = [], []
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(DEVICE, non_blocking=True), lbls.to(DEVICE, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
                    outputs = model(imgs)
                    loss = criterion(outputs, lbls)
                v_loss += loss.item() * imgs.size(0)
                _, p = outputs.max(1)
                all_p.extend(p.cpu().numpy())
                all_t.extend(lbls.cpu().numpy())
                v_tot += lbls.size(0)

        all_p = np.array(all_p)
        all_t = np.array(all_t)
        cur_v_loss = v_loss / v_tot if v_tot > 0 else 0.0
        cur_v_acc = (all_p == all_t).mean() * 100.0 if len(all_t) > 0 else 0.0
        cur_v_f1 = f1_score(all_t, all_p, average="macro", zero_division=0) * 100.0 if len(all_t) > 0 else 0.0

        marker_loss = ""
        marker_f1 = ""

        if cur_v_loss < best_val_loss:
            best_val_loss = cur_v_loss
            best_val_loss_acc = cur_v_acc
            best_val_loss_f1 = cur_v_f1
            best_epoch_loss = ep
            best_state_loss = snapshot_state_dict_cpu(model)
            marker_loss = " [BEST LOSS]"

        if cur_v_f1 > best_val_f1:
            best_val_f1 = cur_v_f1
            best_val_f1_loss = cur_v_loss
            best_val_f1_acc = cur_v_acc
            best_epoch_f1 = ep
            best_state_f1 = snapshot_state_dict_cpu(model)
            marker_f1 = " [BEST F1]"

        marker = f"{marker_loss}{marker_f1}"
        speed = len(train_ds) / ep_sec if ep_sec > 0 else 0.0
        print(f"  [{model_name}] Epoch {ep:02d}/{epochs:02d} | Train Loss: {tr_loss/tr_tot:.4f} (Acc: {tr_acc:.2f}%) | Val Loss: {cur_v_loss:.4f} (Acc: {cur_v_acc:.2f}%, F1: {cur_v_f1:.2f}%) | Speed: {speed:.0f} img/s{marker}", flush=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    disagreement = (best_epoch_loss != best_epoch_f1)
    print(f"  [SELECTION] Loss-selected epoch: {best_epoch_loss} (loss: {best_val_loss:.4f}, f1: {best_val_loss_f1:.2f}%) | F1-selected epoch: {best_epoch_f1} (loss: {best_val_f1_loss:.4f}, f1: {best_val_f1:.2f}%)", flush=True)
    print(f"  [SELECTION] Disagreement: {disagreement}", flush=True)

    val_summary = {
        "best_epoch": best_epoch_loss,
        "best_val_loss": round(best_val_loss, 4),
        "best_val_acc": round(best_val_loss_acc, 2),
        "best_val_macro_f1": round(best_val_loss_f1, 2),
        "criterion_loss": {
            "best_epoch": best_epoch_loss,
            "best_val_loss": round(best_val_loss, 4),
            "best_val_acc": round(best_val_loss_acc, 2),
            "best_val_macro_f1": round(best_val_loss_f1, 2),
        },
        "criterion_f1": {
            "best_epoch": best_epoch_f1,
            "best_val_loss": round(best_val_f1_loss, 4),
            "best_val_acc": round(best_val_f1_acc, 2),
            "best_val_macro_f1": round(best_val_f1, 2),
        },
        "disagreement": disagreement,
    }

    model_loss = build_resnet_model(model_name=model_arch, num_classes=5, dropout_head=dropout_head, dropout_bb=dropout_bb).to(DEVICE)
    model_loss.load_state_dict({k: v.to(DEVICE) for k, v in best_state_loss.items()})

    model_f1 = build_resnet_model(model_name=model_arch, num_classes=5, dropout_head=dropout_head, dropout_bb=dropout_bb).to(DEVICE)
    model_f1.load_state_dict({k: v.to(DEVICE) for k, v in best_state_f1.items()})

    torch.cuda.empty_cache()
    return model_loss, model_f1, val_summary


@torch.no_grad()
def evaluate_on_dataset(model: nn.Module, ds: Dataset, batch_size: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Evaluates model strictly once on an untouched test holdout."""
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=False,
        num_workers=0,
    )
    model.eval()

    all_p, all_t = [], []
    for batch_imgs, batch_lbls in loader:
        batch_imgs = batch_imgs.to(DEVICE, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
            outputs = model(batch_imgs)
        _, p = outputs.max(1)
        all_p.extend(p.cpu().numpy())
        all_t.extend(batch_lbls.numpy())

    del loader
    gc.collect()
    return np.array(all_p), np.array(all_t)


def load_flat_toml_config(path: Path) -> dict:
    """Load a flat TOML file, accepting a UTF-8 BOM and failing on parse errors."""
    import tomllib

    try:
        return tomllib.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        raise RuntimeError(f"Failed to parse configuration file {path}: {exc}") from exc


def seed_everything(seed_val: int):
    torch.manual_seed(seed_val)
    np.random.seed(seed_val)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_val)


def get_checkpoint_meta_path(ckpt_path: Path) -> Path:
    return ckpt_path.with_name(ckpt_path.stem + ".meta.json")


def load_checkpoint_metadata(ckpt_path: Path) -> dict | None:
    meta_path = get_checkpoint_meta_path(ckpt_path)
    if meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def save_checkpoint_metadata(ckpt_path: Path, metadata: dict):
    meta_path = get_checkpoint_meta_path(ckpt_path)
    try:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
    except Exception as e:
        print(f"[!] Notice: unable to write checkpoint metadata: {e}", flush=True)


def validate_checkpoint_provenance(
    ckpt_meta: dict | None,
    requested_config: dict,
    ckpt_name: str,
    allow_unverified: bool = False,
):
    if not ckpt_meta:
        msg = f"Checkpoint '{ckpt_name}' lacks provenance metadata sidecar (*.meta.json)."
        if not allow_unverified:
            raise RuntimeError(f"{msg} Pass --allow-unverified-checkpoints to override.")
        print(f"[!] Warning: {msg} Continuing due to --allow-unverified-checkpoints.", flush=True)
        return

    mismatches = []
    for k, req_v in requested_config.items():
        if k in ckpt_meta and ckpt_meta[k] != req_v:
            mismatches.append(f"{k} (checkpoint: {ckpt_meta[k]} vs requested: {req_v})")
    if mismatches:
        msg = f"Checkpoint '{ckpt_name}' provenance mismatch: {', '.join(mismatches)}."
        if not allow_unverified:
            raise RuntimeError(f"{msg} Pass --allow-unverified-checkpoints to override.")
        print(f"[!] Warning: {msg} Reusing anyway due to --allow-unverified-checkpoints.", flush=True)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Execution aborted per policy to avoid running on CPU.")

    parser = argparse.ArgumentParser(description="Harmonized Cross-Source Cloud Classification")
    parser.add_argument("--model", type=str, default=None, choices=["resnet18", "resnet34", "resnet50"], help="Backbone architecture")
    parser.add_argument("--config", type=Path, default=None, help="Path to TOML configuration file")
    parser.add_argument("--epochs", type=int, default=None, help="Epochs per pool")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--experiment", type=str, default="all", choices=["all", "ccsn", "gcd", "joint"], help="Experiment to run: all, ccsn, gcd, or joint")
    parser.add_argument("--reuse-checkpoints", action="store_true", help="Reuse existing saved checkpoints if available")
    parser.add_argument("--allow-unverified-checkpoints", action="store_true", help="Allow reusing checkpoints without *.meta.json provenance sidecars or with configuration mismatches")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/harmonized_results"), help="Metrics and model weights output directory")
    parser.add_argument("--figures-dir", type=Path, default=Path("artifacts/figures"), help="Figures output directory")
    args = parser.parse_args()

    model_arch = "resnet18"
    epochs = 15
    batch_size = 64
    seed = 42
    lr_bb = 5e-5
    lr_hd = 5e-4
    wd = 1e-2
    ls = 0.0
    drop_hd = 0.2
    drop_bb = 0.3
    opt_type = "adamw"
    sched_type = "cosine_annealing"
    eta_min = 1e-6
    momentum = 0.9
    aug_cfg = {}

    if args.config:
        if not args.config.exists():
            raise FileNotFoundError(f"Configuration file not found: {args.config}")
        cfg = load_flat_toml_config(args.config)
        if "epochs" in cfg:
            epochs = int(cfg["epochs"])
        if "batch_size" in cfg:
            batch_size = int(cfg["batch_size"])
        if "seed" in cfg:
            seed = int(cfg["seed"])
        if "model_architecture" in cfg:
            model_arch = cfg["model_architecture"]
        if "lr_backbone" in cfg:
            lr_bb = float(cfg["lr_backbone"])
        if "lr_head" in cfg:
            lr_hd = float(cfg["lr_head"])
        if "weight_decay" in cfg:
            wd = float(cfg["weight_decay"])
        if "label_smoothing" in cfg:
            ls = float(cfg["label_smoothing"])
        if "dropout_head" in cfg:
            drop_hd = float(cfg["dropout_head"])
        if "dropout_backbone" in cfg:
            drop_bb = float(cfg["dropout_backbone"])
        if "optimizer" in cfg:
            opt_type = cfg["optimizer"]
        if "scheduler" in cfg:
            sched_type = cfg["scheduler"]
        if "eta_min" in cfg:
            eta_min = float(cfg["eta_min"])
        if "momentum" in cfg:
            momentum = float(cfg["momentum"])
        if "augmentation" in cfg and isinstance(cfg["augmentation"], dict):
            aug_cfg = cfg["augmentation"]

    if args.model is not None:
        model_arch = args.model
    if args.epochs is not None:
        epochs = args.epochs
    if args.batch_size is not None:
        batch_size = args.batch_size
    if args.seed is not None:
        seed = args.seed

    if args.config:
        print(f"[*] Loaded configuration from {args.config} (Model: {model_arch}, Epochs: {epochs}, Batch: {batch_size}, Seed: {seed})", flush=True)

    seed_everything(seed)

    args.figures_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest_file = METADATA_SPLITS / "harmonized_5bin_canonical.json"
    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    class_names = manifest["classes"]

    worker_count = get_worker_count()
    compute_env = {
        "cuda_available": True,
        "device_name": torch.cuda.get_device_name(0),
        "device_count": torch.cuda.device_count(),
        "current_device": torch.cuda.current_device(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cpu_count": os.cpu_count(),
        "worker_count": worker_count,
    }

    print("=" * 95, flush=True)
    print(f"[*] CANONICAL HARMONIZED BENCHMARK: {manifest['taxonomy_name']}", flush=True)
    print(f"[*] Architecture: {model_arch.upper()} | Samples: {len(manifest['samples'])} | Classes: {manifest['num_classes']} | Compute: {DEVICE}", flush=True)
    print(f"[*] Compute Environment: GPU={compute_env['device_name']} (Device 0 of {compute_env['device_count']}) | CUDA={compute_env['cuda_version']} | Workers={compute_env['worker_count']}", flush=True)
    print("=" * 95, flush=True)

    compute_env_file = REPO_ROOT / "artifacts" / "harmonized_thorough" / "compute_environment.json"
    compute_env_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(compute_env_file, "w", encoding="utf-8") as f:
            json.dump(compute_env, f, indent=2)
    except Exception as e:
        print(f"[!] Warning saving compute_environment.json: {e}", flush=True)

    # Separate samples by source dataset and split
    ccsn_samples_by_split = {"train": [], "val": [], "test": []}
    gcd_samples_by_split = {"train": [], "val": [], "test": []}

    for s in manifest["samples"]:
        src = s.get("dataset", s.get("source_dataset"))
        split = s["split"]
        target_path = (CCSN_DIR if src == "ccsn" else GCD_DIR) / s["path"]
        target_label = s["label"]
        if src == "ccsn":
            ccsn_samples_by_split[split].append((target_path, target_label))
        else:
            gcd_samples_by_split[split].append((target_path, target_label))

    need_ccsn_tr = args.experiment in ("all", "ccsn", "joint")
    need_ccsn_va = args.experiment in ("all", "ccsn", "joint")
    need_gcd_tr = args.experiment in ("all", "gcd", "joint")
    need_gcd_va = args.experiment in ("all", "gcd", "joint")

    # Preload raw images into RAM tensors
    print("\n--- Preloading CCSN Partitions ---", flush=True)
    if need_ccsn_tr:
        ccsn_tr_imgs, ccsn_tr_lbls = preload_images_parallel_preallocated(ccsn_samples_by_split["train"])
    else:
        ccsn_tr_imgs = torch.empty((0, 3, 224, 224), dtype=torch.uint8)
        ccsn_tr_lbls = torch.empty((0,), dtype=torch.long)

    if need_ccsn_va:
        ccsn_va_imgs, ccsn_va_lbls = preload_images_parallel_preallocated(ccsn_samples_by_split["val"])
    else:
        ccsn_va_imgs = torch.empty((0, 3, 224, 224), dtype=torch.uint8)
        ccsn_va_lbls = torch.empty((0,), dtype=torch.long)

    ccsn_te_imgs, ccsn_te_lbls = preload_images_parallel_preallocated(ccsn_samples_by_split["test"])

    print("\n--- Preloading GCD Partitions ---", flush=True)
    if need_gcd_tr:
        gcd_tr_imgs, gcd_tr_lbls = preload_images_parallel_preallocated(gcd_samples_by_split["train"])
    else:
        gcd_tr_imgs = torch.empty((0, 3, 224, 224), dtype=torch.uint8)
        gcd_tr_lbls = torch.empty((0,), dtype=torch.long)

    if need_gcd_va:
        gcd_va_imgs, gcd_va_lbls = preload_images_parallel_preallocated(gcd_samples_by_split["val"])
    else:
        gcd_va_imgs = torch.empty((0, 3, 224, 224), dtype=torch.uint8)
        gcd_va_lbls = torch.empty((0,), dtype=torch.long)

    gcd_te_imgs, gcd_te_lbls = preload_images_parallel_preallocated(gcd_samples_by_split["test"])

    train_tf, val_tf = get_physically_valid_transforms(aug_cfg)

    # Datasets (Zero-copy ConcatDataset)
    ccsn_tr_ds = FastCachedCloudDataset(ccsn_tr_imgs, ccsn_tr_lbls, transform=train_tf)
    ccsn_va_ds = FastCachedCloudDataset(ccsn_va_imgs, ccsn_va_lbls, transform=val_tf)
    ccsn_te_ds = FastCachedCloudDataset(ccsn_te_imgs, ccsn_te_lbls, transform=val_tf)

    gcd_tr_ds = FastCachedCloudDataset(gcd_tr_imgs, gcd_tr_lbls, transform=train_tf)
    gcd_va_ds = FastCachedCloudDataset(gcd_va_imgs, gcd_va_lbls, transform=val_tf)
    gcd_te_ds = FastCachedCloudDataset(gcd_te_imgs, gcd_te_lbls, transform=val_tf)

    joint_tr_ds = ConcatDataset([ccsn_tr_ds, gcd_tr_ds])
    joint_va_ds = ConcatDataset([ccsn_va_ds, gcd_va_ds])
    joint_te_ds = ConcatDataset([ccsn_te_ds, gcd_te_ds])

    print(f"\n[*] CCSN -> Train: {len(ccsn_tr_ds)} | Val: {len(ccsn_va_ds)} | Test: {len(ccsn_te_ds)}", flush=True)
    print(f"[*] GCD  -> Train: {len(gcd_tr_ds)} | Val: {len(gcd_va_ds)} | Test: {len(gcd_te_ds)}", flush=True)
    print(f"[*] Joint -> Train: {len(joint_tr_ds)} | Val: {len(joint_va_ds)} | Test: {len(joint_te_ds)}", flush=True)

    summary_file = args.output_dir / f"harmonized_summary_{model_arch}.json"
    results = {}
    if args.reuse_checkpoints and summary_file.exists():
        try:
            with open(summary_file, "r", encoding="utf-8") as f:
                results = json.load(f)
        except Exception as exc:
            raise RuntimeError(f"Failed to read existing summary file {summary_file}: {exc}") from exc

    ccsn_ckpt = args.output_dir / f"ccsn_model_{model_arch}.pth"
    ccsn_ckpt_f1 = args.output_dir / f"ccsn_model_{model_arch}_best_f1.pth"
    gcd_ckpt = args.output_dir / f"gcd_model_{model_arch}.pth"
    gcd_ckpt_f1 = args.output_dir / f"gcd_model_{model_arch}_best_f1.pth"
    joint_ckpt = args.output_dir / f"harmonized_joint_{model_arch}.pth"
    joint_ckpt_f1 = args.output_dir / f"harmonized_joint_{model_arch}_best_f1.pth"

    # =========================================================================
    # EXPERIMENT 1: CCSN IN-DOMAIN MODEL & CROSS-SOURCE TRANSFER TO GCD
    # =========================================================================
    if args.experiment in ("all", "ccsn"):
        seed_everything(seed)
        print("\n" + "=" * 90, flush=True)
        print(f"EXPERIMENT 1: CCSN IN-DOMAIN TRAINING & CROSS-SOURCE TRANSFER TO GCD ({model_arch.upper()})", flush=True)
        print("=" * 90, flush=True)
        ccsn_ckpt_meta = load_checkpoint_metadata(ccsn_ckpt)
        ccsn_ckpt_f1_meta = load_checkpoint_metadata(ccsn_ckpt_f1)
        if args.reuse_checkpoints and ccsn_ckpt.exists() and ccsn_ckpt_f1.exists():
            print(f"[*] Reusing checkpoints: {ccsn_ckpt} and {ccsn_ckpt_f1}", flush=True)
            validate_checkpoint_provenance(
                ccsn_ckpt_meta, {
                    "model_architecture": model_arch,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "seed": seed,
                },
                ccsn_ckpt.name,
                allow_unverified=args.allow_unverified_checkpoints,
            )
            model_ccsn_loss = build_resnet_model(model_name=model_arch, num_classes=5, dropout_head=drop_hd, dropout_bb=drop_bb).to(DEVICE)
            model_ccsn_loss.load_state_dict(torch.load(ccsn_ckpt, map_location=DEVICE))
            model_ccsn_f1 = build_resnet_model(model_name=model_arch, num_classes=5, dropout_head=drop_hd, dropout_bb=drop_bb).to(DEVICE)
            model_ccsn_f1.load_state_dict(torch.load(ccsn_ckpt_f1, map_location=DEVICE))
            val_ccsn = results.get("ccsn_in_domain", {}).get("val_metrics", {"reused": True})
        else:
            model_ccsn_loss, model_ccsn_f1, val_ccsn = train_pool_model(
                "CCSN_Model", ccsn_tr_ds, ccsn_va_ds,
                model_arch=model_arch, epochs=epochs, batch_size=batch_size,
                lr_backbone=lr_bb, lr_head=lr_hd, weight_decay=wd, label_smoothing=ls,
                dropout_head=drop_hd, dropout_bb=drop_bb, optimizer_type=opt_type,
                scheduler_type=sched_type, eta_min=eta_min, momentum=momentum,
                seed=seed,
            )
            torch.save(model_ccsn_loss.state_dict(), ccsn_ckpt)
            torch.save(model_ccsn_f1.state_dict(), ccsn_ckpt_f1)
            common_meta = {
                "model_architecture": model_arch,
                "experiment_condition": "ccsn_in_domain",
                "epochs": epochs,
                "batch_size": batch_size,
                "seed": seed,
                "optimizer": opt_type,
                "lr_backbone": lr_bb,
                "lr_head": lr_hd,
                "weight_decay": wd,
                "label_smoothing": ls,
                "dropout_backbone": drop_bb,
                "dropout_head": drop_hd,
                "scheduler": sched_type,
                "eta_min": eta_min,
                "manifest_path": str(manifest_file),
                "augmentation": {
                    "vertical_flip": False,
                    "horizontal_flip_p": aug_cfg.get("horizontal_flip_p", 0.5),
                    "rotation_degrees": aug_cfg.get("rotation_degrees", 15),
                    "color_jitter": aug_cfg.get("color_jitter", [0.1, 0.1, 0.1]),
                    "resized_crop_scale": aug_cfg.get("resized_crop_scale", [0.8, 1.0]),
                },
            }
            save_checkpoint_metadata(ccsn_ckpt, {
                **common_meta,
                "checkpoint_file": ccsn_ckpt.name,
                "selection_criterion": "min_val_loss",
                "selected_epoch": val_ccsn["criterion_loss"]["best_epoch"],
                "val_metrics": val_ccsn["criterion_loss"],
            })
            save_checkpoint_metadata(ccsn_ckpt_f1, {
                **common_meta,
                "checkpoint_file": ccsn_ckpt_f1.name,
                "selection_criterion": "max_val_macro_f1",
                "selected_epoch": val_ccsn["criterion_f1"]["best_epoch"],
                "val_metrics": val_ccsn["criterion_f1"],
            })
            ccsn_ckpt_meta = load_checkpoint_metadata(ccsn_ckpt)
            ccsn_ckpt_f1_meta = load_checkpoint_metadata(ccsn_ckpt_f1)

        # 1A. CCSN In-Domain Test Evaluation (Loss)
        p_ccsn_in_loss, t_ccsn_in = evaluate_on_dataset(model_ccsn_loss, ccsn_te_ds, batch_size=batch_size)
        rep_ccsn_in_loss = generate_evaluation_report(
            t_ccsn_in, p_ccsn_in_loss, HARMONIZED_CLASSES,
            dataset_name=f"Harmonized CCSN In-Domain (Five-Class, {model_arch.upper()})",
            model_name=model_arch,
            output_dir=args.figures_dir,
            title_suffix=f"CCSN In-Domain Holdout ({model_arch.upper()}, Loss)",
        )

        # 1B. Cross-Source Transfer: CCSN Model on GCD Test Holdout (Loss)
        p_ccsn_on_gcd_loss, t_gcd_te = evaluate_on_dataset(model_ccsn_loss, gcd_te_ds, batch_size=batch_size)
        rep_ccsn_on_gcd_loss = generate_evaluation_report(
            t_gcd_te, p_ccsn_on_gcd_loss, HARMONIZED_CLASSES,
            dataset_name=f"Cross-Source CCSN to GCD (Five-Class, {model_arch.upper()})",
            model_name=model_arch,
            output_dir=args.figures_dir,
            title_suffix=f"CCSN Model -> GCD Holdout ({model_arch.upper()}, Loss)",
        )

        # 1A-F1. CCSN In-Domain Test Evaluation (F1)
        p_ccsn_in_f1, _ = evaluate_on_dataset(model_ccsn_f1, ccsn_te_ds, batch_size=batch_size)
        rep_ccsn_in_f1 = generate_evaluation_report(
            t_ccsn_in, p_ccsn_in_f1, HARMONIZED_CLASSES,
            dataset_name=f"Harmonized CCSN In-Domain (Five-Class, {model_arch.upper()})",
            model_name=f"{model_arch}_best_f1",
            output_dir=args.figures_dir,
            title_suffix=f"CCSN In-Domain Holdout ({model_arch.upper()}, F1)",
        )

        # 1B-F1. Cross-Source Transfer: CCSN Model on GCD Test Holdout (F1)
        p_ccsn_on_gcd_f1, _ = evaluate_on_dataset(model_ccsn_f1, gcd_te_ds, batch_size=batch_size)
        rep_ccsn_on_gcd_f1 = generate_evaluation_report(
            t_gcd_te, p_ccsn_on_gcd_f1, HARMONIZED_CLASSES,
            dataset_name=f"Cross-Source CCSN to GCD (Five-Class, {model_arch.upper()})",
            model_name=f"{model_arch}_best_f1",
            output_dir=args.figures_dir,
            title_suffix=f"CCSN Model -> GCD Holdout ({model_arch.upper()}, F1)",
        )

        results.setdefault("criterion_loss", {})
        results.setdefault("criterion_f1", {})

        results["criterion_loss"]["ccsn_in_domain"] = {
            "selected_epoch": val_ccsn["criterion_loss"]["best_epoch"],
            "val_metrics": val_ccsn["criterion_loss"],
            "test_holdout": rep_ccsn_in_loss["metrics_summary"],
            "checkpoint_reused": bool(args.reuse_checkpoints and ccsn_ckpt.exists()),
            "checkpoint_metadata": ccsn_ckpt_meta,
        }
        results["criterion_loss"]["cross_source_ccsn_to_gcd"] = {
            "test_holdout": rep_ccsn_on_gcd_loss["metrics_summary"],
        }

        results["criterion_f1"]["ccsn_in_domain"] = {
            "selected_epoch": val_ccsn["criterion_f1"]["best_epoch"],
            "val_metrics": val_ccsn["criterion_f1"],
            "test_holdout": rep_ccsn_in_f1["metrics_summary"],
            "checkpoint_reused": bool(args.reuse_checkpoints and ccsn_ckpt_f1.exists()),
            "checkpoint_metadata": ccsn_ckpt_f1_meta,
        }
        results["criterion_f1"]["cross_source_ccsn_to_gcd"] = {
            "test_holdout": rep_ccsn_on_gcd_f1["metrics_summary"],
        }

        results["ccsn_in_domain"] = results["criterion_loss"]["ccsn_in_domain"]
        results["cross_source_ccsn_to_gcd"] = results["criterion_loss"]["cross_source_ccsn_to_gcd"]

        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    # =========================================================================
    # EXPERIMENT 2: GCD IN-DOMAIN MODEL & CROSS-SOURCE TRANSFER TO CCSN
    # =========================================================================
    if args.experiment in ("all", "gcd"):
        seed_everything(seed)
        print("\n" + "=" * 90, flush=True)
        print(f"EXPERIMENT 2: GCD IN-DOMAIN TRAINING & CROSS-SOURCE TRANSFER TO CCSN ({model_arch.upper()})", flush=True)
        print("=" * 90, flush=True)
        gcd_ckpt_meta = load_checkpoint_metadata(gcd_ckpt)
        gcd_ckpt_f1_meta = load_checkpoint_metadata(gcd_ckpt_f1)
        if args.reuse_checkpoints and gcd_ckpt.exists() and gcd_ckpt_f1.exists():
            print(f"[*] Reusing checkpoints: {gcd_ckpt} and {gcd_ckpt_f1}", flush=True)
            validate_checkpoint_provenance(
                gcd_ckpt_meta, {
                    "model_architecture": model_arch,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "seed": seed,
                },
                gcd_ckpt.name,
                allow_unverified=args.allow_unverified_checkpoints,
            )
            model_gcd_loss = build_resnet_model(model_name=model_arch, num_classes=5, dropout_head=drop_hd, dropout_bb=drop_bb).to(DEVICE)
            model_gcd_loss.load_state_dict(torch.load(gcd_ckpt, map_location=DEVICE))
            model_gcd_f1 = build_resnet_model(model_name=model_arch, num_classes=5, dropout_head=drop_hd, dropout_bb=drop_bb).to(DEVICE)
            model_gcd_f1.load_state_dict(torch.load(gcd_ckpt_f1, map_location=DEVICE))
            val_gcd = results.get("gcd_in_domain", {}).get("val_metrics", {"reused": True})
        else:
            model_gcd_loss, model_gcd_f1, val_gcd = train_pool_model(
                "GCD_Model", gcd_tr_ds, gcd_va_ds,
                model_arch=model_arch, epochs=epochs, batch_size=batch_size,
                lr_backbone=lr_bb, lr_head=lr_hd, weight_decay=wd, label_smoothing=ls,
                dropout_head=drop_hd, dropout_bb=drop_bb, optimizer_type=opt_type,
                scheduler_type=sched_type, eta_min=eta_min, momentum=momentum,
                seed=seed,
            )
            torch.save(model_gcd_loss.state_dict(), gcd_ckpt)
            torch.save(model_gcd_f1.state_dict(), gcd_ckpt_f1)
            common_meta = {
                "model_architecture": model_arch,
                "experiment_condition": "gcd_in_domain",
                "epochs": epochs,
                "batch_size": batch_size,
                "seed": seed,
                "optimizer": opt_type,
                "lr_backbone": lr_bb,
                "lr_head": lr_hd,
                "weight_decay": wd,
                "label_smoothing": ls,
                "dropout_backbone": drop_bb,
                "dropout_head": drop_hd,
                "scheduler": sched_type,
                "eta_min": eta_min,
                "manifest_path": str(manifest_file),
                "augmentation": {
                    "vertical_flip": False,
                    "horizontal_flip_p": aug_cfg.get("horizontal_flip_p", 0.5),
                    "rotation_degrees": aug_cfg.get("rotation_degrees", 15),
                    "color_jitter": aug_cfg.get("color_jitter", [0.1, 0.1, 0.1]),
                    "resized_crop_scale": aug_cfg.get("resized_crop_scale", [0.8, 1.0]),
                },
            }
            save_checkpoint_metadata(gcd_ckpt, {
                **common_meta,
                "checkpoint_file": gcd_ckpt.name,
                "selection_criterion": "min_val_loss",
                "selected_epoch": val_gcd["criterion_loss"]["best_epoch"],
                "val_metrics": val_gcd["criterion_loss"],
            })
            save_checkpoint_metadata(gcd_ckpt_f1, {
                **common_meta,
                "checkpoint_file": gcd_ckpt_f1.name,
                "selection_criterion": "max_val_macro_f1",
                "selected_epoch": val_gcd["criterion_f1"]["best_epoch"],
                "val_metrics": val_gcd["criterion_f1"],
            })
            gcd_ckpt_meta = load_checkpoint_metadata(gcd_ckpt)
            gcd_ckpt_f1_meta = load_checkpoint_metadata(gcd_ckpt_f1)

        # 2A. GCD In-Domain Test Evaluation (Loss)
        p_gcd_in_loss, t_gcd_in = evaluate_on_dataset(model_gcd_loss, gcd_te_ds, batch_size=batch_size)
        rep_gcd_in_loss = generate_evaluation_report(
            t_gcd_in, p_gcd_in_loss, HARMONIZED_CLASSES,
            dataset_name=f"Harmonized GCD In-Domain (Five-Class, {model_arch.upper()})",
            model_name=model_arch,
            output_dir=args.figures_dir,
            title_suffix=f"GCD In-Domain Holdout ({model_arch.upper()}, Loss)",
        )

        # 2B. Cross-Source Transfer: GCD Model on CCSN Test Holdout (Loss)
        p_gcd_on_ccsn_loss, t_ccsn_te = evaluate_on_dataset(model_gcd_loss, ccsn_te_ds, batch_size=batch_size)
        rep_gcd_on_ccsn_loss = generate_evaluation_report(
            t_ccsn_te, p_gcd_on_ccsn_loss, HARMONIZED_CLASSES,
            dataset_name=f"Cross-Source GCD to CCSN (Five-Class, {model_arch.upper()})",
            model_name=model_arch,
            output_dir=args.figures_dir,
            title_suffix=f"GCD Model -> CCSN Holdout ({model_arch.upper()}, Loss)",
        )

        # 2A-F1. GCD In-Domain Test Evaluation (F1)
        p_gcd_in_f1, _ = evaluate_on_dataset(model_gcd_f1, gcd_te_ds, batch_size=batch_size)
        rep_gcd_in_f1 = generate_evaluation_report(
            t_gcd_in, p_gcd_in_f1, HARMONIZED_CLASSES,
            dataset_name=f"Harmonized GCD In-Domain (Five-Class, {model_arch.upper()})",
            model_name=f"{model_arch}_best_f1",
            output_dir=args.figures_dir,
            title_suffix=f"GCD In-Domain Holdout ({model_arch.upper()}, F1)",
        )

        # 2B-F1. Cross-Source Transfer: GCD Model on CCSN Test Holdout (F1)
        p_gcd_on_ccsn_f1, _ = evaluate_on_dataset(model_gcd_f1, ccsn_te_ds, batch_size=batch_size)
        rep_gcd_on_ccsn_f1 = generate_evaluation_report(
            t_ccsn_te, p_gcd_on_ccsn_f1, HARMONIZED_CLASSES,
            dataset_name=f"Cross-Source GCD to CCSN (Five-Class, {model_arch.upper()})",
            model_name=f"{model_arch}_best_f1",
            output_dir=args.figures_dir,
            title_suffix=f"GCD Model -> CCSN Holdout ({model_arch.upper()}, F1)",
        )

        results.setdefault("criterion_loss", {})
        results.setdefault("criterion_f1", {})

        results["criterion_loss"]["gcd_in_domain"] = {
            "selected_epoch": val_gcd["criterion_loss"]["best_epoch"],
            "val_metrics": val_gcd["criterion_loss"],
            "test_holdout": rep_gcd_in_loss["metrics_summary"],
            "checkpoint_reused": bool(args.reuse_checkpoints and gcd_ckpt.exists()),
            "checkpoint_metadata": gcd_ckpt_meta,
        }
        results["criterion_loss"]["cross_source_gcd_to_ccsn"] = {
            "test_holdout": rep_gcd_on_ccsn_loss["metrics_summary"],
        }

        results["criterion_f1"]["gcd_in_domain"] = {
            "selected_epoch": val_gcd["criterion_f1"]["best_epoch"],
            "val_metrics": val_gcd["criterion_f1"],
            "test_holdout": rep_gcd_in_f1["metrics_summary"],
            "checkpoint_reused": bool(args.reuse_checkpoints and gcd_ckpt_f1.exists()),
            "checkpoint_metadata": gcd_ckpt_f1_meta,
        }
        results["criterion_f1"]["cross_source_gcd_to_ccsn"] = {
            "test_holdout": rep_gcd_on_ccsn_f1["metrics_summary"],
        }

        results["gcd_in_domain"] = results["criterion_loss"]["gcd_in_domain"]
        results["cross_source_gcd_to_ccsn"] = results["criterion_loss"]["cross_source_gcd_to_ccsn"]

        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    # =========================================================================
    # EXPERIMENT 3: JOINT CCSN+GCD MODEL (SOURCE-BALANCED CCSN + GCD TRAINING)
    # =========================================================================
    if args.experiment in ("all", "joint"):
        seed_everything(seed)
        print("\n" + "=" * 90, flush=True)
        print(f"EXPERIMENT 3: JOINT CCSN+GCD MODEL (SOURCE-BALANCED CCSN + GCD TRAINING, {model_arch.upper()})", flush=True)
        print("=" * 90, flush=True)
        n_ccsn_tr = len(ccsn_tr_ds)
        n_gcd_tr = len(gcd_tr_ds)
        w_ccsn = 0.5 / n_ccsn_tr
        w_gcd = 0.5 / n_gcd_tr
        joint_weights = torch.cat([
            torch.full((n_ccsn_tr,), w_ccsn, dtype=torch.double),
            torch.full((n_gcd_tr,), w_gcd, dtype=torch.double),
        ])
        print(f"[*] Enabled source-balanced batch sampling for Joint CCSN+GCD Model: {n_ccsn_tr} CCSN ({w_ccsn:.6f}) + {n_gcd_tr} GCD ({w_gcd:.6f})", flush=True)

        joint_ckpt_meta = load_checkpoint_metadata(joint_ckpt)
        joint_ckpt_f1_meta = load_checkpoint_metadata(joint_ckpt_f1)
        if args.reuse_checkpoints and joint_ckpt.exists() and joint_ckpt_f1.exists():
            print(f"[*] Reusing checkpoints: {joint_ckpt} and {joint_ckpt_f1}", flush=True)
            validate_checkpoint_provenance(
                joint_ckpt_meta, {
                    "model_architecture": model_arch,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "seed": seed,
                },
                joint_ckpt.name,
                allow_unverified=args.allow_unverified_checkpoints,
            )
            model_joint_loss = build_resnet_model(model_name=model_arch, num_classes=5, dropout_head=drop_hd, dropout_bb=drop_bb).to(DEVICE)
            model_joint_loss.load_state_dict(torch.load(joint_ckpt, map_location=DEVICE))
            model_joint_f1 = build_resnet_model(model_name=model_arch, num_classes=5, dropout_head=drop_hd, dropout_bb=drop_bb).to(DEVICE)
            model_joint_f1.load_state_dict(torch.load(joint_ckpt_f1, map_location=DEVICE))
            val_joint = results.get("joint_model", {}).get("val_metrics", {"reused": True})
        else:
            model_joint_loss, model_joint_f1, val_joint = train_pool_model(
                "Joint_CCSN_GCD_Model",
                joint_tr_ds,
                joint_va_ds,
                model_arch=model_arch,
                epochs=epochs,
                batch_size=batch_size,
                lr_backbone=lr_bb,
                lr_head=lr_hd,
                weight_decay=wd,
                label_smoothing=ls,
                dropout_head=drop_hd,
                dropout_bb=drop_bb,
                optimizer_type=opt_type,
                sample_weights=joint_weights,
                scheduler_type=sched_type,
                eta_min=eta_min,
                momentum=momentum,
                seed=seed,
            )
            torch.save(model_joint_loss.state_dict(), joint_ckpt)
            torch.save(model_joint_f1.state_dict(), joint_ckpt_f1)
            common_meta = {
                "model_architecture": model_arch,
                "experiment_condition": "joint_model",
                "epochs": epochs,
                "batch_size": batch_size,
                "seed": seed,
                "optimizer": opt_type,
                "lr_backbone": lr_bb,
                "lr_head": lr_hd,
                "weight_decay": wd,
                "label_smoothing": ls,
                "dropout_backbone": drop_bb,
                "dropout_head": drop_hd,
                "scheduler": sched_type,
                "eta_min": eta_min,
                "manifest_path": str(manifest_file),
                "augmentation": {
                    "vertical_flip": False,
                    "horizontal_flip_p": aug_cfg.get("horizontal_flip_p", 0.5),
                    "rotation_degrees": aug_cfg.get("rotation_degrees", 15),
                    "color_jitter": aug_cfg.get("color_jitter", [0.1, 0.1, 0.1]),
                    "resized_crop_scale": aug_cfg.get("resized_crop_scale", [0.8, 1.0]),
                },
            }
            save_checkpoint_metadata(joint_ckpt, {
                **common_meta,
                "checkpoint_file": joint_ckpt.name,
                "selection_criterion": "min_val_loss",
                "selected_epoch": val_joint["criterion_loss"]["best_epoch"],
                "val_metrics": val_joint["criterion_loss"],
            })
            save_checkpoint_metadata(joint_ckpt_f1, {
                **common_meta,
                "checkpoint_file": joint_ckpt_f1.name,
                "selection_criterion": "max_val_macro_f1",
                "selected_epoch": val_joint["criterion_f1"]["best_epoch"],
                "val_metrics": val_joint["criterion_f1"],
            })
            joint_ckpt_meta = load_checkpoint_metadata(joint_ckpt)
            joint_ckpt_f1_meta = load_checkpoint_metadata(joint_ckpt_f1)

        # 3A. Joint Model on CCSN Test Holdout (Loss)
        p_m_ccsn_loss, t_m_ccsn = evaluate_on_dataset(model_joint_loss, ccsn_te_ds, batch_size=batch_size)
        rep_m_ccsn_loss = generate_evaluation_report(
            t_m_ccsn, p_m_ccsn_loss, HARMONIZED_CLASSES,
            dataset_name=f"Joint CCSN+GCD Model on CCSN (Five-Class, {model_arch.upper()})",
            model_name=model_arch,
            output_dir=args.figures_dir,
            title_suffix=f"Joint Model -> CCSN Holdout ({model_arch.upper()}, Loss)",
        )

        # 3B. Joint Model on GCD Test Holdout (Loss)
        p_m_gcd_loss, t_m_gcd = evaluate_on_dataset(model_joint_loss, gcd_te_ds, batch_size=batch_size)
        rep_m_gcd_loss = generate_evaluation_report(
            t_m_gcd, p_m_gcd_loss, HARMONIZED_CLASSES,
            dataset_name=f"Joint CCSN+GCD Model on GCD (Five-Class, {model_arch.upper()})",
            model_name=model_arch,
            output_dir=args.figures_dir,
            title_suffix=f"Joint Model -> GCD Holdout ({model_arch.upper()}, Loss)",
        )

        # 3C. Joint Model on Combined Test Holdout (Loss)
        p_m_joint_loss, t_m_joint = evaluate_on_dataset(model_joint_loss, joint_te_ds, batch_size=batch_size)
        rep_m_joint_loss = generate_evaluation_report(
            t_m_joint, p_m_joint_loss, HARMONIZED_CLASSES,
            dataset_name=f"Joint CCSN+GCD Model Combined Holdout (Five-Class, {model_arch.upper()})",
            model_name=model_arch,
            output_dir=args.figures_dir,
            title_suffix=f"Joint Model -> Combined Holdout ({model_arch.upper()}, Loss)",
        )

        raw_ccsn_acc_l = rep_m_ccsn_loss.get("raw_overall_accuracy", rep_m_ccsn_loss["overall_accuracy"])
        raw_gcd_acc_l = rep_m_gcd_loss.get("raw_overall_accuracy", rep_m_gcd_loss["overall_accuracy"])
        raw_ccsn_bal_l = rep_m_ccsn_loss.get("raw_balanced_accuracy", rep_m_ccsn_loss["balanced_accuracy"])
        raw_gcd_bal_l = rep_m_gcd_loss.get("raw_balanced_accuracy", rep_m_gcd_loss["balanced_accuracy"])
        raw_ccsn_f1_l = rep_m_ccsn_loss.get("raw_macro_f1", rep_m_ccsn_loss["macro_f1"])
        raw_gcd_f1_l = rep_m_gcd_loss.get("raw_macro_f1", rep_m_gcd_loss["macro_f1"])

        sb_acc_l = (raw_ccsn_acc_l + raw_gcd_acc_l) / 2.0
        sb_bal_acc_l = (raw_ccsn_bal_l + raw_gcd_bal_l) / 2.0
        sb_macro_f1_l = (raw_ccsn_f1_l + raw_gcd_f1_l) / 2.0

        # 3A-F1. Joint Model on CCSN Test Holdout (F1)
        p_m_ccsn_f1, _ = evaluate_on_dataset(model_joint_f1, ccsn_te_ds, batch_size=batch_size)
        rep_m_ccsn_f1 = generate_evaluation_report(
            t_m_ccsn, p_m_ccsn_f1, HARMONIZED_CLASSES,
            dataset_name=f"Joint CCSN+GCD Model on CCSN (Five-Class, {model_arch.upper()})",
            model_name=f"{model_arch}_best_f1",
            output_dir=args.figures_dir,
            title_suffix=f"Joint Model -> CCSN Holdout ({model_arch.upper()}, F1)",
        )

        # 3B-F1. Joint Model on GCD Test Holdout (F1)
        p_m_gcd_f1, _ = evaluate_on_dataset(model_joint_f1, gcd_te_ds, batch_size=batch_size)
        rep_m_gcd_f1 = generate_evaluation_report(
            t_m_gcd, p_m_gcd_f1, HARMONIZED_CLASSES,
            dataset_name=f"Joint CCSN+GCD Model on GCD (Five-Class, {model_arch.upper()})",
            model_name=f"{model_arch}_best_f1",
            output_dir=args.figures_dir,
            title_suffix=f"Joint Model -> GCD Holdout ({model_arch.upper()}, F1)",
        )

        # 3C-F1. Joint Model on Combined Test Holdout (F1)
        p_m_joint_f1, _ = evaluate_on_dataset(model_joint_f1, joint_te_ds, batch_size=batch_size)
        rep_m_joint_f1 = generate_evaluation_report(
            t_m_joint, p_m_joint_f1, HARMONIZED_CLASSES,
            dataset_name=f"Joint CCSN+GCD Model Combined Holdout (Five-Class, {model_arch.upper()})",
            model_name=f"{model_arch}_best_f1",
            output_dir=args.figures_dir,
            title_suffix=f"Joint Model -> Combined Holdout ({model_arch.upper()}, F1)",
        )

        raw_ccsn_acc_f = rep_m_ccsn_f1.get("raw_overall_accuracy", rep_m_ccsn_f1["overall_accuracy"])
        raw_gcd_acc_f = rep_m_gcd_f1.get("raw_overall_accuracy", rep_m_gcd_f1["overall_accuracy"])
        raw_ccsn_bal_f = rep_m_ccsn_f1.get("raw_balanced_accuracy", rep_m_ccsn_f1["balanced_accuracy"])
        raw_gcd_bal_f = rep_m_gcd_f1.get("raw_balanced_accuracy", rep_m_gcd_f1["balanced_accuracy"])
        raw_ccsn_f1_f = rep_m_ccsn_f1.get("raw_macro_f1", rep_m_ccsn_f1["macro_f1"])
        raw_gcd_f1_f = rep_m_gcd_f1.get("raw_macro_f1", rep_m_gcd_f1["macro_f1"])

        sb_acc_f = (raw_ccsn_acc_f + raw_gcd_acc_f) / 2.0
        sb_bal_acc_f = (raw_ccsn_bal_f + raw_gcd_bal_f) / 2.0
        sb_macro_f1_f = (raw_ccsn_f1_f + raw_gcd_f1_f) / 2.0

        results.setdefault("criterion_loss", {})
        results.setdefault("criterion_f1", {})

        results["criterion_loss"]["joint_model"] = {
            "selected_epoch": val_joint["criterion_loss"]["best_epoch"],
            "val_metrics": val_joint["criterion_loss"],
            "test_on_ccsn": rep_m_ccsn_loss["metrics_summary"],
            "test_on_gcd": rep_m_gcd_loss["metrics_summary"],
            "test_on_joint": rep_m_joint_loss["metrics_summary"],
            "source_balanced_average": {
                "overall_accuracy": round(sb_acc_l, 2),
                "balanced_accuracy": round(sb_bal_acc_l, 2),
                "macro_f1": round(sb_macro_f1_l, 2),
            },
            "checkpoint_reused": bool(args.reuse_checkpoints and joint_ckpt.exists()),
            "checkpoint_metadata": joint_ckpt_meta,
        }

        results["criterion_f1"]["joint_model"] = {
            "selected_epoch": val_joint["criterion_f1"]["best_epoch"],
            "val_metrics": val_joint["criterion_f1"],
            "test_on_ccsn": rep_m_ccsn_f1["metrics_summary"],
            "test_on_gcd": rep_m_gcd_f1["metrics_summary"],
            "test_on_joint": rep_m_joint_f1["metrics_summary"],
            "source_balanced_average": {
                "overall_accuracy": round(sb_acc_f, 2),
                "balanced_accuracy": round(sb_bal_acc_f, 2),
                "macro_f1": round(sb_macro_f1_f, 2),
            },
            "checkpoint_reused": bool(args.reuse_checkpoints and joint_ckpt_f1.exists()),
            "checkpoint_metadata": joint_ckpt_f1_meta,
        }

        results["joint_model"] = results["criterion_loss"]["joint_model"]

        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        try:
            from experiment_registry import register_experiment
            joint_meta = joint_ckpt_meta or {}
            reused_flag = bool(args.reuse_checkpoints and joint_ckpt.exists())
            register_experiment(
                experiment_id=f"harmonized_5bin_{model_arch}_joint",
                dataset_key="harmonized_5bin",
                taxonomy_name="Five-Class Cross-Source Compatibility Taxonomy",
                manifest_path=manifest_file,
                protocol="grouped_stratified_holdout_v1.0",
                model_architecture=model_arch,
                hyperparameters={
                    "epochs": joint_meta.get("epochs", epochs) if reused_flag else epochs,
                    "batch_size": joint_meta.get("batch_size", batch_size) if reused_flag else batch_size,
                    "source_balanced_sampling": True,
                    "seed": joint_meta.get("seed", seed) if reused_flag else seed,
                    "lr_backbone": joint_meta.get("lr_backbone", lr_bb) if reused_flag else lr_bb,
                    "lr_head": joint_meta.get("lr_head", lr_hd) if reused_flag else lr_hd,
                    "checkpoint_reused": reused_flag,
                },
                metrics=results["joint_model"]["test_on_joint"],
                checkpoint_path=joint_ckpt,
                figure_path=results["joint_model"]["test_on_joint"].get("figure_path"),
                notes=f"CCSN + GCD joint training ({model_arch}) on the five-class compatibility taxonomy with independent validation model selection, source-balanced batch sampling, and source-stratified evaluation.",
            )
        except Exception as e:
            print(f"[!] Registry notice: {e}", flush=True)

    for crit_key, crit_title in [
        ("criterion_loss", "Criterion: Min Validation Loss (Canonical)"),
        ("criterion_f1", "Criterion: Max Validation Macro-F1 (Robustness Diagnostic)"),
    ]:
        crit_data = results.get(crit_key, {})
        rows = []
        if "ccsn_in_domain" in crit_data:
            r = crit_data["ccsn_in_domain"]["test_holdout"]
            ep = crit_data["ccsn_in_domain"].get("selected_epoch", "?")
            rows.append({
                "Condition": f"1. CCSN In-Domain (ep {ep})",
                "Evaluated On": "CCSN Holdout",
                "Top-1 Acc (%)": f"{r['overall_accuracy']:.2f}%",
                "Balanced Acc (%)": f"{r['balanced_accuracy']:.2f}%",
                "Macro-F1 (%)": f"{r['macro_f1']:.2f}%",
                "95% CI (Accuracy)": f"[{r['bootstrap_95ci']['accuracy']['ci_lower']}%, {r['bootstrap_95ci']['accuracy']['ci_upper']}%]",
            })
        if "cross_source_ccsn_to_gcd" in crit_data:
            r = crit_data["cross_source_ccsn_to_gcd"]["test_holdout"]
            rows.append({
                "Condition": f"2. Cross-Source CCSN -> GCD",
                "Evaluated On": "GCD Holdout",
                "Top-1 Acc (%)": f"{r['overall_accuracy']:.2f}%",
                "Balanced Acc (%)": f"{r['balanced_accuracy']:.2f}%",
                "Macro-F1 (%)": f"{r['macro_f1']:.2f}%",
                "95% CI (Accuracy)": f"[{r['bootstrap_95ci']['accuracy']['ci_lower']}%, {r['bootstrap_95ci']['accuracy']['ci_upper']}%]",
            })
        if "gcd_in_domain" in crit_data:
            r = crit_data["gcd_in_domain"]["test_holdout"]
            ep = crit_data["gcd_in_domain"].get("selected_epoch", "?")
            rows.append({
                "Condition": f"3. GCD In-Domain (ep {ep})",
                "Evaluated On": "GCD Holdout",
                "Top-1 Acc (%)": f"{r['overall_accuracy']:.2f}%",
                "Balanced Acc (%)": f"{r['balanced_accuracy']:.2f}%",
                "Macro-F1 (%)": f"{r['macro_f1']:.2f}%",
                "95% CI (Accuracy)": f"[{r['bootstrap_95ci']['accuracy']['ci_lower']}%, {r['bootstrap_95ci']['accuracy']['ci_upper']}%]",
            })
        if "cross_source_gcd_to_ccsn" in crit_data:
            r = crit_data["cross_source_gcd_to_ccsn"]["test_holdout"]
            rows.append({
                "Condition": f"4. Cross-Source GCD -> CCSN",
                "Evaluated On": "CCSN Holdout",
                "Top-1 Acc (%)": f"{r['overall_accuracy']:.2f}%",
                "Balanced Acc (%)": f"{r['balanced_accuracy']:.2f}%",
                "Macro-F1 (%)": f"{r['macro_f1']:.2f}%",
                "95% CI (Accuracy)": f"[{r['bootstrap_95ci']['accuracy']['ci_lower']}%, {r['bootstrap_95ci']['accuracy']['ci_upper']}%]",
            })
        if "joint_model" in crit_data:
            jm = crit_data["joint_model"]
            ep = jm.get("selected_epoch", "?")
            r_c = jm["test_on_ccsn"]
            r_g = jm["test_on_gcd"]
            r_j = jm["test_on_joint"]
            r_sb = jm["source_balanced_average"]
            rows.append({
                "Condition": f"5. Joint CCSN+GCD Model (ep {ep})",
                "Evaluated On": "CCSN Holdout",
                "Top-1 Acc (%)": f"{r_c['overall_accuracy']:.2f}%",
                "Balanced Acc (%)": f"{r_c['balanced_accuracy']:.2f}%",
                "Macro-F1 (%)": f"{r_c['macro_f1']:.2f}%",
                "95% CI (Accuracy)": f"[{r_c['bootstrap_95ci']['accuracy']['ci_lower']}%, {r_c['bootstrap_95ci']['accuracy']['ci_upper']}%]",
            })
            rows.append({
                "Condition": f"6. Joint CCSN+GCD Model (ep {ep})",
                "Evaluated On": "GCD Holdout",
                "Top-1 Acc (%)": f"{r_g['overall_accuracy']:.2f}%",
                "Balanced Acc (%)": f"{r_g['balanced_accuracy']:.2f}%",
                "Macro-F1 (%)": f"{r_g['macro_f1']:.2f}%",
                "95% CI (Accuracy)": f"[{r_g['bootstrap_95ci']['accuracy']['ci_lower']}%, {r_g['bootstrap_95ci']['accuracy']['ci_upper']}%]",
            })
            rows.append({
                "Condition": f"7. Joint CCSN+GCD Model (ep {ep})",
                "Evaluated On": "Combined Holdout (GCD-heavy)",
                "Top-1 Acc (%)": f"{r_j['overall_accuracy']:.2f}%",
                "Balanced Acc (%)": f"{r_j['balanced_accuracy']:.2f}%",
                "Macro-F1 (%)": f"{r_j['macro_f1']:.2f}%",
                "95% CI (Accuracy)": f"[{r_j['bootstrap_95ci']['accuracy']['ci_lower']}%, {r_j['bootstrap_95ci']['accuracy']['ci_upper']}%]",
            })
            rows.append({
                "Condition": f"8. Joint CCSN+GCD Model (ep {ep})",
                "Evaluated On": "Source-Balanced Average",
                "Top-1 Acc (%)": f"{r_sb['overall_accuracy']:.2f}%",
                "Balanced Acc (%)": f"{r_sb['balanced_accuracy']:.2f}%",
                "Macro-F1 (%)": f"{r_sb['macro_f1']:.2f}%",
                "95% CI (Accuracy)": "Macro-mean over domains",
            })
        if rows:
            print("\n" + "=" * 95, flush=True)
            print(f"HARMONIZED CROSS-SOURCE RESULTS TABLE ({model_arch.upper()} | {crit_title})", flush=True)
            print("=" * 95, flush=True)
            df_comp = pd.DataFrame(rows)
            print(df_comp.to_string(index=False), flush=True)
            print("=" * 95, flush=True)

    print(f"[+] Master summary saved to: {summary_file}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        import traceback
        with open("crash_debug.txt", "w", encoding="utf-8") as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)

