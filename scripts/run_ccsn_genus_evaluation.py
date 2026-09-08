# -*- coding: utf-8 -*-
"""
CCSN 11-Class Genus Training & Evaluation Orchestrator.

Orchestrates:
1. `scripts/train_ccsn_genus.py`: Trains ResNet-18 across 3 seeds {42, 43, 44}
   on train and val partitions with STRICT TEST SPLIT ISOLATION.
2. `scripts/eval_ccsn_genus.py`: Evaluates the frozen min-val-loss checkpoints
   on the held-out test split (508 images), generating raw predictions,
   the 11x11 confusion matrix plot, metrics, and documentation.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from train_ccsn_genus import train_ccsn_genus, CONFIG_PATH, SEEDS, ARTIFACTS_DIR
from eval_ccsn_genus import evaluate_ccsn_genus


def main():
    parser = argparse.ArgumentParser(description="CCSN 11-Class Unified Orchestrator")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH, help="Path to TOML configuration")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS, help="Seeds to train and evaluate")
    parser.add_argument("--output-dir", type=Path, default=ARTIFACTS_DIR, help="Directory for all artifacts")
    parser.add_argument("--skip-train", action="store_true", help="Skip training if checkpoints already exist")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Training with strict test split isolation
    checkpoints_exist = all((args.output_dir / f"resnet18_ccsn11_seed{s}.pth").exists() for s in args.seeds)
    if args.skip_train and checkpoints_exist:
        print("[*] Checkpoints already exist and --skip-train was specified. Skipping training phase.")
    else:
        print("\n" + "=" * 90)
        print("[PHASE 1] TRAINING RESNET-18 (STRICT TEST ISOLATION)")
        print("=" * 90)
        train_ccsn_genus(
            config_path=args.config,
            seeds=args.seeds,
            output_dir=args.output_dir,
        )

    # Step 2: Test evaluation
    print("\n" + "=" * 90)
    print("[PHASE 2] HELD-OUT TEST EVALUATION")
    print("=" * 90)
    evaluate_ccsn_genus(
        config_path=args.config,
        seeds=args.seeds,
        checkpoints_dir=args.output_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
