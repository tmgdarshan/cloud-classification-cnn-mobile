# -*- coding: utf-8 -*-
"""
Build Canonical Manifests with Group-Aware Splitting.

Generates deterministic, group-aware JSON manifests for:
1. CCSN 11-Class Fine-Grained Taxonomy (2,537 samples, excluding 6 cross-class duplicates)
2. GCD 6-Class Sky-Condition Taxonomy (18,045 samples, excluding 7_mixed)
3. GCD 5-Class Cloud-Only Operational Taxonomy (14,306 samples, excluding 7_mixed & 4_clearsky)
4. Harmonized five-class cross-source taxonomy (16,643 samples: 2,337 CCSN + 14,306 GCD)

Forwards:
- Duplicate groups (within-class and cross-split) share atomic group_ids.
- StratifiedGroupKFold ensures no duplicate group ever crosses train/val/test partitions.
- Grouped stratified holdout (single fixed partition): 64% Train / 16% Val / 20% Permanent Test Holdout.
- 5-Fold Stratified Group CV fold assignments (0..4) within the 80% Development Pool.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

REPO_ROOT = Path(__file__).resolve().parent.parent
METADATA_DIR = REPO_ROOT / "metadata"
SPLITS_DIR = METADATA_DIR / "splits"
CCSN_DIR = REPO_ROOT / "CCSN" / "CCSN_v2"
GCD_DIR = REPO_ROOT / "GCD"

CCSN_EXCLUDED = {
    "Ac/Ac-N186.jpg", "As/As-N139.jpg",
    "Ac/Ac-N202.jpg", "As/As-N175.jpg",
    "Cc/Cc-N179.jpg", "Cs/Cs-N244.jpg",
}

CCSN_CLASSES = ["Ac", "As", "Cb", "Cc", "Ci", "Cs", "Ct", "Cu", "Ns", "Sc", "St"]

GCD_6_CLASSES = ["1_cumulus", "2_altocumulus", "3_cirrus", "4_clearsky", "5_stratocumulus", "6_cumulonimbus"]
GCD_5_CLASSES = ["1_cumulus", "2_altocumulus", "3_cirrus", "5_stratocumulus", "6_cumulonimbus"]

HARMONIZED_CLASSES = ["cumulus", "altocumulus", "cirrus", "stratocumulus", "cumulonimbus"]

CCSN_TO_HARMONIZED = {
    "Cu": "cumulus",
    "Ac": "altocumulus", "Cc": "altocumulus",
    "Ci": "cirrus", "Cs": "cirrus",
    "Sc": "stratocumulus", "St": "stratocumulus", "As": "stratocumulus",
    "Cb": "cumulonimbus", "Ns": "cumulonimbus",
}

GCD_TO_HARMONIZED = {
    "1_cumulus": "cumulus",
    "2_altocumulus": "altocumulus",
    "3_cirrus": "cirrus",
    "5_stratocumulus": "stratocumulus",
    "6_cumulonimbus": "cumulonimbus",
}


def compute_sha256(data: Any) -> str:
    serialized = json.dumps(data, sort_keys=True).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def partition_samples_stratified_group(
    samples: list[dict[str, Any]],
    n_outer_splits: int = 5,
    n_inner_splits: int = 5,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Partitions samples into 20% test holdout and 80% dev pool (split into 5 CV folds).

    Fold 0 of dev pool serves as the 16% validation split for 3-way training,
    while folds 1..4 serve as the 64% training split.
    """
    labels = np.array([s["label"] for s in samples])
    groups = np.array([s["group_id"] for s in samples])

    # 1. Outer split: 20% permanent test holdout, 80% development pool
    sgkf_outer = StratifiedGroupKFold(n_splits=n_outer_splits, shuffle=True, random_state=seed)
    dev_idx, test_idx = next(sgkf_outer.split(samples, labels, groups))

    dev_groups = groups[dev_idx]
    dev_labels = labels[dev_idx]

    # 2. Inner split: 5-fold CV within the development pool
    sgkf_inner = StratifiedGroupKFold(n_splits=n_inner_splits, shuffle=True, random_state=seed)
    fold_assignments = np.zeros(len(dev_idx), dtype=int)
    for fold_num, (_, fold_val_idx) in enumerate(sgkf_inner.split(dev_idx, dev_labels, dev_groups)):
        fold_assignments[fold_val_idx] = fold_num

    # Assign partitions
    annotated = [dict(s) for s in samples]
    test_set = set(test_idx)
    dev_map = {orig_idx: dev_pos for dev_pos, orig_idx in enumerate(dev_idx)}

    for i, item in enumerate(annotated):
        if i in test_set:
            item["split"] = "test"
            item["fold"] = -1
        else:
            dev_pos = dev_map[i]
            fold = int(fold_assignments[dev_pos])
            item["fold"] = fold
            item["split"] = "val" if fold == 0 else "train"

    # Integrity Assertions
    s_train = {i for i, item in enumerate(annotated) if item["split"] == "train"}
    s_val = {i for i, item in enumerate(annotated) if item["split"] == "val"}
    s_test = {i for i, item in enumerate(annotated) if item["split"] == "test"}
    assert len(s_train & s_val) == 0, "Train and Val overlap!"
    assert len(s_train & s_test) == 0, "Train and Test overlap!"
    assert len(s_val & s_test) == 0, "Val and Test overlap!"
    assert len(s_train | s_val | s_test) == len(samples), "Partition union mismatch!"

    # Verify no group straddles partitions
    unique_groups = set(groups)
    for g in unique_groups:
        indices = np.where(groups == g)[0]
        partitions = {annotated[idx]["split"] for idx in indices}
        assert len(partitions) == 1, f"Group {g} leaked across partitions: {partitions}!"

    return annotated


def build_ccsn_manifest():
    print("\n" + "=" * 80)
    print("[*] Generating CCSN 11-Class Fine-Grained Manifest...")
    with open(METADATA_DIR / "ccsn_inventory.json", "r", encoding="utf-8") as f:
        inv = json.load(f)

    # Build duplicate group lookup
    dup_map = {}
    for g_idx, group in enumerate(inv.get("duplicates", {}).get("groups", [])):
        if any(p in CCSN_EXCLUDED for p in group):
            continue
        for p in group:
            dup_map[p] = f"ccsn_dup_group_{g_idx:03d}"

    class_to_idx = {c: i for i, c in enumerate(CCSN_CLASSES)}
    raw_samples = []
    idx_counter = 0

    for c in CCSN_CLASSES:
        c_dir = CCSN_DIR / c
        for p in sorted(c_dir.glob("*.jpg")):
            rel = f"{c}/{p.name}"
            if rel in CCSN_EXCLUDED:
                continue
            gid = dup_map.get(rel, f"ccsn_single_{idx_counter:05d}")
            idx_counter += 1
            raw_samples.append({
                "path": rel,
                "class": c,
                "label": class_to_idx[c],
                "group_id": gid,
            })

    print(f"    - Clean samples: {len(raw_samples)} (Excluded {len(CCSN_EXCLUDED)} contradictory duplicates)")
    annotated = partition_samples_stratified_group(raw_samples, seed=42)

    train_cnt = sum(1 for s in annotated if s["split"] == "train")
    val_cnt = sum(1 for s in annotated if s["split"] == "val")
    test_cnt = sum(1 for s in annotated if s["split"] == "test")
    print(f"    - Partition: Train={train_cnt} ({train_cnt/len(annotated):.1%}) | Val={val_cnt} ({val_cnt/len(annotated):.1%}) | Test={test_cnt} ({test_cnt/len(annotated):.1%})")

    out_file = SPLITS_DIR / "ccsn_11class_canonical.json"
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset_key": "ccsn",
        "taxonomy_name": "CCSN 11-class genera taxonomy",
        "protocol": "grouped_stratified_holdout_v1.0",
        "seed": 42,
        "classes": CCSN_CLASSES,
        "class_to_idx": class_to_idx,
        "num_classes": len(CCSN_CLASSES),
        "excluded_samples": sorted(list(CCSN_EXCLUDED)),
        "summary": {
            "total_samples": len(annotated),
            "train_samples": train_cnt,
            "val_samples": val_cnt,
            "test_samples": test_cnt,
            "num_duplicate_groups": len({s["group_id"] for s in annotated if "dup" in s["group_id"]}),
        },
        "samples": annotated,
    }
    manifest["sha256"] = compute_sha256(annotated)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[+] Saved CCSN manifest to: {out_file}")


def build_gcd_manifests():
    print("\n" + "=" * 80)
    print("[*] Generating GCD Manifests (6-Class Sky-Condition & 5-Class Cloud-Only)...")
    with open(METADATA_DIR / "gcd_inventory.json", "r", encoding="utf-8") as f:
        inv = json.load(f)

    # Build duplicate group lookup
    dup_map = {}
    for g_idx, group in enumerate(inv.get("duplicates", {}).get("groups", [])):
        for p in group:
            dup_map[p] = f"gcd_dup_group_{g_idx:03d}"

    # 1. GCD 6-Class (dropping 7_mixed, keeping 4_clearsky)
    c6_to_idx = {c: i for i, c in enumerate(GCD_6_CLASSES)}
    c6_samples = []
    idx_counter = 0

    for sp in ["train", "test"]:
        for c in GCD_6_CLASSES:
            c_dir = GCD_DIR / sp / c
            for p in sorted(c_dir.glob("*.jpg")):
                rel = f"{sp}/{c}/{p.name}"
                gid = dup_map.get(rel, f"gcd_single_{idx_counter:06d}")
                idx_counter += 1
                c6_samples.append({
                    "path": rel,
                    "class": c,
                    "label": c6_to_idx[c],
                    "group_id": gid,
                })

    print(f"    - GCD 6-Class Clean samples: {len(c6_samples)} (Dropped 7_mixed)")
    annotated_c6 = partition_samples_stratified_group(c6_samples, seed=42)

    tr_c6 = sum(1 for s in annotated_c6 if s["split"] == "train")
    va_c6 = sum(1 for s in annotated_c6 if s["split"] == "val")
    te_c6 = sum(1 for s in annotated_c6 if s["split"] == "test")
    print(f"    - GCD 6-Class Partition: Train={tr_c6} ({tr_c6/len(annotated_c6):.1%}) | Val={va_c6} ({va_c6/len(annotated_c6):.1%}) | Test={te_c6} ({te_c6/len(annotated_c6):.1%})")

    out_file_c6 = SPLITS_DIR / "gcd_6class_canonical.json"
    manifest_c6 = {
        "dataset_key": "gcd_6class",
        "taxonomy_name": "GCD 6-class sky-condition taxonomy",
        "protocol": "grouped_stratified_holdout_v1.0",
        "seed": 42,
        "classes": GCD_6_CLASSES,
        "class_to_idx": c6_to_idx,
        "num_classes": len(GCD_6_CLASSES),
        "dropped_classes": ["7_mixed"],
        "summary": {
            "total_samples": len(annotated_c6),
            "train_samples": tr_c6,
            "val_samples": va_c6,
            "test_samples": te_c6,
            "num_duplicate_groups": len({s["group_id"] for s in annotated_c6 if "dup" in s["group_id"]}),
        },
        "samples": annotated_c6,
    }
    manifest_c6["sha256"] = compute_sha256(annotated_c6)
    with open(out_file_c6, "w", encoding="utf-8") as f:
        json.dump(manifest_c6, f, indent=2)
    print(f"[+] Saved GCD 6-Class manifest to: {out_file_c6}")

    # 2. GCD 5-Class (dropping 7_mixed AND 4_clearsky)
    c5_to_idx = {c: i for i, c in enumerate(GCD_5_CLASSES)}
    c5_samples = []
    idx_counter = 0

    for sp in ["train", "test"]:
        for c in GCD_5_CLASSES:
            c_dir = GCD_DIR / sp / c
            for p in sorted(c_dir.glob("*.jpg")):
                rel = f"{sp}/{c}/{p.name}"
                gid = dup_map.get(rel, f"gcd5_single_{idx_counter:06d}")
                idx_counter += 1
                c5_samples.append({
                    "path": rel,
                    "class": c,
                    "label": c5_to_idx[c],
                    "group_id": gid,
                })

    print(f"    - GCD 5-Class Clean samples: {len(c5_samples)} (Dropped 7_mixed and 4_clearsky)")
    annotated_c5 = partition_samples_stratified_group(c5_samples, seed=42)

    tr_c5 = sum(1 for s in annotated_c5 if s["split"] == "train")
    va_c5 = sum(1 for s in annotated_c5 if s["split"] == "val")
    te_c5 = sum(1 for s in annotated_c5 if s["split"] == "test")
    print(f"    - GCD 5-Class Partition: Train={tr_c5} ({tr_c5/len(annotated_c5):.1%}) | Val={va_c5} ({va_c5/len(annotated_c5):.1%}) | Test={te_c5} ({te_c5/len(annotated_c5):.1%})")

    out_file_c5 = SPLITS_DIR / "gcd_5class_canonical.json"
    manifest_c5 = {
        "dataset_key": "gcd_5class",
        "taxonomy_name": "GCD 5-class cloud-only taxonomy",
        "protocol": "grouped_stratified_holdout_v1.0",
        "seed": 42,
        "classes": GCD_5_CLASSES,
        "class_to_idx": c5_to_idx,
        "num_classes": len(GCD_5_CLASSES),
        "dropped_classes": ["7_mixed", "4_clearsky"],
        "summary": {
            "total_samples": len(annotated_c5),
            "train_samples": tr_c5,
            "val_samples": va_c5,
            "test_samples": te_c5,
            "num_duplicate_groups": len({s["group_id"] for s in annotated_c5 if "dup" in s["group_id"]}),
        },
        "samples": annotated_c5,
    }
    manifest_c5["sha256"] = compute_sha256(annotated_c5)
    with open(out_file_c5, "w", encoding="utf-8") as f:
        json.dump(manifest_c5, f, indent=2)
    print(f"[+] Saved GCD 5-Class manifest to: {out_file_c5}")


def build_harmonized_manifest():
    print("\n" + "=" * 80)
    print("[*] Generating harmonized five-class manifest...")
    h_to_idx = {c: i for i, c in enumerate(HARMONIZED_CLASSES)}

    with open(METADATA_DIR / "ccsn_inventory.json", "r", encoding="utf-8") as f:
        ccsn_inv = json.load(f)
    with open(METADATA_DIR / "gcd_inventory.json", "r", encoding="utf-8") as f:
        gcd_inv = json.load(f)

    # 1. CCSN Harmonized Samples (Dropping Ct and 6 cross-class duplicates)
    ccsn_dup_map = {}
    for g_idx, group in enumerate(ccsn_inv.get("duplicates", {}).get("groups", [])):
        if any(p in CCSN_EXCLUDED for p in group):
            continue
        for p in group:
            ccsn_dup_map[p] = f"ccsn_dup_{g_idx:03d}"

    raw_ccsn_h = []
    idx_ccsn = 0
    for raw_c, h_c in CCSN_TO_HARMONIZED.items():
        c_dir = CCSN_DIR / raw_c
        for p in sorted(c_dir.glob("*.jpg")):
            rel = f"{raw_c}/{p.name}"
            if rel in CCSN_EXCLUDED:
                continue
            gid = ccsn_dup_map.get(rel, f"ccsn_h_single_{idx_ccsn:05d}")
            idx_ccsn += 1
            raw_ccsn_h.append({
                "dataset": "ccsn",
                "path": rel,
                "source_class": raw_c,
                "harmonized_class": h_c,
                "label": h_to_idx[h_c],
                "group_id": gid,
            })

    # Group-partition CCSN harmonized samples
    annotated_ccsn_h = partition_samples_stratified_group(raw_ccsn_h, seed=42)

    # 2. GCD Harmonized Samples (Dropping 4_clearsky and 7_mixed)
    gcd_dup_map = {}
    for g_idx, group in enumerate(gcd_inv.get("duplicates", {}).get("groups", [])):
        for p in group:
            gcd_dup_map[p] = f"gcd_dup_{g_idx:03d}"

    raw_gcd_h = []
    idx_gcd = 0
    for sp in ["train", "test"]:
        for raw_c, h_c in GCD_TO_HARMONIZED.items():
            c_dir = GCD_DIR / sp / raw_c
            for p in sorted(c_dir.glob("*.jpg")):
                rel = f"{sp}/{raw_c}/{p.name}"
                gid = gcd_dup_map.get(rel, f"gcd_h_single_{idx_gcd:06d}")
                idx_gcd += 1
                raw_gcd_h.append({
                    "dataset": "gcd",
                    "path": rel,
                    "source_class": raw_c,
                    "harmonized_class": h_c,
                    "label": h_to_idx[h_c],
                    "group_id": gid,
                })

    # Group-partition GCD harmonized samples
    annotated_gcd_h = partition_samples_stratified_group(raw_gcd_h, seed=42)

    # Combined pool
    combined_samples = annotated_ccsn_h + annotated_gcd_h

    n_ccsn = len(annotated_ccsn_h)
    n_gcd = len(annotated_gcd_h)
    n_total = len(combined_samples)

    tr_tot = sum(1 for s in combined_samples if s["split"] == "train")
    va_tot = sum(1 for s in combined_samples if s["split"] == "val")
    te_tot = sum(1 for s in combined_samples if s["split"] == "test")

    print(f"    - Harmonized Pool: {n_total} images | CCSN: {n_ccsn} ({n_ccsn/n_total*100:.2f}%) | GCD: {n_gcd} ({n_gcd/n_total*100:.2f}%)")
    print(f"    - Partition: Train={tr_tot} ({tr_tot/n_total:.1%}) | Val={va_tot} ({va_tot/n_total:.1%}) | Test={te_tot} ({te_tot/n_total:.1%})")

    out_file_h = SPLITS_DIR / "harmonized_5bin_canonical.json"
    manifest_h = {
        "dataset_key": "harmonized_5bin",
        "taxonomy_name": "five-class cross-source compatibility taxonomy",
        "protocol": "grouped_stratified_holdout_v1.0",
        "seed": 42,
        "classes": HARMONIZED_CLASSES,
        "class_to_idx": h_to_idx,
        "num_classes": len(HARMONIZED_CLASSES),
        "mappings": {
            "ccsn_to_harmonized": CCSN_TO_HARMONIZED,
            "gcd_to_harmonized": GCD_TO_HARMONIZED,
        },
        "volume_dominance": {
            "ccsn_samples": n_ccsn,
            "ccsn_percentage": round(n_ccsn / n_total * 100.0, 2),
            "gcd_samples": n_gcd,
            "gcd_percentage": round(n_gcd / n_total * 100.0, 2),
            "total_samples": n_total,
        },
        "summary": {
            "total_samples": n_total,
            "train_samples": tr_tot,
            "val_samples": va_tot,
            "test_samples": te_tot,
            "ccsn": {
                "total": n_ccsn,
                "train": sum(1 for s in annotated_ccsn_h if s["split"] == "train"),
                "val": sum(1 for s in annotated_ccsn_h if s["split"] == "val"),
                "test": sum(1 for s in annotated_ccsn_h if s["split"] == "test"),
            },
            "gcd": {
                "total": n_gcd,
                "train": sum(1 for s in annotated_gcd_h if s["split"] == "train"),
                "val": sum(1 for s in annotated_gcd_h if s["split"] == "val"),
                "test": sum(1 for s in annotated_gcd_h if s["split"] == "test"),
            }
        },
        "samples": combined_samples,
    }
    manifest_h["sha256"] = compute_sha256(combined_samples)
    with open(out_file_h, "w", encoding="utf-8") as f:
        json.dump(manifest_h, f, indent=2)
    print(f"[+] Saved Harmonized manifest to: {out_file_h}")


def main():
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    build_ccsn_manifest()
    build_gcd_manifests()
    build_harmonized_manifest()
    print("\n" + "=" * 80)
    print("[SUCCESS] All 4 canonical manifests successfully generated with zero leakage!")
    print("=" * 80)


if __name__ == "__main__":
    main()

