"""Discover, validate, and inventory a dataset (Phase 3).

Reads a dataset directory and writes a deterministic inventory to
``metadata/<key>_inventory.json``, then prints a human-readable report. It never
modifies data and never interprets folder names as scientific classes.

Usage:
    # data root comes from the environment profile ($CLOUD_DATA_ROOT)
    python scripts/discover_dataset.py --dataset ccsn --relative-path CCSN_v2
    python scripts/discover_dataset.py --dataset gcd  --relative-path GCD

The dataset's `relative_data_path` is intentionally NOT yet in the config for the
source datasets (it is approved after discovery), so pass --relative-path for
those. If a config already has `relative_data_path`, it is used by default.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config_loader import ConfigError, load_dataset, load_environment  # noqa: E402
from dataset_discovery import DiscoveryError, discover_dataset, to_json  # noqa: E402

_METADATA_DIR = Path(__file__).resolve().parent.parent / "metadata"


def _resolve_scan_path(args, data_root: str, dataset_cfg: dict) -> Path:
    if args.path:
        return Path(args.path)
    if args.relative_path:
        return Path(data_root) / args.relative_path
    configured = dataset_cfg.get("relative_data_path")
    if configured:
        return Path(data_root) / configured
    raise DiscoveryError(
        f"Dataset '{args.dataset}' has no 'relative_data_path' in its config yet "
        "(approved after discovery). Pass --relative-path <dir-under-data-root> "
        "or --path <absolute-path>."
    )


def _print_report(inv: dict) -> None:
    s = inv["summary"]
    st = inv["structure"]
    print(f"\nDataset inventory: {inv['dataset_label']}  (key: {inv['dataset_key']})")
    print(f"Scanned: {inv['scanned_path']}")
    print("-" * 68)
    print(f"Images: {s['num_images']}   Non-image files: {s['num_non_image_files']}   "
          f"Total files: {s['num_files']}")
    print(f"Corrupt images: {s['num_corrupt_images']}   "
          f"Duplicate groups: {s['num_duplicate_groups']} "
          f"({s['num_duplicate_extra_files']} extra copies)")
    dup = inv["duplicates"]
    if dup["cross_split_applicable"]:
        n = dup["num_cross_split_groups"]
        flag = "   <-- DATA LEAKAGE RISK" if n else ""
        print(f"Cross-split duplicate groups (train<->test): {n}{flag}")
    else:
        print("Cross-split duplicate check: n/a (no splits)")
    print(f"Detected splits: {st['detected_splits'] or 'none'}")
    print("\nClass folders (names AS FOUND -- not interpreted):")
    cf = st["class_folders"]
    if st["detected_splits"]:
        for split, classes in cf.items():
            print(f"  [{split}] {len(classes)} folders")
            for name, count in classes.items():
                print(f"      {name:<16} {count}")
    else:
        for name, count in cf.items():
            print(f"      {name:<16} {count}")
    if st["images_without_class_folder"]:
        print(f"  (images not inside a class folder: {st['images_without_class_folder']})")
    print("\nObserved image dimensions:")
    for dim, count in sorted(inv["images"]["dimensions"].items(), key=lambda kv: (-kv[1], kv[0]))[:8]:
        print(f"      {dim:<12} {count}")
    print(f"\nFormats: {inv['images']['formats']}   Modes: {inv['images']['modes']}")
    fs = inv["file_sizes_bytes"]
    if fs:
        print(f"File sizes (bytes): min={fs['min']} median={fs['median']} "
              f"mean={fs['mean']} max={fs['max']}")
    if inv["corrupt_images"]:
        print(f"\nFirst corrupt files: {[c['path'] for c in inv['corrupt_images'][:5]]}")
    if dup["cross_split_groups"]:
        print("\nCross-split duplicate examples (leakage):")
        for grp in dup["cross_split_groups"][:5]:
            print(f"      splits={grp['splits']}  {grp['paths']}")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Discover and inventory a dataset (read-only).")
    parser.add_argument("--environment", "-e", default=os.environ.get("CLOUD_ENV", "local"))
    parser.add_argument("--dataset", "-d", required=True, help="Dataset config key (e.g. ccsn, gcd).")
    parser.add_argument("--relative-path", "-r", help="Path under the data root to scan.")
    parser.add_argument("--path", "-p", help="Absolute path to scan (overrides --relative-path).")
    parser.add_argument("--output", "-o", help="Output JSON path (default metadata/<key>_inventory.json).")
    parser.add_argument("--quiet", "-q", action="store_true", help="Do not print the report.")
    args = parser.parse_args(argv)

    try:
        env = load_environment(args.environment)
        dataset_cfg = load_dataset(args.dataset)
        scan_path = _resolve_scan_path(args, env["data_root"], dataset_cfg)
        inventory = discover_dataset(scan_path, key=args.dataset, label=dataset_cfg.get("name"))
    except (ConfigError, DiscoveryError) as exc:
        print(f"Discovery error: {exc}", file=sys.stderr)
        return 1

    out_path = Path(args.output) if args.output else _METADATA_DIR / f"{args.dataset}_inventory.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(to_json(inventory), encoding="utf-8")

    if not args.quiet:
        _print_report(inventory)
    print(f"Wrote inventory: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
