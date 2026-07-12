"""
Merge CCSN v2 and GCD into one dataset using a simple folder-based rule.

GCD already has train/test folders, so we keep that split.
CCSN v2 is raw, so we split each class 80/20 first, then map only the
shared classes into the merged dataset.
"""

import os
import random
import shutil


BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")
GCD_PATH = os.path.join(BASE_DIR, "processed_GCD")
CCSN_PATH = os.path.join(BASE_DIR, "CCSN_processed")
OUTPUT_PATH = os.path.join(BASE_DIR, "merged_dataset")

CLASS_MAP = {
    "ac": "2_altocumulus",
    "cb": "6_cumulonimbus",
    "ci": "3_cirrus",
    "cu": "1_cumulus",
    "sc": "5_stratocumulus",
}

GCD_KEEP_CLASSES = {"1_cumulus", "2_altocumulus", "3_cirrus", "4_clearsky", "5_stratocumulus", "6_cumulonimbus"}
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
TRAIN_RATIO = 0.8


def copy_images(source_dir, file_list, destination_dir, prefix):
    os.makedirs(destination_dir, exist_ok=True)
    for file_name in file_list:
        shutil.copy2(
            os.path.join(source_dir, file_name),
            os.path.join(destination_dir, file_name),
        )


def merge_datasets():
    random.seed(42)

    if os.path.exists(OUTPUT_PATH):
        print(f"Cleaning up old directory: {OUTPUT_PATH}")
        shutil.rmtree(OUTPUT_PATH)

    os.makedirs(os.path.join(OUTPUT_PATH, "train"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_PATH, "test"), exist_ok=True)

    print("--- Starting Merger ---")

    for split in ["train", "test"]:
        gcd_split_path = os.path.join(GCD_PATH, split)
        if not os.path.exists(gcd_split_path):
            continue

        print(f"Processing GCD {split} data...")
        for class_name in os.listdir(gcd_split_path):
            if class_name not in GCD_KEEP_CLASSES:
                continue

            source_dir = os.path.join(gcd_split_path, class_name)
            if not os.path.isdir(source_dir):
                continue

            files = [f for f in os.listdir(source_dir) if f.lower().endswith(IMAGE_EXTENSIONS)]
            copy_images(source_dir, files, os.path.join(OUTPUT_PATH, split, class_name), "gcd")

    print("Processing CCSN data with class-wise 80/20 split...")
    if os.path.exists(CCSN_PATH):
        for class_name in os.listdir(CCSN_PATH):
            source_dir = os.path.join(CCSN_PATH, class_name)
            if not os.path.isdir(source_dir):
                continue

            class_key = class_name.lower()
            if class_key not in CLASS_MAP:
                continue

            target_class = CLASS_MAP[class_key]
            files = [f for f in os.listdir(source_dir) if f.lower().endswith(IMAGE_EXTENSIONS)]
            random.shuffle(files)

            split_index = int(len(files) * TRAIN_RATIO)
            train_files = files[:split_index]
            test_files = files[split_index:]

            copy_images(source_dir, train_files, os.path.join(OUTPUT_PATH, "train", target_class), "ccsn")
            copy_images(source_dir, test_files, os.path.join(OUTPUT_PATH, "test", target_class), "ccsn")

    print(f"Success! Merged dataset created at: {OUTPUT_PATH}")


if __name__ == "__main__":
    merge_datasets()