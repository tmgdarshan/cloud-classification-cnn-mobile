"""
Dataset Merger Script for Cloud Classification

Description:
This script merges two distinct datasets (GCD and CCSN) into a single
unified structure for training a ResNet model.

1. GCD (Ground-based Cloud Dataset):
   - Already has Train/Test folders.
   - We copy these structure-wise as is.

2. CCSN (Cloud Classification Segmentation Noise):
   - Contains only raw class folders (no train/test split).
   - We map its specific classes (like 'ac', 'ci') to the GCD full names.
   - We treat non-matching classes (like 'ns', 'st') as 'mixed'.
   - We perform a manual 80/20 stratified split for training and validation.

Usage:
Update the 'raw_dir' paths in the config section and run.
The script ensures a reproducible split using a fixed random seed.
"""

import os
import shutil
import random

# --- Configuration ---
# Base directory where data currently lives
base_dir = '/home/snufkin/PycharmProjects/cloud-classification-cnn-mobile/data/raw'

# Input Directories
gcd_path = os.path.join(base_dir, 'processed_GCD')
ccsn_path = os.path.join(base_dir, 'CCSN_processed')

# Output Directory (Where the merged data will go)
output_path = os.path.join(base_dir, 'merged_dataset')

# Class Mapping: CCSN Code -> GCD Full Name
# Any CCSN class NOT in this dictionary will be automatically mapped to 'mixed'
class_map = {
    'ac': 'altocumulus',
    'cb': 'cumulonimbus',
    'ci': 'cirrus',
    'cu': 'cumulus',
    'sc': 'stratocumulus'
}


def merge_datasets():
    # Set a fixed seed so our "random" split is the same every time we run this.
    # This is important for scientific reproducibility.
    random.seed(42)

    # 1. Clean Setup: Remove the old merged folder if it exists
    if os.path.exists(output_path):
        print(f"Cleaning up old directory: {output_path}")
        shutil.rmtree(output_path)

    # Create the new Train/Test structure
    for split in ['train', 'test']:
        os.makedirs(os.path.join(output_path, split), exist_ok=True)

    print("--- Starting Merger ---")

    # ---------------------------------------------------------
    # PART 1: Process GCD Data
    # GCD already has a trusted train/test split, so we copy it directly.
    # ---------------------------------------------------------
    for split in ['train', 'test']:
        current_gcd_path = os.path.join(gcd_path, split)

        # Skip if folder doesn't exist (safety check)
        if not os.path.exists(current_gcd_path):
            continue

        print(f"Processing GCD {split} data...")

        for class_name in os.listdir(current_gcd_path):
            src_dir = os.path.join(current_gcd_path, class_name)
            dst_dir = os.path.join(output_path, split, class_name)

            os.makedirs(dst_dir, exist_ok=True)

            # Copy all images
            for img_file in os.listdir(src_dir):
                # Add prefix to avoid filename conflicts
                new_filename = f"gcd_{img_file}"
                shutil.copy2(os.path.join(src_dir, img_file),
                             os.path.join(dst_dir, new_filename))

    # ---------------------------------------------------------
    # PART 2: Process CCSN Data
    # CCSN needs to be mapped and split manually.
    # ---------------------------------------------------------
    print("Processing CCSN data (with 80/20 split)...")

    if os.path.exists(ccsn_path):
        for folder_name in os.listdir(ccsn_path):
            src_dir = os.path.join(ccsn_path, folder_name)

            # Skip if it's a file, not a folder
            if not os.path.isdir(src_dir):
                continue

            # Determine the target class name
            # If it's not in our map (like 'ns'), it goes to 'mixed'
            if folder_name in class_map:
                target_class = class_map[folder_name]
            else:
                target_class = 'mixed'

            # Get list of all images
            images = [f for f in os.listdir(src_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

            # Shuffle them randomly (but reproducibly because of seed=42)
            random.shuffle(images)

            # Calculate the split index for 80% training
            split_idx = int(len(images) * 0.8)
            train_files = images[:split_idx]
            test_files = images[split_idx:]

            # Helper function to copy a list of files to destination
            def copy_files(file_list, split_type):
                dst_dir = os.path.join(output_path, split_type, target_class)
                os.makedirs(dst_dir, exist_ok=True)
                for img in file_list:
                    new_filename = f"ccsn_{img}"
                    shutil.copy2(os.path.join(src_dir, img),
                                 os.path.join(dst_dir, new_filename))

            # Perform the copy
            copy_files(train_files, 'train')
            copy_files(test_files, 'test')

    print(f"Success! Merged dataset created at: {output_path}")


if __name__ == "__main__":
    merge_datasets()