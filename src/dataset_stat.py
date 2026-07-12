import os
import numpy as np
from PIL import Image
from pathlib import Path


"Calculate the mean and standard deviation of a dataset of images."
" This is useful for normalizing the dataset before training a neural network."
def compute_mean_std(image_folder):
    pixel_sum = np.zeros(3)
    pixel_sqsum = np.zeros(3)
    count = 0
    for class_folder in os.listdir(image_folder):
        path = os.path.join(image_folder, class_folder)
        if not os.path.isdir(path):
            continue
        for img_file in os.listdir(path):
            if img_file.lower().endswith(".jpg"):
                img_path = os.path.join(path, img_file)
                img = (
                    np.array(Image.open(img_path).convert("RGB")) / 255.0
                )  # normalization
                pixel_sum += img.mean(axis=(0, 1))
                pixel_sqsum += (img**2).mean(axis=(0, 1))
                count += 1
    mean = pixel_sum / count
    std = np.sqrt(pixel_sqsum / count - mean**2)
    return mean, std

# Define the path to the dataset and compute the mean and standard deviation
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
dataset_path = project_root / "data" / "raw" / "processed_GCD" / "test"
mean, std = compute_mean_std(dataset_path)
print(f"Test set mean: {mean}")
print(f"Test set std: {std}")
