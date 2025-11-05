import os
import numpy as np
from PIL import Image


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


mean, std = compute_mean_std(
    "/home/snufkin/PycharmProjects/cloud-classification-cnn-mobile/data/raw/processed_GCD/test"
)
print(f"Test set mean: {mean}")
print(f"Test set std: {std}")
