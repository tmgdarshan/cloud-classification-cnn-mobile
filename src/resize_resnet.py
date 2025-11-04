import os
from PIL import Image

# Define datasets - both resize to 224x224 for ResNet
datasets = {
    "ccsn_v2": {
        "input_dir": "/home/snufkin/PycharmProjects/cloud-classification-cnn-mobile/data/raw/CCSN_v2",
        "output_dir": "/home/snufkin/PycharmProjects/cloud-classification-cnn-mobile/data/raw/CCSN_processed",
        "size": (224, 224),  # ResNet standard
    },
    "gcd": {
        "input_dir": "/home/snufkin/PycharmProjects/cloud-classification-cnn-mobile/data/raw/GCD/train",
        "output_dir": "/home/snufkin/PycharmProjects/cloud-classification-cnn-mobile/data/raw/processed_GCD/train",
        "size": (224, 224),  # ResNet standard
    }
    "gcd": {
        "input_dir": "/home/snufkin/PycharmProjects/cloud-classification-cnn-mobile/data/raw/GCD/test",
        "output_dir": "/home/snufkin/PycharmProjects/cloud-classification-cnn-mobile/data/raw/processed_GCD/test",
        "size": (224, 224),  # ResNet standard
    },
}

# Process each dataset
for dataset_name, config in datasets.items():
    print(f"\nProcessing {dataset_name}...")
    input_dir = config["input_dir"]
    output_dir = config["output_dir"]
    target_size = config["size"]

    # Loop through each class
    for class_name in os.listdir(input_dir):
        class_input_path = os.path.join(input_dir, class_name)
        class_output_path = os.path.join(output_dir, class_name)

        # Skip if not a directory
        if not os.path.isdir(class_input_path):
            continue

        # Create output class folder
        os.makedirs(class_output_path, exist_ok=True)

        # Get image files
        image_files = [
            f
            for f in os.listdir(class_input_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        print(f"  {class_name}: {len(image_files)} images")

        # Resize each image
        for i, img_name in enumerate(image_files):
            try:
                img_path = os.path.join(class_input_path, img_name)
                img = Image.open(img_path).convert("RGB")
                img_resized = img.resize(target_size, Image.LANCZOS)

                output_path = os.path.join(class_output_path, img_name)
                img_resized.save(output_path, "JPEG", quality=95)

                # Progress update every 100 images
                if (i + 1) % 100 == 0:
                    print(f"    Processed {i + 1}/{len(image_files)} images")
            except Exception as e:
                print(f"    Error with {img_name}: {e}")

        print(f"Completed {class_name}")

print("\n All datasets resized to 224x224 for ResNet!")
