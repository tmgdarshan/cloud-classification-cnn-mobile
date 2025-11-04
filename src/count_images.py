import os

Path = "/home/snufkin/PycharmProjects/cloud-classification-cnn-mobile/data/raw/CCSN_v2"
base_dir = "%s" % Path  # Set this to your dataset directory
for class_name in os.listdir(base_dir):
    class_folder = os.path.join(base_dir, class_name)
    num_images = len(
        [f for f in os.listdir(class_folder) if f.lower().endswith(".jpg")]
    )
    print(f"{class_name}: {num_images} images")
