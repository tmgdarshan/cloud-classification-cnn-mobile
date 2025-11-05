import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# Mean and std values
CCSN_std = [0.26025242, 0.23717379, 0.25529793]
CCSN_mean = [0.4798462, 0.52342568, 0.56209201]
GCD_train_std = [0.1954099, 0.1282709, 0.13690854]
GCD_train_mean = [0.48257098, 0.61535535, 0.76705415]

# GCD transforms and dataset
gcd_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=GCD_train_mean, std=GCD_train_std),
])
gcd_train_dataset = ImageFolder(
    "/home/snufkin/PycharmProjects/cloud-classification-cnn-mobile/data/raw/processed_GCD/train",
    transform=gcd_transform,
)
gcd_test_dataset = ImageFolder(
    "/home/snufkin/PycharmProjects/cloud-classification-cnn-mobile/data/raw/processed_GCD/test",
    transform=gcd_transform,
)

# CCSN transforms and dataset
ccsn_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=CCSN_mean, std=CCSN_std),
])
ccsn_dataset = ImageFolder(
    "/home/snufkin/PycharmProjects/cloud-classification-cnn-mobile/data/raw/CCSN_processed",
    transform=ccsn_transform,
)


# DataLoaders (set batch size as needed)
BATCH_SIZE = 32

gcd_train_loader = DataLoader(gcd_train_dataset, batch_size=BATCH_SIZE, shuffle=True)
gcd_test_loader = DataLoader(gcd_test_dataset, batch_size=BATCH_SIZE, shuffle=False)
ccsn_loader = DataLoader(ccsn_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Quick check: get a batch from one loader
images, labels = next(iter(gcd_train_loader))
print("GCD batch shape:", images.shape)  # Should be [BATCH_SIZE, 3, 224, 224]
