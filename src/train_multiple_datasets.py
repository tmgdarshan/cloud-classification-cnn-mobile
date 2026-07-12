"""
Cloud Classification training script for the merged CCSN + GCD dataset.

The merged dataset already has train/test folders, so this script trains
directly from that structure.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
import pandas as pd
import platform

# ---------- CONFIGURABLE PARAMETERS ---------- #
DATASET        = 'merged'      # merged dataset only
BATCH_SIZE     = 16
NUM_EPOCHS     = 10
LEARNING_RATE  = 0.001
WEIGHT_DECAY   = 1e-3
MODEL_NAME     = 'resnet18'
AUGMENTATION   = 'none'
# Tune workers for faster data loading (safe default: up to 8)
NUM_WORKERS = 0
PIN_MEMORY = torch.cuda.is_available()

BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'raw')
train_dir = os.path.join(BASE_DIR, 'merged_dataset', 'train')
test_dir = os.path.join(BASE_DIR, 'merged_dataset', 'test')
csv_name = 'merged_training_metrics_resnet34_reg.csv'
weightfile = 'merged_resnet34_reg.pth'

# Use ImageNet normalization for the pretrained ResNet backbone.
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Merged dataset contains six classes: five shared cloud types plus clearsky.
num_classes = 6

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"I am using device: {device} ({platform.platform()})")
print(f"DataLoader uses num_workers={NUM_WORKERS}, pin_memory={PIN_MEMORY}")

# Improve cudnn performance for fixed-size inputs (ResNet 224x224)
if device.type == 'cuda':
    torch.backends.cudnn.benchmark = True

# ----------- DATA AUGMENTATION/TRANSFORMS ---------- #
if AUGMENTATION == 'none':
    train_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
elif AUGMENTATION == 'basic':
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
elif AUGMENTATION == 'strong':
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
else:
    raise ValueError("Unknown AUGMENTATION: choose 'none', 'basic', 'strong'.")

test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])


def main():
    # ----------- DATASET AND LOADER SETUP ------------- #
    train_dataset = ImageFolder(train_dir, transform=train_transform)
    test_dataset = ImageFolder(test_dir, transform=test_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )

    # ----------- MODEL, LOSS, OPTIMIZER SETUP ---------- #
    model = getattr(models, MODEL_NAME)(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)
    print(f"Model: {MODEL_NAME} with {num_classes} output classes.")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Use automatic mixed precision when CUDA is available to speed up training
    #use_amp = device.type == 'cuda'
    #scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # ----------- TRAINING AND VALIDATION LOOP ------------ #
    train_losses, test_losses = [], []
    train_accs, test_accs = [], []

    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss, correct, total = 0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()

            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)

                assert scaler is not None
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            epoch_loss += loss.item() * imgs.size(0)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        train_losses.append(epoch_loss / total)
        train_accs.append(100. * correct / total)

        model.eval()
        test_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                test_loss += loss.item() * imgs.size(0)
                _, preds = outputs.max(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        test_losses.append(test_loss / total)
        test_accs.append(100. * correct / total)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Acc: {train_accs[-1]:.2f}% | Val Acc: {test_accs[-1]:.2f}% | Train Loss: {train_losses[-1]:.4f}")

    # ------------- SAVE METRICS/WEIGHTS ------------- #
    results = pd.DataFrame({
        'epoch': list(range(1, NUM_EPOCHS+1)),
        'train_loss': train_losses,
        'val_loss': test_losses,
        'train_acc': train_accs,
        'val_acc': test_accs
    })
    results.to_csv(csv_name, index=False)
    print(f"Saved metrics to {csv_name}")

    torch.save(model.state_dict(), weightfile)
    print(f"Saved model weights to {weightfile}")


if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    main()

