"""
Cloud Classification -

2025-11-15:
- I switched the model from ResNet18 to ResNet34 for more learning capacity.
- Increased NUM_EPOCHS from 10 to 20 to allow better convergence.
- Set NUM_WORKERS=16 for faster DataLoader performance on my i9-12900K.
- Used batch_size=32 to fit comfortably in RAM/GPU.
- Augmentation is 'basic' (random flip, rotation).
- Data is on SSD; GPU is RTX 3070 Ti for quick training, but supports CPU fallback.
- CCSN and GCD both supported; I set DATASET='gcd' or 'ccsn' at the top before each run.
- Validation accuracy improved after all changes.
2025-11-15:
- I trained ResNet34 on the GCD dataset with 7 output classes, batch size 32, and basic augmentation (flip, rotation), for 20 epochs. Learning rate was 0.001 and weight_decay was set to 0.0.
- My train accuracy steadily rose above 91%, but validation accuracy plateaued at ~74%. The training loss dropped well below 0.25 yet val acc did not improve further.
- This gap between train and validation (over 15%) indicates overfitting ([web:605], [web:601]).
- According to the literature, increasing augmentation and applying regularization (weight_decay, stronger augmentation) should help ([web:606], [web:600], [web:602]).
- For this run I am:
    - Using stronger augmentation: horizontal/vertical flips, color jitter, random resized crop, and rotation ([web:606]).
    - Adding weight_decay=1e-4 to my optimizer ([web:605], [web:600]).
    - Lowering the learning rate to 0.0001, since validation is not improving with low training loss ([web:605]).

"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
import pandas as pd
import platform

# ---------- CONFIGURABLE PARAMETERS ---------- #
DATASET        = 'gcd'         # 'gcd' or 'ccsn'
BATCH_SIZE     = 32
NUM_EPOCHS     = 20
LEARNING_RATE  = 0.0001        # Lowered for better generalization [web:605]
WEIGHT_DECAY   = 1e-4          # L2 regularization to decrease overfitting [web:600]
MODEL_NAME     = 'resnet34'
AUGMENTATION   = 'strong'
NUM_WORKERS    = 0             # 0 to avoid BrokenPipeError, increase for faster loading if robust
PIN_MEMORY     = torch.cuda.is_available()

if DATASET == 'gcd':
    num_classes = 7
    mean = [0.48257098, 0.61535535, 0.76705415]
    std = [0.1954099, 0.1282709, 0.13690854]
    train_dir = '/home/snufkin/PycharmProjects/cloud-classification-cnn-mobile/data/raw/processed_GCD/train'
    test_dir  = '/home/snufkin/PycharmProjects/cloud-classification-cnn-mobile/data/raw/processed_GCD/test'
    csv_name  = 'gcd_training_metrics_resnet34_reg.csv'
    weightfile = 'gcd_resnet34_reg.pth'
elif DATASET == 'ccsn':
    num_classes = 11
    mean = [0.4798462, 0.52342568, 0.56209201]
    std = [0.26025242, 0.23717379, 0.25529793]
    train_dir = '/home/snufkin/PycharmProjects/cloud-classification-cnn-mobile/data/raw/CCSN_processed'
    test_dir  = None
    csv_name  = 'ccsn_training_metrics_resnet34_reg.csv'
    weightfile = 'ccsn_resnet34_reg.pth'
else:
    raise ValueError("Unknown dataset: choose 'gcd' or 'ccsn'.")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"I am using device: {device} ({platform.platform()})")
print(f"DataLoader uses num_workers={NUM_WORKERS}, pin_memory={PIN_MEMORY}")

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

# ----------- DATASET AND LOADER SETUP ------------- #
train_dataset = ImageFolder(train_dir, transform=train_transform)
if DATASET == 'gcd':
    test_dataset = ImageFolder(test_dir, transform=test_transform)
else:
    total = len(train_dataset)
    train_len = int(0.8 * total)
    val_len = total - train_len
    generator = torch.Generator().manual_seed(42)
    train_dataset, test_dataset = random_split(
        train_dataset, [train_len, val_len], generator=generator
    )
    train_dataset.dataset.transform = train_transform
    test_dataset.dataset.transform = test_transform

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                         num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

# ----------- MODEL, LOSS, OPTIMIZER SETUP ---------- #
model = getattr(models, MODEL_NAME)(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)
print(f"Model: {MODEL_NAME} with {num_classes} output classes.")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# ----------- TRAINING AND VALIDATION LOOP ------------ #
train_losses, test_losses = [], []
train_accs, test_accs = [], []

for epoch in range(NUM_EPOCHS):
    model.train()
    epoch_loss, correct, total = 0, 0, 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
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

# I always update my docstring "Experiment Log" with date, settings, and results after each run.
