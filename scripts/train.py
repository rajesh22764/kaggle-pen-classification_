#!/usr/bin/env python3
"""
Kaggle Pen Classification - Training Script
Usage: python train.py --model convnext_tiny --epochs 20
"""

import argparse
import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score
from tqdm import tqdm


class PenDataset(Dataset):
    def __init__(self, df, root_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.root_dir, row["image_path"])

        image = Image.open(img_path).convert("RGB")
        label = int(row["pen_id"]) - 1  # 0-based

        if self.transform:
            image = self.transform(image)

        return image, label


def get_model(model_name, num_classes=8):
    """Get the specified model with modified head"""
    if model_name == "convnext_tiny":
        model = models.convnext_tiny(weights="ConvNeXt_Tiny_Weights.DEFAULT")
        in_features = model.classifier[2].in_features
        model.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )
    elif model_name == "efficientnet_b4":
        model = models.efficientnet_b4(weights="EfficientNet_B4_Weights.DEFAULT")
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )
    elif model_name == "resnet50":
        model = models.resnet50(weights="ResNet50_Weights.DEFAULT")
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )
    elif model_name == "swin_tiny":
        model = models.swin_t(weights="Swin_T_Weights.DEFAULT")
        in_features = model.head.in_features
        model.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for images, labels in tqdm(loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    return total_loss / len(loader), acc


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average="macro")
    rec = recall_score(all_labels, all_preds, average="macro")

    return total_loss / len(loader), acc, prec, rec


def main():
    parser = argparse.ArgumentParser(description="Train pen classification model")
    parser.add_argument("--model", type=str, required=True,
                       choices=["convnext_tiny", "efficientnet_b4", "resnet50", "swin_tiny"],
                       help="Model architecture to train")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience")
    parser.add_argument("--data_dir", type=str, default="", help="Data directory")

    args = parser.parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load and split data
    df = pd.read_csv("train.csv")
    gss = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
    train_idx, val_idx = next(gss.split(df, groups=df["writer_id"]))
    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]

    print(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}")

    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(10),
        transforms.RandomApply([
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05))
        ], p=0.5),
        transforms.ColorJitter(contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # DataLoaders
    train_dataset = PenDataset(train_df, args.data_dir, train_transform)
    val_dataset = PenDataset(val_df, args.data_dir, val_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

    # Model
    model = get_model(args.model)
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    best_val_loss = float("inf")
    counter = 0

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_prec, val_rec = evaluate(model, val_loader, criterion, device)

        print(".4f")
        scheduler.step()

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            model_path = f"models/best_model_{args.model}.pth"
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), model_path)
            print(f"✅ Best model saved to {model_path}")
        else:
            counter += 1
            print(f"Early stopping counter: {counter}/{args.patience}")
            if counter >= args.patience:
                print("⛔ Early stopping triggered")
                break

    print("Training completed!")


if __name__ == "__main__":
    main()