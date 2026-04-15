#!/usr/bin/env python3
"""
Kaggle Pen Classification - Inference Script
Usage: python inference.py --model convnext_tiny
"""

import argparse
import os
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from tqdm import tqdm


class TestDataset(Dataset):
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

        if self.transform:
            image = self.transform(image)

        return image, row["image_id"]


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


def load_and_predict(model_name, test_df, data_dir="", device="cuda"):
    """Load model and make predictions"""
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Load model
    model = get_model(model_name)
    model_path = f"models/best_model_{model_name}.pth"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # Test transform
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # DataLoader
    test_dataset = TestDataset(test_df, data_dir, test_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=8)

    # Inference with TTA
    predictions = []
    image_ids = []

    with torch.no_grad():
        for images, ids in tqdm(test_loader, desc=f"Predicting with {model_name}"):
            images = images.to(device)

            # Original
            outputs = model(images)

            # Horizontal flip
            flipped_images = torch.flip(images, dims=[3])
            outputs_flipped = model(flipped_images)

            # Vertical flip
            flipped_v_images = torch.flip(images, dims=[2])
            outputs_v = model(flipped_v_images)

            # Average
            outputs = (outputs + outputs_flipped + outputs_v) / 3
            preds = torch.argmax(outputs, dim=1)

            predictions.extend(preds.cpu().numpy())
            image_ids.extend(ids)

    # Convert to 1-based labels
    predictions = [p + 1 for p in predictions]

    return image_ids, predictions


def main():
    parser = argparse.ArgumentParser(description="Run inference with trained model")
    parser.add_argument("--model", type=str, required=True,
                       choices=["convnext_tiny", "efficientnet_b4", "resnet50", "swin_tiny", "ensemble"],
                       help="Model architecture or ensemble")
    parser.add_argument("--data_dir", type=str, default="", help="Data directory")
    parser.add_argument("--output", type=str, default=None, help="Output CSV file")

    args = parser.parse_args()

    # Load test data
    test_df = pd.read_csv("test.csv")

    if args.model == "ensemble":
        # Load all models for ensemble
        models_list = ["convnext_tiny", "efficientnet_b4", "resnet50", "swin_tiny"]
        all_predictions = []

        for model_name in models_list:
            print(f"Loading {model_name}...")
            image_ids, preds = load_and_predict(model_name, test_df, args.data_dir)
            all_predictions.append(preds)

        # Average predictions (simple voting)
        import numpy as np
        ensemble_preds = []
        for i in range(len(all_predictions[0])):
            votes = [pred[i] for pred in all_predictions]
            # Take most common prediction
            ensemble_preds.append(max(set(votes), key=votes.count))

        predictions = ensemble_preds
        image_ids = image_ids  # Same for all models

    else:
        # Single model
        image_ids, predictions = load_and_predict(args.model, test_df, args.data_dir)

    # Create submission
    submission = pd.DataFrame({
        "image_id": image_ids,
        "pen_id": predictions
    })

    # Set output filename
    if args.output is None:
        if args.model == "ensemble":
            output_file = "submissions/submission_ensemble.csv"
        else:
            output_file = f"submissions/submission_{args.model}.csv"
    else:
        output_file = args.output

    os.makedirs("submissions", exist_ok=True)
    submission.to_csv(output_file, index=False)

    print(f"✅ Submission saved to {output_file}")
    print(f"Total predictions: {len(predictions)}")
    print(f"Unique predictions: {len(set(predictions))}")


if __name__ == "__main__":
    main()