# Pen Classification Under Distribution Shift

This project addresses an 8-class image classification task for identifying pen types from handwritten images. Developed as part of a Kaggle competition, it achieved a **top 30% leaderboard rank (116/389)**.

------------------------------------------------------------------------

## 🏆 Competition Details

- **Platform**: Kaggle
- **Task**: Multi-class image classification (8 classes)
- **Objective**: Identify pen type from handwritten stroke images
- **Evaluation Metric**: Accuracy
- **Leaderboard Rank**: 116 / 389 (Top ~30%)
- **Competition Link**: [ICDAR 2026 CircleID Pen Classification](https://www.kaggle.com/competitions/icdar-2026-circleid-pen-classification)

------------------------------------------------------------------------

## ⚠️ Key Insight: Data Leakage

> Initial results were misleading due to **data leakage**.

- The dataset includes a `writer_id` field
- A standard random split caused the **same writers to appear in both train and validation sets**
- The model learned **writer-specific patterns instead of pen characteristics**

### Solution

- Used **GroupShuffleSplit** (grouped by `writer_id`)
- Ensured **no writer overlap between train and validation**

### Impact

| Setup | Validation Accuracy | Interpretation |
|-------|---------------------|-----------------|
| Random Split | ~96% | ❌ Misleading (leakage) |
| Group Split | ~89–90% | ✅ Real generalization |

------------------------------------------------------------------------

## 🧠 Problem Overview

- **Task**: Classify pen type (8 classes)
- **Input**: RGB images of handwritten strokes
- **Challenges**:
  - Strong **writer-dependent patterns**
  - Subtle visual differences between pen types
  - Distribution shift between training and test data

------------------------------------------------------------------------

## 🏗️ Approach

### Models Explored

- **ResNet50** – Strong CNN baseline
- **ConvNeXt-Tiny** – Modern convolutional architecture
- **EfficientNet-B4** – Scaled CNN for fine-grained features
- **Swin-Tiny** – Transformer-based vision model

------------------------------------------------------------------------

### Training Strategy

- Group-based splitting to prevent leakage
- Moderate data augmentation:
  - Rotation
  - Affine transforms
  - Contrast jitter
- Early stopping
- Cosine learning rate scheduling
- Label smoothing

------------------------------------------------------------------------

### Final Solution

- Ensemble of multiple architectures
- Soft voting (logit averaging)
- Test-Time Augmentation (horizontal flip)

------------------------------------------------------------------------

## 📈 Results

| Model | Validation Accuracy |
|-------|---------------------|
| ResNet50 | ~89% |
| ConvNeXt-Tiny | ~89% |
| EfficientNet-B4 | ~89% |
| Swin-Tiny | ~89% |
| **Ensemble** | **~89.9%** |

> Ensemble gains were modest due to high correlation between model errors.

------------------------------------------------------------------------

## 🧪 Key Learnings

- **Data leakage can completely invalidate results**
- Validation strategy is often more important than model choice
- High accuracy does not guarantee real-world performance
- Ensemble effectiveness depends on **model diversity**
- Domain shift (writer variation) is a major challenge in practical ML systems

------------------------------------------------------------------------

## 📁 Project Structure

```
kaggle-pen-classification/
├── notebooks/                          # Jupyter notebooks for model experimentation
│   ├── resnet34.ipynb
│   ├── resnet50_.ipynb
│   ├── convnext_tiny.ipynb
│   ├── efficientnetb4.ipynb
│   ├── swin_tiny.ipynb
│   ├── resent34_da.ipynb               # ResNet34 with data augmentation
│   └── ensemble.ipynb                  # Ensemble model combining multiple architectures
├── scripts/                            # Python scripts for training and inference
│   ├── train.py                        # Model training pipeline
│   └── inference.py                    # Inference/prediction script
├── submissions/                        # Kaggle submission files
│   ├── submission_resnet50.csv
│   ├── submission_convnext_tiny.csv
│   ├── submission_efficientnet_b4.csv
│   ├── submission_swin_tiny.csv
│   └── submission_ensemble.csv
├── models/                             # Directory for trained model weights (currently empty)
├── train.csv                           # Training dataset
├── test.csv                            # Test dataset
├── requirements.txt                    # Python package dependencies
├── README.md                           # Project documentation
├── .gitignore                          # Git ignore patterns
├── .gitattributes                      # Git attributes
└── .git/                               # Git repository metadata
```

------------------------------------------------------------------------

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- CUDA (optional, for GPU acceleration)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/rajesh22764/kaggle-pen-classification.git
   cd kaggle-pen-classification
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

**Training:**
```bash
python scripts/train.py
```

**Inference:**
```bash
python scripts/inference.py
```

**Notebooks:**
- Open any notebook in `notebooks/` for detailed model exploration and training logs
- Start with `notebooks/ensemble.ipynb` for the final ensemble approach

------------------------------------------------------------------------

## 🎯 Key Takeaway

> A model is only as good as its evaluation strategy. Proper validation design (group-based splitting) revealed the true performance of our models and prevented overly optimistic assessments.

