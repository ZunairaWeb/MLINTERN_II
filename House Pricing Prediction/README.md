
# Project Title

A brief description of what this project does and who it's for

markdown
# 🏠 House Price Prediction (Fusion Model)

This repository contains a **multimodal machine learning project** that predicts house sale prices by combining **tabular features** (structured dataset) with **visual features** (room images).  
The model fuses a **tabular MLP** with **CNN image embeddings** extracted from ResNet18.

---

## 📂 Project Structure

House Pricing Prediction/
│
├── Data/
│   └── house-prices-advanced-regression-techniques/
│       ├── train.csv
│       ├── test.csv
│       └── sample_submission.csv
│
├── images/
│   └── House_Room_Dataset/
│       ├── Bathroom/
│       ├── Bedroom/
│       ├── Dinning/
│       ├── Kitchen/
│       └── Livingroom/
│
├── fusion_model.py   # Main training script
├── requirements.txt  # Dependencies
└── README.md         # Documentation

Code

---

## ⚙️ Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/ZunairaWeb/MLINTERN_II.git
cd House Pricing Prediction
pip install -r requirements.txt
requirements.txt

Code
pandas
numpy
scikit-learn
torch
torchvision
Pillow
🚀 Workflow
Data Cleaning

Drop irrelevant columns (PoolQC, MiscFeature, Alley, Fence).

Fill missing numeric values with median.

Fill missing categorical values with mode.

Feature Engineering

One‑hot encode categorical features.

Align train/test features.

Split into train/validation sets.

Image Preprocessing

Resize images to 224×224.

Convert to tensors.

Extract CNN features using ResNet18 pretrained on ImageNet.

Fusion Model

Tabular branch: MLP → 64‑dimensional features.

Image branch: ResNet18 → 512‑dimensional features.

Fusion regressor: Concatenate (64 + 512) → predict SalePrice.

Training

Optimizer: Adam

Loss: MSE

Metrics: MAE, RMSE

Submission

Predict house prices on test set.

Save predictions in submission.csv.

📊 Example Training Log
Code
Epoch 1/5 - Train Loss: 1.23e+09, Val Loss: 1.10e+09, MAE: 24567.89, RMSE: 33210.45
Epoch 2/5 - Train Loss: 9.87e+08, Val Loss: 9.45e+08, MAE: 23012.34, RMSE: 30789.12
...
Fusion Validation MAE: 22567.12
Fusion Validation RMSE: 29876.45