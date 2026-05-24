# -------------------------------
# 1. Import libraries
# -------------------------------
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# -------------------------------
# 2. Load tabular dataset
# -------------------------------
base_path = r"D:\Pprojects\MLintern_II\House Pricing Prediction\Data\house-prices-advanced-regression-techniques"
train_df = pd.read_csv(base_path + r"\train.csv")
test_df = pd.read_csv(base_path + r"\test.csv")
submission_df = pd.read_csv(base_path + r"\sample_submission.csv")

# -------------------------------
# 3. Clean tabular data
# -------------------------------
train_df = train_df.drop(["PoolQC", "MiscFeature", "Alley", "Fence"], axis=1)
test_df = test_df.drop(["PoolQC", "MiscFeature", "Alley", "Fence"], axis=1)

# Fill missing numeric values
nums_col = train_df.select_dtypes(include=["float64", "int64"]).columns
nums_col = [col for col in nums_col if col in test_df.columns]
for col in nums_col:
    train_df[col] = train_df[col].fillna(train_df[col].median())
    test_df[col] = test_df[col].fillna(test_df[col].median())

# Fill missing categorical values
cat_col = train_df.select_dtypes(include=["object"]).columns
for col in cat_col:
    train_df[col] = train_df[col].fillna(train_df[col].mode()[0])
    test_df[col]  = test_df[col].fillna(test_df[col].mode()[0])

# -------------------------------
# 4. Add image paths (corrected)
# -------------------------------
train_df['bathroom_img'] = train_df['Id'].apply(
    lambda x: f"D:/Pprojects/MLintern_II/House Pricing Prediction/images/House_Room_Dataset/Bathroom/bath_{x}.jpg"
)
train_df['bedroom_img'] = train_df['Id'].apply(
    lambda x: f"D:/Pprojects/MLintern_II/House Pricing Prediction/images/House_Room_Dataset/Bedroom/bed_{x}.jpg"
)
train_df['dining_img'] = train_df['Id'].apply(
    lambda x: f"D:/Pprojects/MLintern_II/House Pricing Prediction/images/House_Room_Dataset/Dinning/din_{x}.jpg"
)
train_df['kitchen_img'] = train_df['Id'].apply(
    lambda x: f"D:/Pprojects/MLintern_II/House Pricing Prediction/images/House_Room_Dataset/Kitchen/kitchen_{x}.jpg"
)
train_df['living_img'] = train_df['Id'].apply(
    lambda x: f"D:/Pprojects/MLintern_II/House Pricing Prediction/images/House_Room_Dataset/Livingroom/living_{x}.jpg"
)

# -------------------------------
# 5. Preprocess tabular features
# -------------------------------
X = pd.get_dummies(train_df.drop("SalePrice", axis=1))
X_test = pd.get_dummies(test_df)
X, X_test = X.align(X_test, join="left", axis=1, fill_value=0)
y = train_df["SalePrice"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------------
# 6. Image preprocessing
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def load_image(path):
    img = Image.open(path).convert("RGB")
    return transform(img)

# -------------------------------
# 7. CNN feature extractor
# -------------------------------
cnn = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)  # updated for new torchvision
cnn.fc = nn.Identity()  # remove final classification layer

# Example: extract features for one image
img_path = train_df['bathroom_img'].iloc[0]
sample_img = load_image(img_path).unsqueeze(0)
features = cnn(sample_img)
print("Image feature shape:", features.shape)  # e.g., torch.Size([1, 512])

# -------------------------------
# 8. Fusion model (simplified)
# -------------------------------
# Tabular branch (MLP)
tabular_model = nn.Sequential(
    nn.Linear(X_train.shape[1], 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU()
)

# Fusion: concatenate tabular + image features
fusion_regressor = nn.Sequential(
    nn.Linear(64 + 512, 128),
    nn.ReLU(),
    nn.Linear(128, 1)  # predict SalePrice
)
# -------------------------------
# 8b. Fusion training loop
# -------------------------------

# Function to extract features for all room images of a house
def extract_house_image_features(row):
    imgs = []
    for col in ['bathroom_img', 'bedroom_img', 'dining_img', 'kitchen_img', 'living_img']:
        try:
            img = load_image(row[col]).unsqueeze(0)
            feat = cnn(img).detach()  # 1x512
            imgs.append(feat)
        except:
            # If image missing, use zeros
            imgs.append(torch.zeros((1, 512)))
    # Average features across rooms
    house_feat = torch.mean(torch.cat(imgs, dim=0), dim=0, keepdim=True)
    return house_feat

# Precompute image features for train/val sets
train_image_feats = torch.cat([extract_house_image_features(train_df.iloc[i]) for i in X_train.index])
val_image_feats   = torch.cat([extract_house_image_features(train_df.iloc[i]) for i in X_val.index])

# Convert tabular data to tensors
X_train_tensor = torch.from_numpy(X_train.astype(np.float32).to_numpy())
y_train_tensor = torch.from_numpy(y_train.astype(np.float32).to_numpy()).reshape(-1, 1)

X_val_tensor = torch.from_numpy(X_val.astype(np.float32).to_numpy())
y_val_tensor = torch.from_numpy(y_val.astype(np.float32).to_numpy()).reshape(-1, 1)

# Define optimizer + loss
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(list(tabular_model.parameters()) + list(fusion_regressor.parameters()), lr=0.001)

# Training loop (simplified)
epochs = 5
for epoch in range(epochs):
    tab_feats = tabular_model(X_train_tensor)
    preds = fusion_regressor(torch.cat((tab_feats, train_image_feats), dim=1))
    loss = criterion(preds, y_train_tensor)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Validation
    with torch.no_grad():
        val_tab_feats = tabular_model(X_val_tensor)
        val_preds = fusion_regressor(torch.cat((val_tab_feats, val_image_feats), dim=1))
        val_loss = criterion(val_preds, y_val_tensor)
        mae = mean_absolute_error(y_val_tensor.numpy(), val_preds.numpy())
        rmse = np.sqrt(mean_squared_error(y_val_tensor.numpy(), val_preds.numpy()))
    
    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {loss.item():.2f}, Val Loss: {val_loss.item():.2f}, MAE: {mae:.2f}, RMSE: {rmse:.2f}")

# -------------------------------
# 9. Final Evaluation + Submission using Fusion Model
# -------------------------------
with torch.no_grad():
    # Precompute image features for test set
    test_image_feats = torch.cat([extract_house_image_features(test_df.iloc[i]) for i in X_test.index])
    X_test_tensor = torch.from_numpy(X_test.astype(np.float32).to_numpy())

    # Predict on validation set
    val_tab_feats = tabular_model(X_val_tensor)
    val_preds = fusion_regressor(torch.cat((val_tab_feats, val_image_feats), dim=1))
    mae = mean_absolute_error(y_val_tensor.numpy(), val_preds.numpy())
    rmse = np.sqrt(mean_squared_error(y_val_tensor.numpy(), val_preds.numpy()))
    print("Fusion Validation MAE:", mae)
    print("Fusion Validation RMSE:", rmse)

    # Predict on test set
    test_tab_feats = tabular_model(X_test_tensor)
    test_preds = fusion_regressor(torch.cat((test_tab_feats, test_image_feats), dim=1))
    submission_df["SalePrice"] = test_preds.numpy().flatten()
    print(submission_df.head())
