"""
=============================================================================
Script 4: PyTorch Temporal CNN Training (Experiment B - Raw 63 Landmarks)
=============================================================================
Purpose:
    This script executes Experiment B of the sequence modeling dataset. 
    Instead of using human-crafted neuro-symbolic features (angles/drift), 
    this script feeds the complete raw 3D hand skeleton (21 landmarks * 3 
    coordinates = 63 features) directly into the Temporal CNN.
    
    The goal is to determine if deep convolutional filters can independently 
    discover spatial-temporal representations that exceed the ~72% accuracy 
    ceiling of the handcrafted baseline.
    
Inputs:
    - 'p1_dataset_combined_raw.csv': The raw 66-column dataset from Script 1.
      
Outputs:
    - Console logs detailing Epoch Training/Validation Loss and Accuracy.
    - 'temporal_cnn_raw.pth': The saved weights of the trained network.
=============================================================================
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
import os

# ==========================================
# 1. Hyperparameters & Settings
# ==========================================
WINDOW_SIZE = 30       
NUM_FEATURES = 63      # UPDATE: 21 landmarks * (X, Y, Z) = 63 spatial features
NUM_CLASSES = 2        
BATCH_SIZE = 16
EPOCHS = 25
LEARNING_RATE = 0.001

# ==========================================
# 2. Sliding Window Generator
# ==========================================
def create_sliding_windows(df, window_size):
    X, y, groups = [], [], []
    
    # Isolate only the 63 landmark coordinate columns
    landmark_cols = [c for c in df.columns if 'landmark' in c]
    
    for video_name, group in df.groupby('video_name'):
        features = group[landmark_cols].values
        label = group['label'].values[0] 
        
        if len(features) >= window_size:
            for i in range(len(features) - window_size + 1):
                X.append(features[i : i + window_size])
                y.append(label)
                groups.append(video_name) 
                
    return np.array(X), np.array(y), np.array(groups)

# ==========================================
# 3. PyTorch Dataset Class
# ==========================================
class HandshakeDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).transpose(1, 2)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

# ==========================================
# 4. Temporal CNN Architecture (Widened for Raw Data)
# ==========================================
class TemporalCNN(nn.Module):
    def __init__(self, num_features, num_classes, window_size):
        super(TemporalCNN, self).__init__()
        
        # Widened the out_channels to handle 63 spatial dimensions
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels=num_features, out_channels=32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        flattened_length = window_size // 4
        self.fc_input_size = 64 * flattened_length
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.fc_input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        features = self.conv_block(x)
        return self.classifier(features)

# ==========================================
# 5. Main Execution & Training Loop
# ==========================================
if __name__ == "__main__":
    CSV_PATH = "p1_dataset_combined_raw.csv"
    
    if not os.path.exists(CSV_PATH):
        print(f"ERROR: Cannot find {CSV_PATH}.")
        exit()
        
    print("Loading raw data and performing Pandas Interpolation...")
    df = pd.read_csv(CSV_PATH)
    
    # --- CRITICAL: Repeat the interpolation on the raw dataset ---
    landmark_cols = [c for c in df.columns if 'landmark' in c]
    df[landmark_cols] = df[landmark_cols].replace(0.0, np.nan)
    df[landmark_cols] = df.groupby('video_name')[landmark_cols].transform(
        lambda x: x.interpolate(method='linear', limit_direction='both')
    )
    df[landmark_cols] = df[landmark_cols].fillna(0.0)
    # -------------------------------------------------------------
    
    print("Generating sliding windows...")
    X_all, y_all, groups_all = create_sliding_windows(df, WINDOW_SIZE)
    print(f"Total sequences generated: {len(X_all)}")
    
    gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=42)
    train_idx, test_idx = next(gss.split(X_all, y_all, groups_all))
    
    X_train, y_train = X_all[train_idx], y_all[train_idx]
    X_test, y_test = X_all[test_idx], y_all[test_idx]
    
    print(f"Training on {len(np.unique(groups_all[train_idx]))} videos ({len(X_train)} windows)")
    print(f"Testing on {len(np.unique(groups_all[test_idx]))} videos ({len(X_test)} windows)\n")
    
    train_loader = DataLoader(HandshakeDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(HandshakeDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)
    
    model = TemporalCNN(NUM_FEATURES, NUM_CLASSES, WINDOW_SIZE)
    
    class_counts = np.bincount(y_train)
    weights = len(y_train) / (len(class_counts) * class_counts)
    class_weights_tensor = torch.tensor(weights, dtype=torch.float32)
    
    print(f"Computed Class Weights: Class 0 = {weights[0]:.2f}, Class 1 = {weights[1]:.2f}\n")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("Starting Training Loop...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
        train_acc = 100 * train_correct / train_total
        
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
        val_acc = 100 * val_correct / val_total
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}] | "
                  f"Train Loss: {train_loss/len(train_loader):.4f} - Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss/len(test_loader):.4f} - Acc: {val_acc:.2f}%")

    torch.save(model.state_dict(), "temporal_cnn_raw.pth")
    print("\nTraining Complete! Model weights saved as 'temporal_cnn_raw.pth'.")
