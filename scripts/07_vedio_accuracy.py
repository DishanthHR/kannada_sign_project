# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import random

# Configuration
MODEL_PATH = r"C:\Users\savem\OneDrive\Desktop\kannada_sign_project\model\checkpoints\best_model.pth"
CLASS_NAMES_PATH = r"C:\Users\savem\OneDrive\Desktop\kannada_sign_project\models\words_model\classes.txt"
VALIDATION_DIR = r"C:\Users\savem\OneDrive\Desktop\kannada_sign_project\dataset\new_processed_video\val"
TRAIN_DIR = r"C:\Users\savem\OneDrive\Desktop\kannada_sign_project\dataset\new_processed_video\train"
NUM_TEST_SAMPLES = 5

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ====================== MODEL ARCHITECTURE ======================
class KannadaSignModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        # Encoder layers (from state_dict)
        self.encoder = nn.Sequential(
            nn.Conv1d(126, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=64,  # Must match encoder output
            hidden_size=128,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),  # 256 because bidirectional (128*2)
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        # x shape: (batch, features, seq_len)
        x = self.encoder(x)
        
        # Permute for LSTM: (batch, seq_len, features)
        x = x.permute(0, 2, 1)
        
        # LSTM
        lstm_out, _ = self.lstm(x)
        
        # Classifier
        x = lstm_out[:, -1, :]  # Last timestep
        return self.classifier(x)

# ====================== LOAD CLASS NAMES ======================
def load_class_names():
    try:
        with open(CLASS_NAMES_PATH, 'r', encoding='utf-8') as f:
            class_names = [line.strip() for line in f.readlines() if line.strip()]
        print(f"Loaded {len(class_names)} classes from file")
        return class_names
    except Exception as e:
        print(f"Error loading class names: {e}")
        exit(1)

class_names = load_class_names()

# ====================== LOAD MODEL ======================
model = KannadaSignModel(num_classes=len(class_names)).to(device)

try:
    state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# ====================== DATA LOADING ======================
def load_npy_file(file_path):
    """Load and reshape .npy file to match model input"""
    try:
        data = np.load(file_path)
        print(f"Original shape: {data.shape}")  # Debug
        
        if data.ndim == 4 and data.shape[1:] == (2, 21, 3):
            # Reshape from (seq_len, 2, 21, 3) → (seq_len, 126)
            seq_len = data.shape[0]
            data = data.reshape(seq_len, -1)  # [seq_len, 126]
            data = data.T  # [126, seq_len]
        elif data.ndim == 2 and data.shape[1] == 126:
            # Shape is (seq_len, 126)
            data = data.T  # [126, seq_len]
        else:
            raise ValueError(f"Unexpected shape: {data.shape}")
        
        # Add batch dimension [1, 126, seq_len]
        data = data[np.newaxis, :, :]
        
        print(f"Reshaped to: {data.shape}")  # Debug
        return torch.FloatTensor(data).to(device)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

# ====================== VALIDATION ======================
def validate_model(data_dir, num_samples_per_class=NUM_TEST_SAMPLES):
    all_preds = []
    all_labels = []
    
    print(f"\nTesting on {data_dir}...")
    
    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            print(f"Warning: Missing class directory {class_dir}")
            continue
            
        npy_files = [f for f in os.listdir(class_dir) if f.endswith('.npy')]
        if not npy_files:
            print(f"Warning: No .npy files found in {class_dir}")
            continue
            
        test_files = random.sample(npy_files, min(num_samples_per_class, len(npy_files)))
        print(f"Class '{class_name}': Testing {len(test_files)} samples")
        
        for file_name in test_files:
            file_path = os.path.join(class_dir, file_name)
            data = load_npy_file(file_path)
            
            if data is None:
                continue
                
            with torch.no_grad():
                outputs = model(data)
                _, pred = torch.max(outputs, 1)
                confidence = torch.nn.functional.softmax(outputs, dim=1)[0][pred].item()
                
                all_preds.append(pred.item())
                all_labels.append(class_idx)
                
                result = "✓" if pred.item() == class_idx else "✗"
                print(f"  {result} {file_name}: Predicted {class_names[pred.item()]} ({confidence:.2f})")
    
    return all_labels, all_preds

# ====================== MAIN FUNCTION ======================
if __name__ == "__main__":
    val_true, val_pred = validate_model(VALIDATION_DIR)
    
    if not val_true:
        print("\nTrying training set as fallback...")
        val_true, val_pred = validate_model(TRAIN_DIR)
    
    if not val_true:
        print("\nError: Could not process any samples")
        exit(1)
    
    print("\n=== Classification Report ===")
    print(classification_report(val_true, val_pred, target_names=class_names))
    
    cm = confusion_matrix(val_true, val_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    accuracy = np.mean(np.array(val_true) == np.array(val_pred))
    print(f"\nOverall Accuracy: {accuracy:.2%}")