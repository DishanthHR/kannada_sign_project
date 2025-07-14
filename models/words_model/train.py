import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from tqdm import tqdm
import time
from model import KannadaSignModel  # Make sure this import path is correct

# Configuration
DATA_ROOT = r"C:\Users\savem\OneDrive\Desktop\kannada_sign_project\dataset\new_processed_video"
SEQ_LENGTH = 45  # Frames (pad/trim to this length)
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 0.001
PATIENCE = 5  # Early stopping patience
CHECKPOINT_DIR = r"C:\Users\savem\OneDrive\Desktop\kannada_sign_project\model\checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)# Folder to save models

# Dataset Class
class SignDataset(Dataset):
    def __init__(self, split="train"):
        self.data = []
        self.labels = []
        split_dir = os.path.join(DATA_ROOT, split)
        
        self.classes = sorted(os.listdir(split_dir))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        for label in self.classes:
            label_dir = os.path.join(split_dir, label)
            for file in os.listdir(label_dir):
                if file.endswith(".npy"):
                    arr = np.load(os.path.join(label_dir, file))
                    # Ensure shape is [seq_len, 2, 21, 3]
                    if arr.ndim == 4:
                        self.data.append(arr)
                    elif arr.ndim == 3:
                        arr = np.expand_dims(arr, axis=1)  # Add hand dimension
                        self.data.append(arr)
                    else:
                        print(f"Skipping {file}: Unexpected shape {arr.shape}")
                        continue
                    self.labels.append(self.class_to_idx[label])

        print(f"\nSample shapes in {split} dataset:")
        for i in range(min(3, len(self.data))):
            print(f"Sample {i}: shape={self.data[i].shape}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        seq_len = x.shape[0]
        
        # Pad/trim sequence
        if seq_len < SEQ_LENGTH:
            pad_len = SEQ_LENGTH - seq_len
            x = np.pad(x, ((0, pad_len), (0, 0), (0, 0), (0, 0)), mode='edge')
        else:
            x = x[:SEQ_LENGTH]
            
        return torch.FloatTensor(x), torch.tensor(self.labels[idx])

# Training Function
def train():
    # Initialize datasets
    train_set = SignDataset("train")
    val_set = SignDataset("val")
    
    # Data loaders
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, num_workers=4)

    # Model verification
    print("\nModel architecture verification:")
    model = KannadaSignModel(num_classes=len(train_set.classes))
    sample_input = torch.randn(2, SEQ_LENGTH, 2, 21, 3)  # Test batch
    output = model(sample_input)
    print(f"Input shape: {sample_input.shape}")
    print(f"Output shape: {output.shape} (should be [2, {len(train_set.classes)}])")

    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Batch verification
    sample_batch = next(iter(train_loader))
    print("\nBatch shape verification:")
    print(f"Input shape: {sample_batch[0].shape} (should be [B, {SEQ_LENGTH}, 2, 21, 3])")
    print(f"Label shape: {sample_batch[1].shape} (should be [B])")

    # Training loop
    best_val_acc = 0.0
    patience_counter = 0
    
    print(f"\nTraining on {len(train_set)} samples, validating on {len(val_set)} samples")
    print(f"Classes: {train_set.classes}\n")

    for epoch in range(1, EPOCHS + 1):
        start_time = time.time()
        model.train()
        train_loss = 0.0
        
        # Training phase
        for batch_idx, (x, y) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")):
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

            # Print first batch details
            if epoch == 1 and batch_idx == 0:
                print(f"\nFirst batch details:")
                print(f"Input shape: {x.shape}")
                print(f"Input min/max: {x.min():.2f}/{x.max():.2f}")
                print(f"Label distribution: {torch.bincount(y)}")

        # Validation phase
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in val_loader:
                outputs = model(x)
                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        
        val_acc = 100 * correct / total
        epoch_time = time.time() - start_time

        # Print metrics
        print(f"\nEpoch {epoch:02d}: "
              f"Train Loss = {train_loss/len(train_loader):.4f}, "
              f"Val Acc = {val_acc:.2f}%, "
              f"Time = {epoch_time:.1f}s")

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "best_model.pth"))
            patience_counter = 0
            print(f"🔥 New best model saved! (Accuracy: {val_acc:.2f}%)")
        else:
            patience_counter += 1
            print(f"⏳ No improvement ({patience_counter}/{PATIENCE})")
            if patience_counter >= PATIENCE:
                print(f"🛑 Early stopping triggered at epoch {epoch}!")
                break

    print(f"\nTraining complete. Best validation accuracy: {best_val_acc:.2f}%")

if __name__ == "__main__":
    train()