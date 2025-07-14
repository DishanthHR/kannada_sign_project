import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Configuration
SOURCE_DIR = r"C:\Users\savem\OneDrive\Desktop\kannada_sign_project\dataset\vedios_keypoints"
DEST_DIR = r"C:\Users\savem\OneDrive\Desktop\kannada_sign_project\dataset\processed_vedio_data"
MAX_SEQ_LENGTH = 73  # Same as your model expects
TEST_SIZE = 0.2      # 20% validation
RANDOM_STATE = 42    # For reproducibility

def load_sequences_and_labels():
    sequences = []
    labels = []
    
    print("🔍 Scanning dataset folders...")
    
    # Walk through each word folder
    for word in os.listdir(SOURCE_DIR):
        word_dir = os.path.join(SOURCE_DIR, word)
        if not os.path.isdir(word_dir):
            continue
            
        print(f"📁 Processing {word}...")
        
        # Load all .npy files for this word
        for npy_file in os.listdir(word_dir):
            if npy_file.endswith('.npy'):
                file_path = os.path.join(word_dir, npy_file)
                seq = np.load(file_path)
                
                # Skip empty sequences
                if len(seq) == 0:
                    print(f"⚠️ Empty sequence in {file_path} - skipping")
                    continue
                    
                sequences.append(seq)
                labels.append(word)
    
    return sequences, labels

def process_and_save_data():
    # 1. Load raw data
    sequences, labels = load_sequences_and_labels()
    
    # 2. Pad sequences to uniform length
    X = pad_sequences(sequences,
                     maxlen=MAX_SEQ_LENGTH,
                     padding='post',
                     truncating='post',
                     dtype='float32')
    
    # 3. Split into train/val (stratified)
    X_train, X_val, y_train, y_val = train_test_split(
        X, labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels
    )
    
    # 4. Create destination folders
    os.makedirs(os.path.join(DEST_DIR, "train"), exist_ok=True)
    os.makedirs(os.path.join(DEST_DIR, "val"), exist_ok=True)
    
    # 5. Save the splits
    train_path = os.path.join(DEST_DIR, "train", "train_data.npy")
    val_path = os.path.join(DEST_DIR, "val", "val_data.npy")
    
    np.save(train_path, {"data": X_train, "labels": y_train})
    np.save(val_path, {"data": X_val, "labels": y_val})
    
    # 6. Print summary
    print("\n✅ Data successfully processed and split:")
    print(f" - Total samples: {len(X)}")
    print(f" - Training samples: {len(X_train)}")
    print(f" - Validation samples: {len(X_val)}")
    print(f" - Sequence shape: {X_train[0].shape}")
    print(f"\nSaved to:")
    print(f" - Training data: {train_path}")
    print(f" - Validation data: {val_path}")

if __name__ == "__main__":
    process_and_save_data()