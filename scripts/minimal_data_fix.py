import numpy as np
import os

def fix_data():
    print("Starting minimal data fix script...")
    
    # Define paths
    train_data_path = r"C:\Users\savem\OneDrive\Desktop\kannada_sign_project\dataset\processed_vedio_data\train\train_data.npy"
    val_data_path = r"C:\Users\savem\OneDrive\Desktop\kannada_sign_project\dataset\processed_vedio_data\val\val_data.npy"
    
    train_fixed_path = r"C:\Users\savem\OneDrive\Desktop\kannada_sign_project\dataset\processed_vedio_data\train\train_data_fixed.npy"
    val_fixed_path = r"C:\Users\savem\OneDrive\Desktop\kannada_sign_project\dataset\processed_vedio_data\val\val_data_fixed.npy"
    
    # Target dimensions
    target_seq_length = 73
    target_feature_dim = 63
    
    try:
        # Load training data
        print("Loading training data...")
        train_data = np.load(train_data_path, allow_pickle=True)
        
        # Check if it's a dictionary-like object
        if hasattr(train_data, 'item'):
            train_data = train_data.item()
        
        # Extract sequences and labels
        train_sequences = train_data['data']
        train_labels = train_data['labels']
        
        print(f"Loaded {len(train_sequences)} training sequences")
        print(f"Loaded {len(train_labels)} training labels")
        
        # Load validation data
        print("Loading validation data...")
        val_data = np.load(val_data_path, allow_pickle=True)
        
        # Check if it's a dictionary-like object
        if hasattr(val_data, 'item'):
            val_data = val_data.item()
        
        # Extract sequences and labels
        val_sequences = val_data['data']
        val_labels = val_data['labels']
        
        print(f"Loaded {len(val_sequences)} validation sequences")
        print(f"Loaded {len(val_labels)} validation labels")
        
        # Create arrays for normalized data
        print(f"Creating normalized arrays with shape ({len(train_sequences)}, {target_seq_length}, {target_feature_dim})...")
        normalized_train = np.zeros((len(train_sequences), target_seq_length, target_feature_dim))
        normalized_val = np.zeros((len(val_sequences), target_seq_length, target_feature_dim))
        
        # Process training sequences
        print("Processing training sequences...")
        for i, seq in enumerate(train_sequences):
            try:
                # Convert to numpy array if needed
                if not isinstance(seq, np.ndarray):
                    seq = np.array(seq)
                
                # Get dimensions to copy
                seq_length = min(seq.shape[0], target_seq_length)
                feature_dim = min(seq.shape[1], target_feature_dim)
                
                # Copy data
                normalized_train[i, :seq_length, :feature_dim] = seq[:seq_length, :feature_dim]
                
                if i % 100 == 0:
                    print(f"Processed {i}/{len(train_sequences)} training sequences")
            except Exception as e:
                print(f"Error processing training sequence {i}: {e}")
        
        # Process validation sequences
        print("Processing validation sequences...")
        for i, seq in enumerate(val_sequences):
            try:
                # Convert to numpy array if needed
                if not isinstance(seq, np.ndarray):
                    seq = np.array(seq)
                
                # Get dimensions to copy
                seq_length = min(seq.shape[0], target_seq_length)
                feature_dim = min(seq.shape[1], target_feature_dim)
                
                # Copy data
                normalized_val[i, :seq_length, :feature_dim] = seq[:seq_length, :feature_dim]
                
                if i % 100 == 0:
                    print(f"Processed {i}/{len(val_sequences)} validation sequences")
            except Exception as e:
                print(f"Error processing validation sequence {i}: {e}")
        
        # Create new data dictionaries
        print("Creating fixed data dictionaries...")
        train_data_fixed = {}
        train_data_fixed['data'] = normalized_train
        train_data_fixed['labels'] = np.array(train_labels)
        
        val_data_fixed = {}
        val_data_fixed['data'] = normalized_val
        val_data_fixed['labels'] = np.array(val_labels)
        
        # Save the fixed data
        print(f"Saving fixed training data to {train_fixed_path}")
        np.save(train_fixed_path, train_data_fixed)
        
        print(f"Saving fixed validation data to {val_fixed_path}")
        np.save(val_fixed_path, val_data_fixed)
        
        print("Data fixing complete!")
        print(f"Fixed training data shape: {normalized_train.shape}")
        print(f"Fixed validation data shape: {normalized_val.shape}")
        
    except Exception as e:
        print(f"Error in fix_data: {e}")

if __name__ == "__main__":
    fix_data()
