import os
import glob
import sys

def inspect_dataset():
    """Inspect the dataset structure to help diagnose issues"""
    # Define paths
    base_dir = os.getcwd()
    dataset_dir = os.path.join(base_dir, 'dataset')
    processed_dir = os.path.join(dataset_dir, 'processed_data')
    
    print(f"Current working directory: {base_dir}")
    print(f"Looking for dataset in: {dataset_dir}")
    
    # Check if dataset directory exists
    if not os.path.exists(dataset_dir):
        print(f"Dataset directory not found: {dataset_dir}")
        return
    
    print(f"Dataset directory found: {dataset_dir}")
    
    # List contents of dataset directory
    print("\nContents of dataset directory:")
    for item in os.listdir(dataset_dir):
        item_path = os.path.join(dataset_dir, item)
        if os.path.isdir(item_path):
            print(f"  Directory: {item}")
        else:
            print(f"  File: {item}")
    
    # Check if processed_data directory exists
    if not os.path.exists(processed_dir):
        print(f"\nProcessed data directory not found: {processed_dir}")
        return
    
    print(f"\nProcessed data directory found: {processed_dir}")
    
    # List contents of processed_data directory
    print("\nContents of processed_data directory:")
    for item in os.listdir(processed_dir):
        item_path = os.path.join(processed_dir, item)
        if os.path.isdir(item_path):
            print(f"  Directory: {item}")
        else:
            print(f"  File: {item}")
    
    # Check train and val directories
    train_dir = os.path.join(processed_dir, 'train')
    val_dir = os.path.join(processed_dir, 'val')
    
    if os.path.exists(train_dir):
        print(f"\nTrain directory found: {train_dir}")
        print("\nClass directories in train:")
        try:
            for item in os.listdir(train_dir):
                item_path = os.path.join(train_dir, item)
                if os.path.isdir(item_path):
                    num_files = len(glob.glob(os.path.join(item_path, "*.*")))
                    print(f"  Class: {item} - {num_files} files")
        except Exception as e:
            print(f"Error listing train directory: {e}")
    else:
        print(f"\nTrain directory not found: {train_dir}")
    
    if os.path.exists(val_dir):
        print(f"\nVal directory found: {val_dir}")
        print("\nClass directories in val:")
        try:
            for item in os.listdir(val_dir):
                item_path = os.path.join(val_dir, item)
                if os.path.isdir(item_path):
                    num_files = len(glob.glob(os.path.join(item_path, "*.*")))
                    print(f"  Class: {item} - {num_files} files")
        except Exception as e:
            print(f"Error listing val directory: {e}")
    else:
        print(f"\nVal directory not found: {val_dir}")

if __name__ == "__main__":
    inspect_dataset()
