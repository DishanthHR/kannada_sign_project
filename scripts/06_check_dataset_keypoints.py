import os
import numpy as np
from tqdm import tqdm  # For progress bars (install with: pip install tqdm)

# Configuration
BASE_DIR = r"C:\Users\savem\OneDrive\Desktop\kannada_sign_project\dataset\processed_keypoints"
SPLITS = ["train", "val", "test"]  # All dataset splits to check
SHOW_INDIVIDUAL_FILES = False  # Set to True to see every file's stats

def check_dataset_stats(dataset_dir):
    min_vals, max_vals = [], []
    file_count = 0
    
    print(f"\n\n=== Processing: {dataset_dir} ===")
    
    # Get all class directories
    classes = [d for d in os.listdir(dataset_dir) 
              if os.path.isdir(os.path.join(dataset_dir, d))]
    
    for class_name in tqdm(classes, desc="Processing classes"):
        class_dir = os.path.join(dataset_dir, class_name)
        npy_files = [f for f in os.listdir(class_dir) if f.endswith('.npy')]
        
        for file in npy_files:
            path = os.path.join(class_dir, file)
            try:
                kps = np.load(path)
                current_min = kps.min()
                current_max = kps.max()
                
                min_vals.append(current_min)
                max_vals.append(current_max)
                file_count += 1
                
                if SHOW_INDIVIDUAL_FILES:
                    print(f"  {class_name:<15} {file:<25} Min: {current_min:.4f} Max: {current_max:.4f}")
                    
            except Exception as e:
                print(f"\nError in {path}: {str(e)}")
                continue

    # Summary Statistics
    if file_count > 0:
        print("\n=== Dataset Summary ===")
        print(f"Total files processed: {file_count}")
        print(f"Global Minimum: {np.min(min_vals):.6f}")
        print(f"Global Maximum: {np.max(max_vals):.6f}")
        print(f"Average Minimum: {np.mean(min_vals):.6f}")
        print(f"Average Maximum: {np.mean(max_vals):.6f}")
    else:
        print("\nNo valid .npy files found in this split!")

if __name__ == "__main__":
    for split in SPLITS:
        split_dir = os.path.join(BASE_DIR, split)
        if os.path.exists(split_dir):
            check_dataset_stats(split_dir)
        else:
            print(f"\n⚠️ Directory not found: {split_dir}")