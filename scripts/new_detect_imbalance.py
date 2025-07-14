import numpy as np
import os

base_folder = r"C:\Users\savem\OneDrive\Desktop\kannada_sign_project\dataset\new_processed_video"
for split in ["train", "val"]:
    print(f"\n📊 {split.upper()} Stats:")
    for label in os.listdir(os.path.join(base_folder, split)):
        n_samples = len(os.listdir(os.path.join(base_folder, split, label)))
        seq_lengths = [np.load(os.path.join(base_folder, split, label, f)).shape[0] for f in os.listdir(os.path.join(base_folder, split, label))[:5]]
        print(f"  {label}: {n_samples} samples | Seq lengths: {min(seq_lengths)}-{max(seq_lengths)} frames")