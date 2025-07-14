import os
import shutil
from sklearn.model_selection import train_test_split

# Paths - Update this to point to your augmented data
raw_dir = r"C:\Users\savem\OneDrive\Desktop\kannada_sign_project\dataset\vedio_augmented_data"
train_dir = r"C:\Users\savem\OneDrive\Desktop\kannada_sign_project\dataset\processed_data\train"
val_dir = r"C:\Users\savem\OneDrive\Desktop\kannada_sign_project\dataset\processed_data\val"

# Create train and val directories
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Go through each class
for class_name in os.listdir(raw_dir):
    class_path = os.path.join(raw_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    videos = [v for v in os.listdir(class_path) if v.lower().endswith('.mp4')]
    print(f"[{class_name}] Found {len(videos)} videos")

    train_videos, val_videos = train_test_split(videos, test_size=0.2, random_state=42)

    # Create class subfolders
    train_class_dir = os.path.join(train_dir, class_name)
    val_class_dir = os.path.join(val_dir, class_name)
    os.makedirs(train_class_dir, exist_ok=True)
    os.makedirs(val_class_dir, exist_ok=True)

    # Copy training videos
    for video in train_videos:
        src = os.path.join(class_path, video)
        dst = os.path.join(train_class_dir, video)
        shutil.copy2(src, dst)

    # Copy validation videos
    for video in val_videos:
        src = os.path.join(class_path, video)
        dst = os.path.join(val_class_dir, video)
        shutil.copy2(src, dst)

print("\n✅ Dataset split into 'train/' and 'val/' successfully!")
