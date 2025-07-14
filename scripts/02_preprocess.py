import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load keypoints dataframe
keypoints_file = r'C:\Users\savem\OneDrive\Desktop\kannada_sign_project\dataset\processed_keypoints\keypoints.pkl'
df = pd.read_pickle(keypoints_file)
logging.info(f"Loaded dataframe shape: {df.shape}")

# ✅ Encode class labels to numbers
le = LabelEncoder()
df['class_encoded'] = le.fit_transform(df['class'])
logging.info(f"Classes encoded: {list(le.classes_)}")

# ✅ Extract X (keypoints) and y (encoded labels)
X = np.array(df['keypoints'].tolist(), dtype=np.float32)
y = np.array(df['class_encoded'], dtype=np.int32)

logging.info(f"Feature shape (X): {X.shape}, Label shape (y): {y.shape}")

# ✅ Save X and y for model training
output_dir = r'C:\Users\savem\OneDrive\Desktop\kannada_sign_project\dataset\processed_preprocessed'
os.makedirs(output_dir, exist_ok=True)

np.save(os.path.join(output_dir, 'X.npy'), X)
np.save(os.path.join(output_dir, 'y.npy'), y)

# Save label classes for later decoding
np.save(os.path.join(output_dir, 'classes.npy'), le.classes_)

logging.info("✅ Saved X, y and classes for model training!")
