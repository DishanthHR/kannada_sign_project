import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Paths
data_dir = r'C:\Users\savem\OneDrive\Desktop\kannada_sign_project\dataset\processed_preprocessed'
model_dir = r'C:\Users\savem\OneDrive\Desktop\kannada_sign_project\trained_model'
os.makedirs(model_dir, exist_ok=True)

# Load data
X = np.load(os.path.join(data_dir, 'X.npy'))
y = np.load(os.path.join(data_dir, 'y.npy'))
classes = np.load(os.path.join(data_dir, 'classes.npy'), allow_pickle=True)

logging.info(f"Loaded X shape: {X.shape}, y shape: {y.shape}")
logging.info(f"Number of classes: {len(classes)}")

# Scale keypoints (StandardScaler is good practice)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)

logging.info(f"Train shape: {X_train.shape}, Validation shape: {X_val.shape}")

# ✅ Build model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(len(classes), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ✅ Callbacks
checkpoint = ModelCheckpoint(os.path.join(model_dir, 'best_model.h5'), 
                             save_best_only=True, monitor='val_accuracy', mode='max')
early_stop = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)

# ✅ Train model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[checkpoint, early_stop]
)

# ✅ Save scaler for real-time use
import joblib
joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))

# ✅ Save final model in SavedModel format too (best for TensorFlow and TFLite)
model.save(os.path.join(model_dir, 'final_model'))

# ✅ Save classes
np.save(os.path.join(model_dir, 'classes.npy'), classes)

logging.info("✅ Training completed! Model and scaler saved.")
