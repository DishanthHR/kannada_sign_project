import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os
import matplotlib.pyplot as plt
from sklearn.utils import class_weight

# Kannada words mapping
kannada_words = [
    'ಆಡು', 'ಇದೀಯ', 'ಇಲ್ಲ', 'ಇಲ್ಲಿ', 'ಇವತ್ತು', 
    'ಎಲ್ಲಿ', 'ಎಲ್ಲಿಗೆ', 'ಏಕೆ', 'ಏನು', 'ಕುಳಿತುಕೊ', 
    'ಕೇಳು', 'ಗಲಾಟೆ', 'ಜೊತೆ', 'ತಿಂದೆ', 'ತೆಗೆದುಕೋ', 
    'ನಾನು', 'ನಿಧವಾಗಿ', 'ನಿನ್ನ', 'ನೀನು', 'ಪುಸ್ತಕ', 
    'ಬಂದರು', 'ಮಾಡಬೇಡಿ', 'ಮಾತು', 'ಯಾರು', 'ವಾಸವಾಗಿ', 
    'ಸುಮ್ಮನೆ', 'ಹೆಚ್ಚು', 'ಹೋಗಿದ್ದೆ'
]

# Function to load your dataset
def load_dataset():
    """
    Load your dataset from the specified paths.
    """
    print("Loading training and validation data...")
    
    # Define paths to your data files
    train_data_path = r"C:\Users\savem\OneDrive\Desktop\kannada_sign_project\dataset\processed_vedio_data\train\train_data.npy"
    val_data_path = r"C:\Users\savem\OneDrive\Desktop\kannada_sign_project\dataset\processed_vedio_data\val\val_data.npy"
    
    try:
        # Load the data
        train_data = np.load(train_data_path, allow_pickle=True).item()
        val_data = np.load(val_data_path, allow_pickle=True).item()
        
        # Extract features and labels
        X_train = []
        y_train = []
        X_val = []
        y_val = []
        
        # Process training data
        for action_idx, sequences in train_data.items():
            for sequence in sequences:
                X_train.append(sequence)
                y_train.append(action_idx)
        
        # Process validation data
        for action_idx, sequences in val_data.items():
            for sequence in sequences:
                X_val.append(sequence)
                y_val.append(action_idx)
        
        # Convert to numpy arrays
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_val = np.array(X_val)
        y_val = np.array(y_val)
        
        print(f"Loaded training data with shape X: {X_train.shape}, y: {y_train.shape}")
        print(f"Loaded validation data with shape X: {X_val.shape}, y: {y_val.shape}")
        
        return X_train, y_train, X_val, y_val
    
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        print("Please ensure your data files exist at the specified paths.")
        return None, None, None, None
    except Exception as e:
        print(f"Unexpected error loading data: {e}")
        return None, None, None, None

# Function to build the model
def build_model(input_shape, num_classes):
    model = Sequential()
    
    # LSTM layers
    model.add(LSTM(128, return_sequences=True, activation='relu', input_shape=input_shape))
    model.add(Dropout(0.3))
    
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dropout(0.3))
    
    # Dense layers
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    
    # Output layer
    model.add(Dense(num_classes, activation='softmax'))
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Main function to retrain the model
def retrain_model():
    # Set path for saving model
    model_save_path = 'models/words/kannada_sign_model_retrained.h5'
    
    # Load data
    X_train, y_train, X_val, y_val = load_dataset()
    if X_train is None or y_train is None:
        print("Failed to load data. Exiting.")
        return
    
    # Calculate class weights to handle imbalance
    class_weights = class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
    
    # Print class distribution
    print("\nClass distribution in training data:")
    unique, counts = np.unique(y_train, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"Class {u} ({kannada_words[int(u)]}): {c} samples, weight: {class_weight_dict.get(int(u), 1.0):.4f}")
    
    # Build model
    input_shape = (X_train.shape[1], X_train.shape[2])
    num_classes = len(kannada_words)
    model = build_model(input_shape, num_classes)
    
    # Set up callbacks
    checkpoint = ModelCheckpoint(
        model_save_path,
        monitor='val_accuracy',
        verbose=1,
        save_best_only=True,
        mode='max'
    )
    
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=15,
        restore_best_weights=True
    )
    
    # Train the model
    print("\nTraining model...")
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[checkpoint, early_stopping],
        class_weight=class_weight_dict
    )
    
    # Evaluate the model
    print("\nEvaluating model...")
    test_loss, test_accuracy = model.evaluate(X_val, y_val)
    print(f"Validation accuracy: {test_accuracy*100:.2f}%")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    print("Training history plot saved to 'training_history.png'")
    
    # Save the final model
    model.save(model_save_path.replace('.h5', '_final.h5'))
    print(f"Final model saved to {model_save_path.replace('.h5', '_final.h5')}")

if __name__ == "__main__":
    retrain_model()
