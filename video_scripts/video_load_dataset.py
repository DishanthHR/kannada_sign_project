import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle

# Define paths
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
keypoints_dir = os.path.join(base_dir, 'dataset', 'vedio_processed_keypoints')
processed_data_dir = os.path.join(base_dir, 'dataset', 'video_processed_data')
os.makedirs(processed_data_dir, exist_ok=True)

# Define the Kannada words (classes)
kannada_words = [
    "ಆಡು", "ಇದೀಯ", "ಇಲ್ಲ", "ಇಲ್ಲಿ", "ಇವತ್ತು", "ಎಲ್ಲಿ", "ಏಕೆ", "ಏನು", 
    "ಕುಳಿತುಕೊ", "ಕೇಳು", "ಗಲಾಟೆ", "ಜೊತೆ", "ತಿಂದೆ", "ತೆಗೆದುಕೋ", "ನಾನು", 
    "ನಿಧಾನವಾಗಿ", "ನಿನ್ನ", "ನೀನು", "ಪುಸ್ತಕ", "ಬಂದರು", "ಮಾಡಬೇಡಿ", "ಮಾತು", 
    "ಯಾರು", "ವಾಸವಾಗಿ", "ಸುಮ್ಮನೆ", "ಹೆಚ್ಚು", "ಹೋಗಿದ್ದೆ"
]

def load_and_preprocess_data(max_frames=30, normalize=True):
    """
    Load and preprocess keypoints data for all words.
    
    Args:
        max_frames: Maximum number of frames to use from each video
        normalize: Whether to normalize the keypoints
        
    Returns:
        X: Preprocessed keypoints data
        y: One-hot encoded labels
    """
    X = []
    y = []
    
    print("Loading and preprocessing data...")
    
    # Process each word
    for word_idx, word in enumerate(kannada_words):
        word_dir = os.path.join(keypoints_dir, word)
        
        if not os.path.exists(word_dir):
            print(f"Warning: Directory for word '{word}' not found at {word_dir}")
            continue
        
        # Process each keypoints file for this word
        keypoints_files = [f for f in os.listdir(word_dir) if f.endswith('.npy')]
        
        for keypoints_file in keypoints_files:
            keypoints_path = os.path.join(word_dir, keypoints_file)
            
            try:
                # Load keypoints
                keypoints = np.load(keypoints_path)
                
                # Pad or truncate to max_frames
                if keypoints.shape[0] > max_frames:
                    # Truncate to max_frames
                    keypoints = keypoints[:max_frames]
                elif keypoints.shape[0] < max_frames:
                    # Pad with zeros
                    padding = np.zeros((max_frames - keypoints.shape[0], 2, 21, 3))
                    keypoints = np.vstack((keypoints, padding))
                
                # Normalize if requested
                if normalize:
                    # Normalize x, y coordinates to [0, 1] range
                    # z coordinates are already normalized in MediaPipe
                    for i in range(keypoints.shape[0]):
                        for h in range(keypoints.shape[1]):
                            if np.sum(keypoints[i, h]) > 0:  # If hand is detected
                                keypoints[i, h, :, 0] = (keypoints[i, h, :, 0] - np.min(keypoints[i, h, :, 0])) / (np.max(keypoints[i, h, :, 0]) - np.min(keypoints[i, h, :, 0]) + 1e-10)
                                keypoints[i, h, :, 1] = (keypoints[i, h, :, 1] - np.min(keypoints[i, h, :, 1])) / (np.max(keypoints[i, h, :, 1]) - np.min(keypoints[i, h, :, 1]) + 1e-10)
                
                # Add to dataset
                X.append(keypoints)
                y.append(word_idx)
                
            except Exception as e:
                print(f"Error loading {keypoints_path}: {str(e)}")
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    # Convert labels to one-hot encoding
    y_one_hot = to_categorical(y, num_classes=len(kannada_words))
    
    print(f"Loaded dataset with {len(X)} samples, X shape: {X.shape}, y shape: {y_one_hot.shape}")
    
    return X, y_one_hot

def split_dataset(X, y, test_size=0.2, val_size=0.1, random_state=42):
    """
    Split the dataset into training, validation, and test sets.
    
    Args:
        X: Input data
        y: Labels
        test_size: Proportion of data to use for testing
        val_size: Proportion of training data to use for validation
        random_state: Random seed for reproducibility
        
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    # First split: training + validation vs test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if len(y.shape) == 1 else np.argmax(y, axis=1)
    )
    
    # Second split: training vs validation
    # Calculate validation size relative to training + validation size
    relative_val_size = val_size / (1 - test_size)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=relative_val_size, random_state=random_state, 
        stratify=y_train_val if len(y_train_val.shape) == 1 else np.argmax(y_train_val, axis=1)
    )
    
    print(f"Dataset split: Train: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def save_processed_data(X_train, X_val, X_test, y_train, y_val, y_test):
    """Save the processed data to disk"""
    data = {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test
    }
    
    output_path = os.path.join(processed_data_dir, 'processed_data.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"Saved processed data to {output_path}")

def load_processed_data():
    """Load the processed data from disk"""
    input_path = os.path.join(processed_data_dir, 'processed_data.pkl')
    
    if not os.path.exists(input_path):
        print(f"Error: Processed data file not found at {input_path}")
        return None
    
    with open(input_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Loaded processed data from {input_path}")
    
    return data

def visualize_data_distribution(y):
    """Visualize the distribution of classes in the dataset"""
    if len(y.shape) > 1:  # If one-hot encoded
        y_classes = np.argmax(y, axis=1)
    else:
        y_classes = y
    
    # Count samples per class
    class_counts = np.bincount(y_classes)
    
    # Plot
    plt.figure(figsize=(15, 6))
    plt.bar(range(len(class_counts)), class_counts)
    plt.xticks(range(len(class_counts)), kannada_words, rotation=90)
    plt.xlabel('Word')
    plt.ylabel('Number of samples')
    plt.title('Distribution of samples across words')
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(processed_data_dir, 'data_distribution.png')
    plt.savefig(output_path)
    plt.close()
    
    print(f"Saved data distribution plot to {output_path}")

if __name__ == "__main__":
    # Load and preprocess data
    X, y = load_and_preprocess_data(max_frames=30, normalize=True)
    
    # Split dataset
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y)
    
    # Save processed data
    save_processed_data(X_train, X_val, X_test, y_train, y_val, y_test)
    
    # Visualize data distribution
    visualize_data_distribution(y)
