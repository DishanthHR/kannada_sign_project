import tensorflow as tf
import numpy as np
import os
import sys
import cv2
from sklearn.metrics import confusion_matrix, classification_report
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add the project root to the path so we can import from scripts
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.plot_utils import plot_confusion_matrix
from scripts.font_config import get_kannada_class_names, get_english_class_names

def extract_frames_from_video(video_path, num_frames=15, target_size=(64, 64)):
    """Extract frames from a video file"""
    frames = []
    cap = cv2.VideoCapture(video_path)
    
    # Get total frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames <= 0:
        print(f"Warning: Could not read frames from {video_path}")
        return None
    
    # Calculate frame indices to extract (evenly distributed)
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert to RGB and resize
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, target_size)
        
        # Normalize
        frame = frame / 255.0
        
        frames.append(frame)
    
    cap.release()
    
    # Check if we got enough frames
    if len(frames) < num_frames:
        print(f"Warning: Only extracted {len(frames)} frames from {video_path}")
        # Pad with zeros if needed
        while len(frames) < num_frames:
            frames.append(np.zeros((target_size[0], target_size[1], 3), dtype=np.float32))
    
    return np.array(frames)

def load_video_dataset(base_dir, class_names, batch_size=8):
    """Load video dataset from directory"""
    videos = []
    labels = []
    
    print(f"Loading videos from {base_dir}")
    
    # Create class mapping
    class_to_index = {name: i for i, name in enumerate(class_names)}
    
    # For each class directory
    for class_dir in tqdm(os.listdir(base_dir), desc="Loading classes"):
        class_path = os.path.join(base_dir, class_dir)
        
        if not os.path.isdir(class_path):
            continue
            
        # Get class index
        if class_dir in class_to_index:
            class_idx = class_to_index[class_dir]
        else:
            print(f"Warning: Unknown class directory {class_dir}")
            continue
            
        # Get all video files
        video_files = []
        for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv']:
            video_files.extend(glob.glob(os.path.join(class_path, ext)))
            
        print(f"Found {len(video_files)} videos for class {class_dir}")
        
        # Process each video
        for video_file in tqdm(video_files, desc=f"Processing {class_dir}", leave=False):
            try:
                # Extract frames
                frames = extract_frames_from_video(video_file)
                
                if frames is not None:
                    videos.append(frames)
                    labels.append(class_idx)
            except Exception as e:
                print(f"Error processing video {video_file}: {e}")
    
    if not videos:
        print("No videos were successfully loaded")
        return None
        
    # Convert to numpy arrays
    videos = np.array(videos)
    labels = np.array(labels)
    
    print(f"Loaded {len(videos)} videos with shape {videos.shape}")
    
    # Create TensorFlow dataset
    dataset = tf.data.Dataset.from_tensor_slices((videos, labels))
    dataset = dataset.batch(batch_size)
    
    return dataset

def evaluate_model(model, dataset):
    """Evaluate model on a dataset and return metrics"""
    if dataset is None:
        print("Dataset is None, cannot evaluate model")
        return np.zeros((28, 28)), 0.0, [], []
    
    # Get predictions
    all_predictions = []
    all_labels = []
    
    for videos, labels in tqdm(dataset, desc="Evaluating"):
        predictions = model.predict(videos, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)
        
        all_predictions.extend(predicted_classes)
        all_labels.extend(labels.numpy())
    
    if not all_labels:
        print("No labels found in dataset")
        return np.zeros((28, 28)), 0.0, [], []
    
    # Calculate confusion matrix
    cm = confusion_matrix(all_labels, all_predictions, 
                          labels=range(len(get_kannada_class_names())))
    
    # Calculate accuracy
    accuracy = sum(1 for x, y in zip(all_predictions, all_labels) if x == y) / len(all_labels)
    
    return cm, accuracy, all_labels, all_predictions

def main():
    # Create output directories
    os.makedirs('evaluation_results', exist_ok=True)
    
    # Load the model
    model_path = 'models/words/best_model.h5'  # Updated to match your path
    print(f"Loading model from {model_path}...")
    
    try:
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully")
        print(f"Model input shape: {model.input_shape}")
        print(f"Model output shape: {model.output_shape}")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please check if the model file exists and is valid.")
        return
    
    # Get class names
    class_names = get_kannada_class_names()
    english_names = get_english_class_names()
    
    # Define dataset paths
    train_dir = os.path.join(os.getcwd(), 'dataset', 'processed_data', 'train')
    val_dir = os.path.join(os.getcwd(), 'dataset', 'processed_data', 'val')
    
    # Load validation dataset
    print("\nLoading validation dataset...")
    val_dataset = load_video_dataset(val_dir, class_names)
    
    if val_dataset is None:
        print("Failed to load validation dataset. Exiting.")
        return
    
    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    val_cm, val_accuracy, val_true, val_pred = evaluate_model(model, val_dataset)
    
    # Generate classification report
    val_report = classification_report(
        val_true, 
        val_pred, 
        target_names=english_names,
        zero_division=0
    )
    
    # Save classification report
    with open('evaluation_results/validation_report.txt', 'w') as f:
        f.write(f"Validation Accuracy: {val_accuracy:.2%}\n\n")
        f.write(val_report)
    
    # Plot and save confusion matrix
    try:
        plot_confusion_matrix(
            val_cm, 
            'evaluation_results/validation_confusion_matrix.png',
            'Validation Confusion Matrix',
            use_kannada=True
        )
        
        # Also save with English labels for better readability
        plot_confusion_matrix(
            val_cm, 
            'evaluation_results/validation_confusion_matrix_english.png',
            'Validation Confusion Matrix (English Labels)',
            use_kannada=False
        )
    except Exception as e:
        print(f"Error plotting validation confusion matrix: {e}")
    
    # Load training dataset (optional, can be commented out if too time-consuming)
    print("\nLoading training dataset...")
    train_dataset = load_video_dataset(train_dir, class_names)
    
    if train_dataset is None:
        print("Failed to load training dataset. Skipping training evaluation.")
    else:
        # Evaluate on training set
        print("\nEvaluating on training set...")
        train_cm, train_accuracy, train_true, train_pred = evaluate_model(model, train_dataset)
        
        # Generate classification report
        train_report = classification_report(
            train_true, 
            train_pred, 
            target_names=english_names,
            zero_division=0
        )
        
        # Save classification report
        with open('evaluation_results/training_report.txt', 'w') as f:
            f.write(f"Training Accuracy: {train_accuracy:.2%}\n\n")
            f.write(train_report)
        
        # Plot and save confusion matrix
        try:
            plot_confusion_matrix(
                train_cm, 
                'evaluation_results/training_confusion_matrix.png',
                'Training Confusion Matrix',
                use_kannada=True
            )
            
            # Also save with English labels for better readability
            plot_confusion_matrix(
                train_cm, 
                'evaluation_results/training_confusion_matrix_english.png',
                'Training Confusion Matrix (English Labels)',
                use_kannada=False
            )
        except Exception as e:
            print(f"Error plotting training confusion matrix: {e}")
    
    # Print accuracies
    print(f"\nOverall Accuracy (Validation Set): {val_accuracy:.2%}")
    if train_dataset is not None:
        print(f"Overall Accuracy (Training Set): {train_accuracy:.2%}")

if __name__ == "__main__":
    main()
