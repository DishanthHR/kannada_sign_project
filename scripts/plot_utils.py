import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import sys

# Add the project root to the path so we can import from scripts
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.font_config import setup_kannada_fonts, get_kannada_class_names, get_english_class_names

def plot_confusion_matrix(cm, output_path, title='Confusion Matrix', use_kannada=True):
    """Plot and save a confusion matrix with proper handling of Kannada labels"""
    # Get class names
    if use_kannada:
        # Try to set up Kannada fonts
        kannada_fonts_available = setup_kannada_fonts()
        if kannada_fonts_available:
            class_names = get_kannada_class_names()
        else:
            # Fall back to English if Kannada fonts not available
            class_names = get_english_class_names()
            use_kannada = False
    else:
        class_names = get_english_class_names()
    
    # Create a figure with appropriate size
    plt.figure(figsize=(20, 18))
    
    # Normalize the confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot the confusion matrix
    ax = sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=class_names, 
                yticklabels=class_names)
    
    # Rotate the tick labels
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    
    # Set title and labels
    plt.title(title, fontsize=16)
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {output_path}")

def plot_accuracy_history(history, output_path):
    """Plot training and validation accuracy history"""
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()
    print(f"Accuracy history saved to {output_path}")

def plot_loss_history(history, output_path):
    """Plot training and validation loss history"""
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()
    print(f"Loss history saved to {output_path}")

def save_sample_frames(dataset, output_dir):
    """Save sample frames from each class for visual inspection"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get class names
    kannada_class_names = get_kannada_class_names()
    english_class_names = get_english_class_names()
    
    # Create a dictionary to store one sample per class
    samples = {}
    
    # Collect one sample per class
    for images, labels in dataset:
        for i, label in enumerate(labels.numpy()):
            if label not in samples:
                samples[label] = images[i].numpy()
                
            # Break if we have samples for all classes
            if len(samples) == len(kannada_class_names):
                break
        
        if len(samples) == len(kannada_class_names):
            break
    
    # Save the samples
    for label, image in samples.items():
        # Convert from [0,1] to [0,255]
        image = (image * 255).astype(np.uint8)
        
        # Get class name
        class_name = kannada_class_names[label]
        english_name = english_class_names[label]
        
        # Save the image
        output_path = os.path.join(output_dir, f"{english_name}.png")
        plt.figure(figsize=(5, 5))
        plt.imshow(image)
        plt.title(f"Class: {class_name}")
        plt.axis('off')
        plt.savefig(output_path)
        plt.close()
    
    print(f"Sample frames saved to {output_dir}")

# Test the functions if this file is run directly
if __name__ == "__main__":
    print("Plot utilities loaded successfully")
    print("This file contains functions for plotting confusion matrices and saving sample frames")
