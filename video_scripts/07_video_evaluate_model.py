import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import sys

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from video_scripts.video_load_dataset import load_processed_data

# Define paths
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
models_dir = os.path.join(base_dir, 'model', 'video_model')
model_path = os.path.join(models_dir, 'video_model.h5')

# Define the Kannada words (classes)
kannada_words = [
    "ಆಡು", "ಇದೀಯ", "ಇಲ್ಲ", "ಇಲ್ಲಿ", "ಇವತ್ತು", "ಎಲ್ಲಿ", "ಏಕೆ", "ಏನು", 
    "ಕುಳಿತುಕೊ", "ಕೇಳು", "ಗಲಾಟೆ", "ಜೊತೆ", "ತಿಂದೆ", "ತೆಗೆದುಕೋ", "ನಾನು", 
    "ನಿಧಾನವಾಗಿ", "ನಿನ್ನ", "ನೀನು", "ಪುಸ್ತಕ", "ಬಂದರು", "ಮಾಡಬೇಡಿ", "ಮಾತು", 
    "ಯಾರು", "ವಾಸವಾಗಿ", "ಸುಮ್ಮನೆ", "ಹೆಚ್ಚು", "ಹೋಗಿದ್ದೆ"
]

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on the test data.
    
    Args:
        model: Trained model
        X_test: Test data
        y_test: Test labels
    """
    # Evaluate model
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
    print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_accuracy:.4f}")
    
    # Get predictions
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    
    # Print classification report
    print("\nClassification Report:")
    report = classification_report(y_true_classes, y_pred_classes, target_names=kannada_words)
    print(report)
    
    # Save classification report to file - use UTF-8 encoding for Kannada characters
    report_path = os.path.join(models_dir, 'classification_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("Test accuracy: {:.4f}\n\n".format(test_accuracy))
        f.write(report)
    
    print(f"Saved classification report to {report_path}")
    
    # Plot confusion matrix
    plt.figure(figsize=(15, 15))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    # Add labels
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Save confusion matrix
    cm_path = os.path.join(models_dir, 'confusion_matrix.png')
    plt.savefig(cm_path)
    plt.close()
    
    print(f"Saved confusion matrix to {cm_path}")
    
    # Create a more detailed confusion matrix with labels
    plt.figure(figsize=(20, 20))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix', fontsize=16)
    plt.colorbar()
    
    # Add class labels
    tick_marks = np.arange(len(kannada_words))
    plt.xticks(tick_marks, kannada_words, rotation=45, fontsize=10)
    plt.yticks(tick_marks, kannada_words, fontsize=10)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=10)
    
    plt.tight_layout()
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    
    # Save detailed confusion matrix
    detailed_cm_path = os.path.join(models_dir, 'detailed_confusion_matrix.png')
    plt.savefig(detailed_cm_path)
    plt.close()
    
    print(f"Saved detailed confusion matrix to {detailed_cm_path}")

if __name__ == "__main__":
    # Load processed data
    data = load_processed_data()
    
    if data is None:
        print("Error: Could not load processed data.")
        sys.exit(1)
    
    # Get test data
    X_test = data['X_test']
    y_test = data['y_test']
    
    # Load model
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        sys.exit(1)
    
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    
    # Evaluate model
    evaluate_model(model, X_test, y_test)
    
    print("Evaluation completed!")
