import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import sys

# Add the parent directory to the path to import the model definition
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from video_scripts.video_model_define import create_lite_transformer_model, create_cnn_lstm_model
from video_scripts.video_load_dataset import load_processed_data

# Define paths
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
models_dir = os.path.join(base_dir, 'model', 'video_model')
os.makedirs(models_dir, exist_ok=True)

# Define the Kannada words (classes)
kannada_words = [
    "ಆಡು", "ಇದೀಯ", "ಇಲ್ಲ", "ಇಲ್ಲಿ", "ಇವತ್ತು", "ಎಲ್ಲಿ", "ಏಕೆ", "ಏನು", 
    "ಕುಳಿತುಕೊ", "ಕೇಳು", "ಗಲಾಟೆ", "ಜೊತೆ", "ತಿಂದೆ", "ತೆಗೆದುಕೋ", "ನಾನು", 
    "ನಿಧಾನವಾಗಿ", "ನಿನ್ನ", "ನೀನು", "ಪುಸ್ತಕ", "ಬಂದರು", "ಮಾಡಬೇಡಿ", "ಮಾತು", 
    "ಯಾರು", "ವಾಸವಾಗಿ", "ಸುಮ್ಮನೆ", "ಹೆಚ್ಚು", "ಹೋಗಿದ್ದೆ"
]

def train_model(X_train, X_val, y_train, y_val, epochs=50, batch_size=32, model_type='transformer'):
    """
    Train the model on the provided data.
    
    Args:
        X_train: Training data
        X_val: Validation data
        y_train: Training labels
        y_val: Validation labels
        epochs: Number of training epochs
        batch_size: Batch size for training
        model_type: Type of model to train ('transformer' or 'cnn_lstm')
        
    Returns:
        model: Trained model
        history: Training history
    """
    # Get input shape from training data
    input_shape = X_train.shape[1:]
    num_classes = y_train.shape[1]
    
    print(f"Input shape: {input_shape}, Number of classes: {num_classes}")
    
    # Create model based on model_type
    if model_type == 'transformer':
        print("Creating transformer model...")
        model = create_lite_transformer_model(input_shape, num_classes)
    else:
        print("Creating CNN-LSTM model...")
        model = create_cnn_lstm_model(input_shape, num_classes)
    
    model.summary()
    
    # Define callbacks
    checkpoint_path = os.path.join(models_dir, 'video_model.h5')
    checkpoint = ModelCheckpoint(
        checkpoint_path,
        monitor='val_accuracy',
        verbose=1,
        save_best_only=True,
        mode='max'
    )
    
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        verbose=1,
        restore_best_weights=True
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    
    callbacks = [checkpoint, early_stopping, reduce_lr]
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save the final model
    final_model_path = os.path.join(models_dir, 'video_model.h5')
    model.save(final_model_path)
    print(f"Saved final model to {final_model_path}")
    
    return model, history

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on the test data.
    
    Args:
        model: Trained model
        X_test: Test data
        y_test: Test labels
        
    Returns:
        test_loss: Test loss
        test_accuracy: Test accuracy
    """
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
    print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_accuracy:.4f}")
    
    # Get predictions
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    # Calculate confusion matrix
    from sklearn.metrics import confusion_matrix, classification_report
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
    
    return test_loss, test_accuracy

def plot_training_history(history):
    """
    Plot the training history.
    
    Args:
        history: Training history
    """
    # Plot accuracy
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(models_dir, 'training_history.png')
    plt.savefig(output_path)
    plt.close()
    
    print(f"Saved training history plot to {output_path}")

def save_model_summary(model):
    """
    Save the model summary to a text file.
    
    Args:
        model: Keras model
    """
    output_path = os.path.join(models_dir, 'model_summary.txt')
    
    # Redirect stdout to file to capture summary
    import sys
    original_stdout = sys.stdout
    with open(output_path, 'w') as f:
        sys.stdout = f
        model.summary()
    sys.stdout = original_stdout
    
    print(f"Saved model summary to {output_path}")

if __name__ == "__main__":
    # Load processed data
    data = load_processed_data()
    
    if data is None:
        print("Error: Could not load processed data. Please run 03_video_load_dataset.py first.")
        sys.exit(1)
    
    X_train, X_val, X_test = data['X_train'], data['X_val'], data['X_test']
    y_train, y_val, y_test = data['y_train'], data['y_val'], data['y_test']
    
    # Choose model type: 'transformer' or 'cnn_lstm'
    model_type = 'cnn_lstm'  # You can change this to 'transformer' if you want to try that model
    
    # Train model
    model, history = train_model(X_train, X_val, y_train, y_val, epochs=50, batch_size=32, model_type=model_type)
    
    # Evaluate model
    evaluate_model(model, X_test, y_test)
    
    # Plot training history
    plot_training_history(history)
    
    # Save model summary
    save_model_summary(model)
    
    print("Model training completed!")
