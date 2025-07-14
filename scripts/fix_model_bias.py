import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

# Load your trained model
model_path = 'models/words/kannada_sign_model.h5'
original_model = load_model(model_path)

# Kannada words mapping
kannada_words = [
    'ಆಡು', 'ಇದೀಯ', 'ಇಲ್ಲ', 'ಇಲ್ಲಿ', 'ಇವತ್ತು', 
    'ಎಲ್ಲಿ', 'ಎಲ್ಲಿಗೆ', 'ಏಕೆ', 'ಏನು', 'ಕುಳಿತುಕೊ', 
    'ಕೇಳು', 'ಗಲಾಟೆ', 'ಜೊತೆ', 'ತಿಂದೆ', 'ತೆಗೆದುಕೋ', 
    'ನಾನು', 'ನಿಧವಾಗಿ', 'ನಿನ್ನ', 'ನೀನು', 'ಪುಸ್ತಕ', 
    'ಬಂದರು', 'ಮಾಡಬೇಡಿ', 'ಮಾತು', 'ಯಾರು', 'ವಾಸವಾಗಿ', 
    'ಸುಮ್ಮನೆ', 'ಹೆಚ್ಚು', 'ಹೋಗಿದ್ದೆ'
]

# Generate random test data
num_samples = 100
random_inputs = np.random.random((num_samples, 73, 189))

# Get predictions from original model
original_preds = original_model.predict(random_inputs)
original_classes = np.argmax(original_preds, axis=1)

# Count occurrences of each class
unique, counts = np.unique(original_classes, return_counts=True)
class_distribution = dict(zip(unique, counts))

print("Original model class distribution:")
for cls, count in class_distribution.items():
    print(f"Class {cls} ({kannada_words[cls]}): {count} occurrences ({count/num_samples*100:.2f}%)")

# Calculate class weights (inverse of frequency)
total_samples = num_samples
class_weights = {}
for cls in range(len(kannada_words)):
    if cls in class_distribution:
        # Inverse frequency with smoothing
        class_weights[cls] = total_samples / (counts[np.where(unique == cls)[0][0]] * len(kannada_words))
    else:
        # For classes that didn't appear in predictions, give high weight
        class_weights[cls] = 3.0

print("\nCalculated class weights:")
for cls, weight in class_weights.items():
    print(f"Class {cls} ({kannada_words[cls]}): {weight:.4f}")

# Create a bias correction layer
# Get the output of the second-to-last layer
penultimate_layer = original_model.layers[-2].output
output_dim = original_model.layers[-1].output_shape[-1]

# Create a new output layer with bias correction
new_output = Dense(output_dim, activation='softmax', name='new_output')(penultimate_layer)

# Create the corrected model
corrected_model = Model(inputs=original_model.input, outputs=new_output)

# Compile the model
corrected_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Generate synthetic training data with class labels
# We'll create balanced synthetic data
synthetic_data = []
synthetic_labels = []

samples_per_class = 50
for cls in range(len(kannada_words)):
    # Create slightly different random data for each class
    class_data = np.random.random((samples_per_class, 73, 189))
    # Add a small bias toward the class to help the model learn
    bias_factor = 0.1
    for i in range(samples_per_class):
        # Add a small pattern specific to this class
        class_data[i, :, cls % 189] += bias_factor
    
    synthetic_data.append(class_data)
    synthetic_labels.append(np.ones(samples_per_class) * cls)

# Combine all synthetic data
synthetic_data = np.vstack(synthetic_data)
synthetic_labels = np.concatenate(synthetic_labels)

# Shuffle the data
indices = np.arange(len(synthetic_data))
np.random.shuffle(indices)
synthetic_data = synthetic_data[indices]
synthetic_labels = synthetic_labels[indices]

# Train the corrected model on synthetic data
print("\nTraining bias correction layer...")
corrected_model.fit(
    synthetic_data, synthetic_labels,
    epochs=10,
    batch_size=32,
    class_weight=class_weights,
    verbose=1
)

# Test the corrected model
corrected_preds = corrected_model.predict(random_inputs)
corrected_classes = np.argmax(corrected_preds, axis=1)

# Count occurrences of each class in corrected predictions
unique_corrected, counts_corrected = np.unique(corrected_classes, return_counts=True)
corrected_distribution = dict(zip(unique_corrected, counts_corrected))

print("\nCorrected model class distribution:")
for cls, count in corrected_distribution.items():
    print(f"Class {cls} ({kannada_words[cls]}): {count} occurrences ({count/num_samples*100:.2f}%)")

# Save the corrected model
corrected_model.save('models/words/kannada_sign_model_corrected.h5')
print("\nCorrected model saved to 'models/words/kannada_sign_model_corrected.h5'")

# Plot comparison
plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
plt.bar(range(len(kannada_words)), 
        [class_distribution.get(i, 0) for i in range(len(kannada_words))])
plt.title('Original Model Predictions')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks(range(len(kannada_words)), range(len(kannada_words)), rotation=90)

plt.subplot(1, 2, 2)
plt.bar(range(len(kannada_words)), 
        [corrected_distribution.get(i, 0) for i in range(len(kannada_words))])
plt.title('Corrected Model Predictions')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks(range(len(kannada_words)), range(len(kannada_words)), rotation=90)

plt.tight_layout()
plt.savefig('model_correction_comparison.png')
print("Comparison plot saved to 'model_correction_comparison.png'")
