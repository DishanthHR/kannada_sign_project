import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load your trained model
model_path = 'models/words/kannada_sign_model.h5'
model = load_model(model_path)

# Kannada words mapping (index to word)
kannada_words = [
    'ಆಡು', 'ಇದೀಯ', 'ಇಲ್ಲ', 'ಇಲ್ಲಿ', 'ಇವತ್ತು', 
    'ಎಲ್ಲಿ', 'ಎಲ್ಲಿಗೆ', 'ಏಕೆ', 'ಏನು', 'ಕುಳಿತುಕೊ', 
    'ಕೇಳು', 'ಗಲಾಟೆ', 'ಜೊತೆ', 'ತಿಂದೆ', 'ತೆಗೆದುಕೋ', 
    'ನಾನು', 'ನಿಧವಾಗಿ', 'ನಿನ್ನ', 'ನೀನು', 'ಪುಸ್ತಕ', 
    'ಬಂದರು', 'ಮಾಡಬೇಡಿ', 'ಮಾತು', 'ಯಾರು', 'ವಾಸವಾಗಿ', 
    'ಸುಮ್ಮನೆ', 'ಹೆಚ್ಚು', 'ಹೋಗಿದ್ದೆ'
]

# Create random input data to test model behavior
# Using the correct input shape from your model: (None, 73, 189)
random_input = np.random.random((10, 73, 189))

# Get predictions
predictions = model.predict(random_input)

# Analyze predictions
for i, pred in enumerate(predictions):
    # Get the predicted class and confidence
    predicted_class = np.argmax(pred)
    confidence = pred[predicted_class] * 100
    
    # Get the word
    predicted_word = kannada_words[predicted_class]
    
    print(f"Sample {i+1}: Predicted '{predicted_word}' with {confidence:.2f}% confidence")
    
    # Print top 3 predictions
    top_indices = np.argsort(pred)[-3:][::-1]
    print("Top 3 predictions:")
    for idx in top_indices:
        print(f"  {kannada_words[idx]}: {pred[idx]*100:.2f}%")
    print()

# Check if model is biased toward certain classes
class_counts = [np.argmax(pred) for pred in predictions]
unique, counts = np.unique(class_counts, return_counts=True)

print("Distribution of predictions across classes:")
for u, c in zip(unique, counts):
    print(f"Class {u} ({kannada_words[u]}): {c} occurrences")

# Plot the average confidence for each class
plt.figure(figsize=(15, 6))
avg_confidence = np.mean(predictions, axis=0)
plt.bar(range(len(kannada_words)), avg_confidence)
plt.xticks(range(len(kannada_words)), kannada_words, rotation=90)
plt.title('Average Confidence per Class')
plt.tight_layout()
plt.savefig('class_confidence.png')
print("\nSaved confidence distribution plot to 'class_confidence.png'")
