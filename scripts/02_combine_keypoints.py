import os
import numpy as np

# Mapping of words (gestures) to numeric labels
word_to_label = {
    "ಆಡು": 0,
    "ಇದೀಯ": 1,
    "ಇಲ್ಲ": 2,
    "ಇಲ್ಲಿ": 3,
    "ಇವತ್ತು": 4,
    "ಎಲ್ಲಿ": 5,
    "ಎಲ್ಲಿಗೆ": 6,
    "ಏಕೆ": 7,
    "ಏನು": 8,
    "ಕುಳಿತುಕೊ": 9,
    "ಕೇಳು": 10,
    "ಗಲಾಟೆ": 11,
    "ಜೊತೆ": 12,
    "ತಿಂದೆ": 13,
    "ತೆಗೆದುಕೋ": 14,
    "ನಾನು": 15,
    "ನಿಧವಾಗಿ": 16,
    "ನಿನ್ನ": 17,
    "ನೀನು": 18,
    "ಪುಸ್ತಕ": 19,
    "ಬಂದರು": 20,
    "ಮಾಡಬೇಡಿ": 21,
    "ಮಾತು": 22,
    "ಯಾರು": 23,
    "ವಾಸವಾಗಿ": 24,
    "ಸುಮ್ಮನೆ": 25,
    "ಹೆಚ್ಚು": 26,
    "ಹೋಗಿದ್ದೆ": 27
}

# Paths
keypoints_folder = r"C:\Users\savem\OneDrive\Desktop\kannada_sign_project\dataset\vedios_keypoints"  # Folder containing all gestures
output_file = r"C:\Users\savem\OneDrive\Desktop\kannada_sign_project\dataset\vedio_combined_keypoints"  # Output file for combined dataset

# Initialize lists to store data and labels
dataset = []
labels = []

# Process each word folder
for word, label in word_to_label.items():
    word_folder = os.path.join(keypoints_folder, word)
    if not os.path.exists(word_folder):
        print(f"Warning: Folder not found for {word}")
        continue
    
    for file in os.listdir(word_folder):
        if file.endswith(".npy"):
            file_path = os.path.join(word_folder, file)
            keypoints_sequence = np.load(file_path)
            dataset.append(keypoints_sequence)
            labels.append(label)

# Save the combined dataset and labels
np.save(output_file, {"data": dataset, "labels": labels})
print(f"✅ Combined dataset saved to {output_file}")