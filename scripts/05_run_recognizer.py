import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import time
from gtts import gTTS
import os
import pygame
from pygame import mixer
import threading
from tensorflow.keras.models import load_model
from PIL import Image, ImageDraw, ImageFont

# Initialize mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Load the trained model
model = load_model('models/words/kannada_sign_model.h5')

# Kannada words mapping (index to word)
kannada_words = [
    'ಆಡು', 'ಇದೀಯ', 'ಇಲ್ಲ', 'ಇಲ್ಲಿ', 'ಇವತ್ತು', 
    'ಎಲ್ಲಿ', 'ಎಲ್ಲಿಗೆ', 'ಏಕೆ', 'ಏನು', 'ಕುಳಿತುಕೊ', 
    'ಕೇಳು', 'ಗಲಾಟೆ', 'ಜೊತೆ', 'ತಿಂದೆ', 'ತೆಗೆದುಕೋ', 
    'ನಾನು', 'ನಿಧವಾಗಿ', 'ನಿನ್ನ', 'ನೀನು', 'ಪುಸ್ತಕ', 
    'ಬಂದರು', 'ಮಾಡಬೇಡಿ', 'ಮಾತು', 'ಯಾರು', 'ವಾಸವಾಗಿ', 
    'ಸುಮ್ಮನೆ', 'ಹೆಚ್ಚು', 'ಹೋಗಿದ್ದೆ'
]

# Initialize pygame for audio playback
pygame.init()
mixer.init()

# Function to extract hand keypoints from mediapipe results
def extract_hand_keypoints(results):
    # Based on your model's input shape, we need to generate 189 features
    # For hands only, we'll extract detailed hand features and pad if necessary
    
    # Initialize empty arrays for hand features
    hand_features = np.zeros(189)
    
    # Extract hand landmarks if detected
    if results.multi_hand_landmarks:
        feature_index = 0
        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Extract coordinates and additional features
            for landmark in hand_landmarks.landmark:
                # Extract x, y, z coordinates
                if feature_index < 189:
                    hand_features[feature_index] = landmark.x
                    feature_index += 1
                if feature_index < 189:
                    hand_features[feature_index] = landmark.y
                    feature_index += 1
                if feature_index < 189:
                    hand_features[feature_index] = landmark.z
                    feature_index += 1
                
                # You can add more derived features if needed
                # For example: distances between landmarks, angles, etc.
    
    return hand_features

# Function to speak the recognized word in Kannada
def speak_kannada_word(word):
    try:
        # Use Kannada for TTS
        tts = gTTS(text=word, lang='kn')
        tts.save("temp_speech.mp3")
        
        # Play the audio
        mixer.music.load("temp_speech.mp3")
        mixer.music.play()
        
        # Wait for audio to finish
        while mixer.music.get_busy():
            pygame.time.Clock().tick(10)
            
        # Clean up
        mixer.music.unload()
        os.remove("temp_speech.mp3")
    except Exception as e:
        print(f"Error in Kannada speech synthesis: {e}")

# Function to create a frame with Kannada text
def create_text_frame(word, confidence, frame_shape):
    # Create a blank image
    img = Image.new('RGB', (frame_shape[1], 150), color=(0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Load Kannada font
    try:
        font_path = "C:\\Users\\savem\\OneDrive\\Desktop\\kannada_sign_project\\scripts\\NotoSansKannada-VariableFont_wdth,wght.ttf"
        font = ImageFont.truetype(font_path, 48)
    except Exception as e:
        print(f"Error loading font: {e}")
        font = ImageFont.load_default()
    
    # Draw the text
    text = f"{word} ({confidence:.2f}%)"
    draw.text((20, 50), text, font=font, fill=(255, 255, 255))
    
    # Convert to numpy array for OpenCV
    return np.array(img)

# Sequence length from your model
sequence_length = 73  # As determined from your model's input shape
feature_dimension = 189  # As determined from your model's input shape

# Main function for real-time recognition
def main():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    # Set mediapipe model for hands only
    with mp_hands.Hands(
        model_complexity=1,  # Use a more complex model for better accuracy
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        max_num_hands=2) as hands:
        
        # Variables for sequence collection
        sequence = []
        threshold = 0.7  # Confidence threshold
        
        # Variables for cooldown between predictions
        last_prediction_time = 0
        cooldown_time = 3  # seconds
        
        # Current prediction
        current_word = ""
        current_confidence = 0
        
        while cap.isOpened():
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
                
            # Mirror the frame for more intuitive interaction
            frame = cv2.flip(frame, 1)
            
            # Convert to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # To improve performance, optionally mark the image as not writeable
            image.flags.writeable = False
            
            # Make detection
            results = hands.process(image)
            
            # Draw the hand annotations on the image
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
            
            # Extract hand keypoints
            keypoints = extract_hand_keypoints(results)
            
            # Update sequence
            sequence.append(keypoints)
            
            # Keep only the last 'sequence_length' frames
            if len(sequence) > sequence_length:
                sequence = sequence[-sequence_length:]
            
            # Make prediction when we have enough frames
            if len(sequence) == sequence_length:
                # Current time for cooldown
                current_time = time.time()
                
                # Only make a new prediction if cooldown has passed
                if current_time - last_prediction_time > cooldown_time:
                    # Prepare the sequence for the model
                    input_data = np.expand_dims(np.array(sequence), axis=0)
                    
                    # Make prediction
                    res = model.predict(input_data, verbose=0)[0]
                    
                    # Get the predicted class and confidence
                    predicted_class = np.argmax(res)
                    confidence = res[predicted_class] * 100
                    
                    # Only update if confidence is above threshold
                    if confidence > threshold * 100:
                        current_word = kannada_words[predicted_class]
                        current_confidence = confidence
                        
                        # Speak the Kannada word in a separate thread
                        threading.Thread(target=speak_kannada_word, args=(current_word,)).start()
                        
                        # Update last prediction time
                        last_prediction_time = current_time
            
            # Display sequence collection progress
            progress = min(len(sequence) / sequence_length * 100, 100)
            cv2.rectangle(image, (0, 0), (int(progress * frame.shape[1] / 100), 10), (0, 255, 0), -1)
            
            # Create text frame with current prediction
            if current_word:
                text_frame = create_text_frame(current_word, current_confidence, frame.shape)
                
                # Combine frames
                combined_frame = np.vstack([image, text_frame])
                
                # Display the combined frame
                cv2.imshow('ಕನ್ನಡ ಸೈನ್ ಲ್ಯಾಂಗ್ವೇಜ್ ರೆಕಗ್ನಿಷನ್', combined_frame)
            else:
                # Display just the camera frame if no prediction yet
                cv2.imshow('ಕನ್ನಡ ಸೈನ್ ಲ್ಯಾಂಗ್ವೇಜ್ ರೆಕಗ್ನಿಷನ್', image)
            
            # Break loop on 'q' press
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
