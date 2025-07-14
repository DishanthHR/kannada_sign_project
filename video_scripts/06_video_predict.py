import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import time
import os
import sys
from tensorflow.keras.models import load_model
import pygame
from collections import deque
import threading
from PIL import Image, ImageDraw, ImageFont
from gtts import gTTS

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Initialize pygame for audio
pygame.mixer.init()

# Define paths
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(base_dir, 'model', 'video_model', 'video_model.h5')
audio_dir = os.path.join(base_dir, 'audio')
font_path = r"C:\Users\savem\OneDrive\Desktop\kannada_sign_project\scripts\NotoSansKannada-VariableFont_wdth,wght.ttf"
temp_dir = os.path.join(base_dir, 'temp')

# Create temp directory if it doesn't exist
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)

# Define the Kannada words (classes)
kannada_words = [
    "ಆಡು", "ಇದೀಯ", "ಇಲ್ಲ", "ಇಲ್ಲಿ", "ಇವತ್ತು", "ಎಲ್ಲಿ", "ಏಕೆ", "ಏನು", 
    "ಕುಳಿತುಕೊ", "ಕೇಳು", "ಗಲಾಟೆ", "ಜೊತೆ", "ತಿಂದೆ", "ತೆಗೆದುಕೋ", "ನಾನು", 
    "ನಿಧಾನವಾಗಿ", "ನಿನ್ನ", "ನೀನು", "ಪುಸ್ತಕ", "ಬಂದರು", "ಮಾಡಬೇಡಿ", "ಮಾತು", 
    "ಯಾರು", "ವಾಸವಾಗಿ", "ಸುಮ್ಮನೆ", "ಹೆಚ್ಚು", "ಹೋಗಿದ್ದೆ"]

class ContinuousSignRecognizer:
    def __init__(self):
        # Load model
        self.model = self.load_trained_model()
        
        # Initialize variables for continuous recognition
        self.keypoints_buffer = deque(maxlen=30)  # Buffer to store recent keypoints
        self.prediction_history = deque(maxlen=10)  # Buffer for prediction smoothing
        self.sentence = []  # Current sentence being built
        self.current_word = None  # Currently detected word
        self.last_word_time = time.time()  # Time when last word was detected
        self.word_timeout = 2.0  # Seconds to wait before considering a new word
        self.sentence_timeout = 4.0  # Seconds to wait before ending sentence (increased to 4 seconds)
        self.last_activity_time = time.time()  # Time of last detected hand activity
        self.confidence_threshold = 0.7  # Minimum confidence to accept a prediction
        self.is_speaking = False  # Flag to track if TTS is active
        self.sentence_confirmed = False  # Flag to track if sentence is confirmed
        self.sentence_confirmation_time = None  # Time when sentence confirmation started
        
        # Load Kannada font
        self.font_size = 30
        try:
            self.kannada_font = ImageFont.truetype(font_path, self.font_size)
            print(f"Loaded Kannada font from {font_path}")
        except Exception as e:
            print(f"Error loading Kannada font: {str(e)}")
            self.kannada_font = None
    
    def load_trained_model(self):
        """Load the trained model"""
        if not os.path.exists(model_path):
            print(f"Error: Model file not found at {model_path}")
            return None
            
        try:
            model = load_model(model_path)
            print(f"Loaded model from {model_path}")
            return model
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return None
    
    def extract_hand_keypoints(self, frame):
        """Extract hand keypoints from a frame using MediaPipe"""
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame with MediaPipe
        results = hands.process(rgb_frame)
        
        # Initialize keypoints array
        keypoints = np.zeros((2, 21, 3))  # 2 hands, 21 keypoints per hand, 3 coordinates (x, y, z)
        
        # If hands are detected, extract keypoints
        if results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                if hand_idx >= 2:  # Only process up to 2 hands
                    break
                
                # Draw hand landmarks on the frame
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Extract keypoints
                for i, landmark in enumerate(hand_landmarks.landmark):
                    keypoints[hand_idx, i, 0] = landmark.x
                    keypoints[hand_idx, i, 1] = landmark.y
                    keypoints[hand_idx, i, 2] = landmark.z
            
            # Update last activity time when hands are detected
            self.last_activity_time = time.time()
        
        return frame, keypoints

    def update_keypoints_buffer(self, keypoints):
        """Update the buffer of keypoints for temporal analysis"""
        # Only add keypoints if hands are detected
        if np.sum(keypoints) > 0:
            self.keypoints_buffer.append(keypoints)
        
        # If buffer is not full, pad with zeros
        while len(self.keypoints_buffer) < 30:
            self.keypoints_buffer.append(np.zeros_like(keypoints))

    def preprocess_keypoints_sequence(self):
        """Preprocess the keypoints sequence for model input"""
        if not self.keypoints_buffer:
            return None
        
        # Convert buffer to numpy array
        sequence = np.array(self.keypoints_buffer)
        
        # Normalize keypoints
        for f in range(sequence.shape[0]):
            for h in range(sequence.shape[1]):
                if np.sum(sequence[f, h]) > 0:  # If hand is detected
                    sequence[f, h, :, 0] = (sequence[f, h, :, 0] - np.min(sequence[f, h, :, 0])) / (np.max(sequence[f, h, :, 0]) - np.min(sequence[f, h, :, 0]) + 1e-10)
                    sequence[f, h, :, 1] = (sequence[f, h, :, 1] - np.min(sequence[f, h, :, 1])) / (np.max(sequence[f, h, :, 1]) - np.min(sequence[f, h, :, 1]) + 1e-10)
        
        # Reshape to match model input shape
        processed_sequence = np.expand_dims(sequence, axis=0)
        
        return processed_sequence
    
    def predict_sign(self):
        """Predict the sign from the current keypoints buffer"""
        if self.model is None:
            return None, 0.0
        
        # Preprocess keypoints sequence
        processed_sequence = self.preprocess_keypoints_sequence()
        if processed_sequence is None:
            return None, 0.0
        
        # Make prediction
        prediction = self.model.predict(processed_sequence, verbose=0)
        
        # Get the class with highest probability
        predicted_class_idx = np.argmax(prediction[0])
        confidence = prediction[0][predicted_class_idx]
        
        return predicted_class_idx, confidence
    
    def update_sentence(self, predicted_class_idx, confidence):
        """Update the sentence based on the current prediction"""
        current_time = time.time()
        
        # Reset sentence confirmation if new activity is detected
        if self.sentence_confirmed and current_time - self.last_activity_time < 1.0:
            self.sentence_confirmed = False
            self.sentence_confirmation_time = None
        
        # Only consider predictions with high confidence
        if confidence >= self.confidence_threshold:
            # Add to prediction history
            self.prediction_history.append(predicted_class_idx)
            
            # Get most common prediction from history
            if self.prediction_history:
                from collections import Counter
                most_common = Counter(self.prediction_history).most_common(1)[0]
                most_common_idx, count = most_common
                
                # If prediction appears in at least half of history
                if count >= len(self.prediction_history) // 2:
                    predicted_word = kannada_words[most_common_idx]
                    
                    # Check if this is a new word (not the same as current word)
                    if self.current_word != predicted_word:
                        # Check if enough time has passed since last word
                        if current_time - self.last_word_time >= self.word_timeout:
                            self.current_word = predicted_word
                            self.last_word_time = current_time
                            
                            # Add word to sentence
                            self.sentence.append(predicted_word)
                            
                            # Play audio for the word (optional - can be disabled)
                            # self.play_audio(predicted_word)
                            
                            # Reset sentence confirmation
                            self.sentence_confirmed = False
                            self.sentence_confirmation_time = None
        
        # Check for sentence timeout (start confirmation)
        if current_time - self.last_activity_time >= self.sentence_timeout and self.sentence and not self.sentence_confirmed:
            self.sentence_confirmed = True
            self.sentence_confirmation_time = current_time
            print(f"Confirming sentence: {' '.join(self.sentence)}")
        
        # Check if confirmation period is complete (3-4 seconds after timeout)
        if self.sentence_confirmed and self.sentence_confirmation_time is not None:
            if current_time - self.sentence_confirmation_time >= 3.0:  # 3 seconds confirmation period
                # Speak the full sentence
                self.speak_full_sentence()
                
                # Reset sentence
                self.sentence = []
                self.sentence_confirmed = False
                self.sentence_confirmation_time = None

    def play_audio(self, word):
        """Play audio for a single word"""
        audio_path = os.path.join(audio_dir, f"{word}.mp3")
        if os.path.exists(audio_path):
            pygame.mixer.music.load(audio_path)
            pygame.mixer.music.play()
        else:
            print(f"Warning: Audio file not found for word '{word}' at {audio_path}")

    def speak_full_sentence(self):
        """Generate and speak the full sentence using TTS"""
        if self.is_speaking or not self.sentence:
            return
        
        # Set speaking flag
        self.is_speaking = True
        
        # Join words into a sentence
        sentence_text = " ".join(self.sentence)
        print(f"Final sentence: {sentence_text}")
        
        # Use a separate thread for TTS to avoid blocking
        def tts_thread():
            try:
                # Generate TTS for the full sentence
                sentence_audio_path = os.path.join(temp_dir, "sentence.mp3")
                tts = gTTS(text=sentence_text, lang='kn', slow=False)
                tts.save(sentence_audio_path)
                
                # Play the full sentence audio
                pygame.mixer.music.load(sentence_audio_path)
                pygame.mixer.music.play()
                
                # Wait for audio to finish
                while pygame.mixer.music.get_busy():
                    pygame.time.wait(100)
                
            except Exception as e:
                print(f"Error in TTS: {str(e)}")
                # Fallback: play individual word audios
                for word in self.sentence:
                    self.play_audio(word)
                    time.sleep(1)  # Wait between words
            
            # Reset speaking flag
            self.is_speaking = False
        
        # Start TTS thread
        threading.Thread(target=tts_thread).start()
    
    def get_sentence_text(self):
        """Get the current sentence as text"""
        if not self.sentence:
            return "ಪ್ರತೀಕ್ಷಿಸುತ್ತಿದೆ..."  # "Waiting..."
        return " ".join(self.sentence)
    
    def run_recognition(self):
        """Run the continuous sign recognition"""
        # Open webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return
        
        print("Starting continuous sign recognition. Press 'q' to quit.")
        
        while cap.isOpened():
            # Read frame from webcam
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from webcam.")
                break
            
            # Extract hand keypoints
            frame, keypoints = self.extract_hand_keypoints(frame)
            
            # Update keypoints buffer
            self.update_keypoints_buffer(keypoints)
            
            # Make prediction every few frames
            if len(self.keypoints_buffer) == 30:  # Buffer is full
                predicted_class_idx, confidence = self.predict_sign()
                
                # Update sentence based on prediction
                if predicted_class_idx is not None:
                    self.update_sentence(predicted_class_idx, confidence)
            
            # Display current word and sentence on frame
            frame = self.display_info_on_frame(frame, confidence if 'confidence' in locals() else 0.0)
            
            # Display the frame
            cv2.imshow('Kannada Sign Language Recognition', frame)
            
            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
    
    def put_kannada_text(self, img, text, position, font_size=None, color=(255, 255, 255)):
        """Put Kannada text on image using PIL"""
        if self.kannada_font is None:
            # Fallback to OpenCV's default font if Kannada font is not available
            cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            return img
        
        # Convert OpenCV image to PIL Image
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        
        # Use specified font size or default
        font = self.kannada_font
        if font_size is not None and font_size != self.font_size:
            try:
                font = ImageFont.truetype(font_path, font_size)
            except:
                font = self.kannada_font
        
        # Draw text
        draw.text(position, text, font=font, fill=color)
        
        # Convert back to OpenCV format
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    
    def display_info_on_frame(self, frame, confidence):
        """Display information on the frame using Kannada font"""
        # Create a copy of the frame to draw on
        display_frame = frame.copy()
        
        # Display current word
        if self.current_word:
            display_frame = self.put_kannada_text(
                display_frame, 
                f"ಪ್ರಸ್ತುತ ಪದ: {self.current_word}", 
                (10, 30), 
                color=(0, 255, 0)
            )
            display_frame = self.put_kannada_text(
                display_frame, 
                f"ವಿಶ್ವಾಸ: {confidence:.2f}", 
                (10, 70), 
                color=(0, 255, 0)
            )
        
        # Display current sentence
        sentence_text = self.get_sentence_text()
        
        # Split long sentences into multiple lines
        max_chars = 40
        y_pos = 110
        
        while sentence_text:
            if len(sentence_text) <= max_chars:
                display_frame = self.put_kannada_text(
                    display_frame, 
                    sentence_text, 
                    (10, y_pos), 
                    color=(255, 0, 0)
                )
                break
            else:
                # Find a good place to split (space)
                split_pos = sentence_text[:max_chars].rfind(' ')
                if split_pos == -1:  # No space found, force split
                    split_pos = max_chars
                
                display_frame = self.put_kannada_text(
                    display_frame, 
                    sentence_text[:split_pos], 
                    (10, y_pos), 
                    color=(255, 0, 0)
                )
                sentence_text = sentence_text[split_pos:].strip()
                y_pos += 40
        
        # Display sentence confirmation status
        current_time = time.time()
        if self.sentence and self.sentence_confirmed:
            # Calculate remaining confirmation time
            if self.sentence_confirmation_time is not None:
                elapsed = current_time - self.sentence_confirmation_time
                remaining = max(0, 3.0 - elapsed)  # 3 seconds confirmation period
                
                # Display confirmation message
                display_frame = self.put_kannada_text(
                    display_frame, 
                    f"ವಾಕ್ಯ ದೃಢೀಕರಣ: {remaining:.1f}s", 
                    (10, display_frame.shape[0] - 40), 
                    color=(0, 0, 255)
                )
        elif self.sentence and not self.sentence_confirmed:
            # Display timeout information
            time_since_last_word = current_time - self.last_word_time
            if time_since_last_word < self.sentence_timeout:
                remaining_time = int(self.sentence_timeout - time_since_last_word)
                display_frame = self.put_kannada_text(
                    display_frame, 
                    f"ವಾಕ್ಯ ಮುಗಿಯುತ್ತದೆ: {remaining_time}s", 
                    (10, display_frame.shape[0] - 40), 
                    color=(0, 0, 255)
                )
        
        return display_frame

def main():
    """Main function for continuous sign recognition"""
    recognizer = ContinuousSignRecognizer()
    recognizer.run_recognition()

if __name__ == "__main__":
    main()
