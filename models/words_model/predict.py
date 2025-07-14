import cv2
import numpy as np
import torch
import mediapipe as mp
import time
import os
import sys
import subprocess
from PIL import Image, ImageDraw, ImageFont
from model import KannadaSignModel

# =============================================
# 1. SYSTEM SETUP - ENSURING PROPER ENCODING
# =============================================
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
os.environ["PYTHONIOENCODING"] = "utf-8"

# =============================================
# 2. KANNADA FONT SETUP - GUARANTEED RENDERING
# =============================================
def load_kannada_font():
    """Load Kannada font with multiple fallback options"""
    font_paths = [
        # Windows
        r"C:\Windows\Fonts\Nirmala.ttf",
        r"C:\Users\savem\OneDrive\Desktop\kannada_sign_project\scripts\NotoSansKannada.ttf",
        # Linux
        "/usr/share/fonts/truetype/noto/NotoSansKannada-Regular.ttf",
        # Mac
        "/Library/Fonts/NotoSansKannada-Regular.ttf"
    ]
    
    for path in font_paths:
        try:
            font = ImageFont.truetype(path, 40)
            print(f"Successfully loaded font: {path}")
            return font, ImageFont.truetype(path, 20)
        except Exception as e:
            print(f"Failed to load font {path}: {str(e)}")
    
    print("Warning: Using default font - Kannada may not display correctly")
    return ImageFont.load_default(), ImageFont.load_default()

kannada_font, small_font = load_kannada_font()

# =============================================
# 3. VOICE OUTPUT - MULTIPLE FALLBACK METHODS
# =============================================
def speak_kannada(text):
    """Ultra-reliable Kannada speech with 4 different methods"""
    print(f"Attempting to speak: {text}")
    
    # Method 1: gTTS with direct system playback
    try:
        from gtts import gTTS
        temp_file = "kannada_temp.mp3"
        
        # Create speech file
        tts = gTTS(text=text, lang='kn')
        tts.save(temp_file)
        
        # Play using system command
        if sys.platform == 'win32':
            os.startfile(temp_file)
        elif sys.platform == 'darwin':
            subprocess.call(['afplay', temp_file])
        else:
            subprocess.call(['mpg123', '-q', temp_file])
        
        # Wait for playback to complete
        time.sleep(1)
        os.remove(temp_file)
        print("Successfully spoke using gTTS")
        return
    except Exception as e:
        print(f"gTTS failed: {str(e)}")
    
    # Method 2: pyttsx3 with voice selection
    try:
        import pyttsx3
        engine = pyttsx3.init()
        
        # Try to find Kannada voice
        voices = engine.getProperty('voices')
        for voice in voices:
            if 'kannada' in voice.name.lower() or 'indic' in voice.name.lower():
                engine.setProperty('voice', voice.id)
                break
        
        engine.say(text)
        engine.runAndWait()
        print("Successfully spoke using pyttsx3")
        return
    except Exception as e:
        print(f"pyttsx3 failed: {str(e)}")
    
    # Method 3: System text-to-speech
    try:
        if sys.platform == 'win32':
            import win32com.client
            speaker = win32com.client.Dispatch("SAPI.SpVoice")
            speaker.Speak(text)
        elif sys.platform == 'darwin':
            subprocess.call(['say', text])
        else:
            subprocess.call(['espeak', '-vkn', text])
        print("Successfully spoke using system TTS")
        return
    except Exception as e:
        print(f"System TTS failed: {str(e)}")
    
    # Method 4: Final fallback - beep codes
    print("All speech methods failed - using beep")
    for _ in range(2):
        winsound.Beep(1000, 300)
        time.sleep(0.2)

# =============================================
# 4. TEST VOICE OUTPUT BEFORE MAIN APPLICATION
# =============================================
print("\nTesting voice output with word 'ನಮಸ್ಕಾರ'...")
speak_kannada("ನಮಸ್ಕಾರ")
print("Voice test complete\n")

# =============================================
# 5. MAIN APPLICATION CODE
# =============================================
class SignLanguagePredictor:
    def __init__(self, model_path="checkpoints/best_model.pth"):
        self.classes = ['ಆಡು', 'ಇದೀಯ', 'ಇಲ್ಲ', 'ಇಲ್ಲಿ', 'ಇವತ್ತು', 'ಎಲ್ಲಿ', 'ಎಲ್ಲಿಗೆ', 'ಏಕೆ', 'ಏನು', 'ಕುಳಿತುಕೊ',
                       'ಕೇಳು', 'ಗಲಾಟೆ', 'ಜೊತೆ', 'ತಿಂದೆ', 'ತೆಗೆದುಕೋ', 'ನಾನು', 'ನಿಧವಾಗಿ', 'ನಿನ್ನ', 'ನೀನು',
                       'ಪುಸ್ತಕ', 'ಬಂದರು', 'ಮಾಡಬೇಡಿ', 'ಮಾತು', 'ಯಾರು', 'ವಾಸವಾಗಿ', 'ಸುಮ್ಮನೆ', 'ಹೆಚ್ಚು', 'ಹೋಗಿದ್ದೆ']
        self.model = KannadaSignModel(num_classes=len(self.classes))
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.seq_length = 45
        self.landmark_buffer = []

    def preprocess(self, landmarks):
        if len(landmarks) < self.seq_length:
            padded = np.pad(landmarks, ((0, self.seq_length - len(landmarks)), (0,0), (0,0), (0,0)), mode='edge')
        else:
            padded = landmarks[:self.seq_length]
        return torch.FloatTensor(padded).unsqueeze(0)

    def predict(self):
        if len(self.landmark_buffer) >= 30:
            inputs = self.preprocess(np.array(self.landmark_buffer[-self.seq_length:]))
            with torch.no_grad():
                outputs = self.model(inputs)
                probs = torch.softmax(outputs, dim=1)
                confidence, pred_idx = torch.max(probs, 1)
                return self.classes[pred_idx.item()], confidence.item()
        return None, 0

def main():
    # Initialize mediapipe
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )

    predictor = SignLanguagePredictor()
    cap = cv2.VideoCapture(0)

    # State variables
    last_prediction = ""
    last_display_time = 0
    min_display_duration = 3
    last_spoken_time = 0
    min_speech_interval = 3

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()

        # Process frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        # Extract landmarks
        frame_landmarks = np.zeros((2, 21, 3))
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks[:2]):
                frame_landmarks[i] = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])

        # Update buffer and predict
        predictor.landmark_buffer.append(frame_landmarks)
        if len(predictor.landmark_buffer) > predictor.seq_length * 2:
            predictor.landmark_buffer = predictor.landmark_buffer[-predictor.seq_length:]

        current_pred, confidence = predictor.predict()

        # Update prediction state
        if current_pred and confidence > 0.5:
            if current_pred != last_prediction:
                last_prediction = current_pred
                last_display_time = current_time
                if confidence > 0.7 and (current_time - last_spoken_time) > min_speech_interval:
                    speak_kannada(current_pred)
                    last_spoken_time = current_time

        # Display prediction
        if current_time - last_display_time < min_display_duration and last_prediction:
            # Create text display
            text_img = np.ones((150, 600, 3), dtype=np.uint8) * 255
            pil_img = Image.fromarray(text_img)
            draw = ImageDraw.Draw(pil_img)
            
            # Set color based on confidence
            text_color = (0, 0, 255)  # Default red
            if confidence > 0.8:
                text_color = (0, 128, 0)  # Green
            elif confidence > 0.6:
                text_color = (255, 165, 0)  # Orange
            
            # Draw text
            try:
                text = f"{last_prediction} ({confidence*100:.0f}%)"
                draw.text((10, 10), text, font=kannada_font, fill=text_color)
            except:
                text = "Prediction"
                draw.text((10, 10), text, font=small_font, fill=text_color)
            
            # Combine with main frame
            text_img = np.array(pil_img)
            frame[-150:, :600] = text_img

        cv2.imshow('Kannada Sign Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()