import os
import sys

# Fix module import path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from font_config import get_kannada_class_names, get_english_class_names, setup_kannada_fonts
except ImportError:
    from scripts.font_config import get_kannada_class_names, get_english_class_names, setup_kannada_fonts

import os
import sys
import cv2
import numpy as np
import tensorflow as tf
from collections import deque
import time
import pygame
from gtts import gTTS
import threading
import h5py
from scripts.font_config import get_kannada_class_names, get_english_class_names, setup_kannada_fonts

def resolve_model_path(input_path=None):
    """Convert any path format to absolute path and verify"""
    if input_path and os.path.exists(input_path):
        return os.path.abspath(input_path)
    
    possible_paths = [
        r"C:\Users\savem\OneDrive\Desktop\kannada_sign_project\models\words\best_model.h5",
        os.path.join(os.getcwd(), "models", "words", "best_model.h5"),
        os.path.join(os.path.dirname(__file__), "models", "words", "best_model.h5")
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return os.path.abspath(path)
    
    raise FileNotFoundError(f"Model not found in: {possible_paths}")

class GestureRecognizer:
    def __init__(self, model_path, num_frames=15, target_size=(64, 64)):
        """Initialize with guaranteed path resolution"""
        # Hardware setup
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam")
            
        # Model loading
        self.model_path = resolve_model_path(model_path)
        print(f"Confirmed model location: {self.model_path}")
        
        try:
            with h5py.File(self.model_path, 'r') as f:
                if 'model_weights' not in f.keys():
                    raise ValueError("Invalid model structure")
            self.model = tf.keras.models.load_model(self.model_path)
            print("Model successfully loaded")
        except Exception as e:
            print(f"Model loading error: {str(e)}")
            raise

        # Motion detection
        self.motion_threshold = 0.015
        self.stillness_threshold = 0.005
        self.stillness_frames_required = 3
        
        # Frame processing
        self.frame_buffer = deque(maxlen=30)
        self.target_size = target_size
        self.num_frames = num_frames
        
        # Class names
        self.kannada_class_names = get_kannada_class_names()
        self.english_class_names = get_english_class_names()
        
        # Audio setup
        pygame.mixer.init()
        self.audio_cache = {}
        self.is_speaking = False
        
        # State management
        self.state = "WAITING"
        self.last_motion_time = 0
        self.current_result = None
        
        # Sentence mode
        self.sentence_mode = False
        self.detected_signs = []
        
        # Debug
        self.debug_mode = False
        self.kannada_font = setup_kannada_fonts()

    # ... [Keep all other methods EXACTLY AS IS from your original file] ...
    # preprocess_frame(), detect_motion(), run_webcam(), process_gesture(), 
    # display_result(), toggle_sentence_mode(), speak_text() methods remain unchanged


    def preprocess_frame(self, frame):
        """Enhanced preprocessing with contrast boost"""
        frame = cv2.resize(frame, self.target_size)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.convertScaleAbs(frame, alpha=1.3, beta=20)  # Boost contrast
        return frame / 255.0

    def detect_motion(self, frame):
        """High-sensitivity motion detection"""
        if len(self.frame_buffer) < 2:
            return 0
            
        # Enhanced motion detection with contrast
        prev = cv2.cvtColor(self.frame_buffer[-1], cv2.COLOR_BGR2GRAY)
        prev = cv2.convertScaleAbs(prev, alpha=1.5, beta=0)
        curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        curr = cv2.convertScaleAbs(curr, alpha=1.5, beta=0)
        
        diff = cv2.absdiff(prev, curr)
        _, thresh = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
        
        motion_level = np.sum(thresh) / (255 * thresh.size)
        return motion_level * 1.5  # Sensitivity boost

    def run_webcam(self):
        """Main loop with guaranteed motion detection"""
        cv2.namedWindow("Kannada Sign Language", cv2.WINDOW_NORMAL)
        
        # Force first frame capture
        ret, frame = self.cap.read()
        self.frame_buffer.append(frame.copy())
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            frame = cv2.flip(frame, 1)
            display_frame = frame.copy()
            
            # Always store frames
            self.frame_buffer.append(frame.copy())
            motion_level = self.detect_motion(frame)
            
            # DEBUG: Show motion level
            if self.debug_mode:
                print(f"Motion: {motion_level:.4f}")
                cv2.putText(display_frame, f"Motion: {motion_level:.4f}", 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                
            # State machine with forced recording
            if self.state == "WAITING":
                if motion_level > self.motion_threshold or len(self.frame_buffer) > 20:
                    self.state = "RECORDING"
                    print("Recording started!")
                    
            elif self.state == "RECORDING":
                # Visual feedback
                cv2.circle(display_frame, (30, 30), 10, (0,0,255), -1)
                
                # Process after stillness or timeout
                if motion_level < self.stillness_threshold:
                    if len(self.frame_buffer) >= self.num_frames:
                        self.state = "PROCESSING"
                elif time.time() - self.last_motion_time > 5.0:  # 5s timeout
                    self.state = "WAITING"
                    
            elif self.state == "PROCESSING":
                self.process_gesture()
                self.state = "RESULT"
                
            elif self.state == "RESULT":
                self.display_result(display_frame)
                if time.time() - self.last_motion_time > 3.0:
                    self.state = "WAITING"
            
            # Keyboard controls
            key = cv2.waitKey(1) & 0xFF
            if key == 27: break  # ESC
            elif key == ord(' '): self.toggle_sentence_mode()
            elif key == ord('d'): self.debug_mode = not self.debug_mode
            
            cv2.imshow("Kannada Sign Language", display_frame)
            
        self.cap.release()
        cv2.destroyAllWindows()

    def process_gesture(self):
        """Process and predict gesture"""
        frames = [self.preprocess_frame(f) for f in list(self.frame_buffer)[-self.num_frames:]]
        preds = self.model.predict(np.array([frames]), verbose=0)[0]
        self.current_result = np.argmax(preds)
        
        # Store result if in sentence mode
        if self.sentence_mode:
            self.detected_signs.append(self.kannada_class_names[self.current_result])
        
        # Speak the result
        self.speak_text(self.kannada_class_names[self.current_result])

    def display_result(self, frame):
        """Show prediction results"""
        cv2.putText(frame, 
                   f"Prediction: {self.kannada_class_names[self.current_result]}",
                   (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        
        if self.sentence_mode and self.detected_signs:
            sentence = " ".join(self.detected_signs)
            cv2.putText(frame, f"Sentence: {sentence}",
                      (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,255), 2)

    def toggle_sentence_mode(self):
        """Toggle sentence collection mode"""
        self.sentence_mode = not self.sentence_mode
        if not self.sentence_mode and self.detected_signs:
            self.speak_text(" ".join(self.detected_signs))
            self.detected_signs = []

    def speak_text(self, text):
        """Threaded text-to-speech"""
        if self.is_speaking: return
        
        def _speak():
            try:
                if text not in self.audio_cache:
                    tts = gTTS(text=text, lang='kn')
                    tts.save(f"temp_{hash(text)}.mp3")
                    self.audio_cache[text] = f"temp_{hash(text)}.mp3"
                
                pygame.mixer.music.load(self.audio_cache[text])
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10)
            except Exception as e:
                print(f"Speech error: {e}")
            finally:
                self.is_speaking = False
                
        self.is_speaking = True
        threading.Thread(target=_speak).start()

def main():
    try:
        model_path = sys.argv[1] if len(sys.argv) > 1 else None
        recognizer = GestureRecognizer(model_path)
        recognizer.run_webcam()
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Verify webcam is working")
        print("2. Check model file exists at the shown path")
        print("3. Try: python gesture_recognizer.py \"C:\\full\\path\\to\\model.h5\"")
        input("Press Enter to exit...")
        sys.exit(1)

if __name__ == "__main__":
    main()