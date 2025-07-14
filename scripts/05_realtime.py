import os
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import joblib
import logging
import pyttsx3
from PIL import Image, ImageDraw, ImageFont
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Paths
MODEL_DIR = r'C:\Users\savem\OneDrive\Desktop\kannada_sign_project\trained_model'
CLASSES_PATH = os.path.join(MODEL_DIR, 'classes.npy')
FONT_PATH = r'C:\Users\savem\OneDrive\Desktop\kannada_sign_project\scripts\NotoSansKannada-VariableFont_wdth,wght.ttf'

# Initialize text-to-speech engine
try:
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)  # Speed of speech
    engine.setProperty('volume', 0.9)  # Volume (0.0 to 1.0)
    logging.info("Text-to-speech engine initialized")
except Exception as e:
    logging.error(f"Failed to initialize text-to-speech: {str(e)}")
    engine = None

# Load model and preprocessing components
try:
    logging.info("Loading model and preprocessing components...")
    model = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'final_model'))
    scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
    classes = np.load(CLASSES_PATH, allow_pickle=True)
    logging.info(f"Loaded model with {len(classes)} classes")
except Exception as e:
    logging.error(f"Error loading model components: {str(e)}")
    raise

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,  # For video
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def detect_hand_type(results):
    """Detect if the hand is left or right"""
    if not results.multi_handedness:
        return None
    
    handedness = results.multi_handedness[0].classification[0].label
    return handedness  # "Left" or "Right"

def extract_keypoints(results, use_mirroring=True):
    """Extract and normalize hand keypoints similar to training pipeline"""
    if not results.multi_hand_landmarks:
        return None, None
    
    hand_landmarks = results.multi_hand_landmarks[0]
    handedness = detect_hand_type(results)
    
    # Extract raw keypoints
    keypoints = []
    for landmark in hand_landmarks.landmark:
        keypoints.extend([landmark.x, landmark.y, landmark.z])
    
    # Convert to numpy array
    keypoints_array = np.array(keypoints).reshape(-1, 3)
    
    # If right hand and mirroring is enabled, mirror the x-coordinates to match left hand
    # This helps if the model was primarily trained on one hand type
    if use_mirroring and handedness == "Right":
        keypoints_array[:, 0] = 1.0 - keypoints_array[:, 0]
    
    # Normalize similar to training pipeline
    center = np.mean(keypoints_array, axis=0)
    centered_keypoints = keypoints_array - center
    
    # Scale using wrist to middle finger tip distance
    wrist = keypoints_array[0]
    middle_tip = keypoints_array[12]
    scale = np.linalg.norm(middle_tip - wrist)
    
    if scale > 1e-6:
        normalized_keypoints = centered_keypoints / scale
    else:
        normalized_keypoints = centered_keypoints
    
    return normalized_keypoints.flatten(), handedness

def put_kannada_text(img, text, position, font_size, color):
    """Add Kannada text to an OpenCV image using PIL"""
    # Convert OpenCV image (BGR) to PIL image (RGB)
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    # Create a draw object
    draw = ImageDraw.Draw(pil_img)
    
    # Use the Kannada font
    try:
        font = ImageFont.truetype(FONT_PATH, font_size)
    except Exception as e:
        logging.error(f"Error loading font: {str(e)}")
        # Fallback to default font
        font = ImageFont.load_default()
    
    # Draw text
    draw.text(position, text, font=font, fill=color)
    
    # Convert back to OpenCV format
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def main():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("Cannot open camera")
        return
    
    logging.info("Starting real-time recognition...")
    
    # For FPS calculation
    prev_frame_time = 0
    new_frame_time = 0
    
    # For smoothing predictions
    last_predictions = []
    prediction_buffer_size = 5
    
    # For speech output
    last_spoken_letter = ""
    last_spoken_time = 0
    speech_cooldown = 2.0  # seconds between speech outputs
    
    # Hand type toggle
    use_hand_mirroring = True
    
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            logging.error("Failed to grab frame")
            break
        
        # Calculate FPS
        new_frame_time = cv2.getTickCount()
        fps = cv2.getTickFrequency() / (new_frame_time - prev_frame_time) if prev_frame_time > 0 else 0
        prev_frame_time = new_frame_time
        
        # Flip frame for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = hands.process(rgb_frame)
        
        # Create a blank info panel
        info_panel = np.zeros((frame.shape[0], 300, 3), dtype=np.uint8)
        
        # Create a large letter display panel at the top
        letter_panel_height = 150
        letter_panel = np.zeros((letter_panel_height, frame.shape[1] + 300, 3), dtype=np.uint8)
        
        # Draw hand landmarks and make prediction
        predicted_letter = ""
        confidence = 0.0
        handedness = None
        
        if results.multi_hand_landmarks:
            # Draw landmarks on frame
            mp_drawing.draw_landmarks(
                frame, 
                results.multi_hand_landmarks[0], 
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            )
            
            # Extract keypoints
            keypoints, handedness = extract_keypoints(results, use_hand_mirroring)
            
            if keypoints is not None:
                try:
                    # Scale keypoints
                    keypoints_scaled = scaler.transform(keypoints.reshape(1, -1))
                    
                    # Predict
                    prediction = model.predict(keypoints_scaled, verbose=0)[0]
                    
                    # Get top 3 predictions
                    top_indices = np.argsort(prediction)[-3:][::-1]
                    top_classes = [classes[i] for i in top_indices]
                    top_probs = [prediction[i] * 100 for i in top_indices]
                    
                    # Add to prediction buffer for smoothing
                    last_predictions.append(top_indices[0])
                    if len(last_predictions) > prediction_buffer_size:
                        last_predictions.pop(0)
                    
                    # Get most common prediction from buffer
                    from collections import Counter
                    most_common = Counter(last_predictions).most_common(1)[0][0]
                    smoothed_class = classes[most_common]
                    predicted_letter = smoothed_class
                    confidence = top_probs[0] if top_indices[0] == most_common else prediction[most_common] * 100
                    
                    # Display on info panel
                    cv2.putText(info_panel, "Kannada Sign Recognition", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    
                    # Use regular OpenCV text for English
                    cv2.putText(info_panel, f"Prediction:", (10, 70), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    
                    # Add Kannada text for the prediction
                    info_panel = put_kannada_text(info_panel, smoothed_class, (180, 60), 36, (0, 255, 0))
                    
                    # Display hand type
                    hand_color = (0, 255, 255) if use_hand_mirroring else (255, 255, 0)
                    cv2.putText(info_panel, f"Hand: {handedness}", (10, 110), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, hand_color, 2)
                    
                    # Display mirroring status
                    mirror_status = "ON" if use_hand_mirroring else "OFF"
                    cv2.putText(info_panel, f"Mirroring: {mirror_status}", (150, 110), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, hand_color, 2)
                    
                    # Display top 3 predictions
                    cv2.putText(info_panel, "Top 3 Predictions:", (10, 140), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    for i, (cls, prob) in enumerate(zip(top_classes, top_probs)):
                        color = (0, 255, 0) if i == 0 else (200, 200, 200)
                        y_pos = 170 + i*40
                        cv2.putText(info_panel, f"{i+1}. ", (10, y_pos), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        info_panel = put_kannada_text(info_panel, cls, (40, y_pos-15), 24, color)
                        cv2.putText(info_panel, f": {prob:.1f}%", (70, y_pos), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    # Speak the letter if it's stable and different from last spoken
                    current_time = time.time()
                    if (engine is not None and 
                        predicted_letter != last_spoken_letter and 
                        current_time - last_spoken_time > speech_cooldown and
                        len(set(last_predictions)) == 1):  # Only speak when prediction is stable
                        
                        engine.say(predicted_letter)
                        engine.runAndWait()
                        last_spoken_letter = predicted_letter
                        last_spoken_time = current_time
                    
                except Exception as e:
                    logging.error(f"Prediction error: {str(e)}")
                    cv2.putText(info_panel, "Error making prediction", (10, 70), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            # No hand detected
            cv2.putText(info_panel, "No hand detected", (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            # Clear prediction buffer
            last_predictions = []
        
        # Display FPS
        cv2.putText(info_panel, f"FPS: {fps:.1f}", (10, info_panel.shape[0] - 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display instructions
        cv2.putText(info_panel, "Press 'q' to quit", (10, info_panel.shape[0] - 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(info_panel, "Press 'm' to toggle mirroring", (10, info_panel.shape[0] - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display large letter in the letter panel using Kannada font
        if predicted_letter:
            # Add the large Kannada letter to the panel
            letter_panel = put_kannada_text(
                letter_panel, 
                predicted_letter, 
                ((letter_panel.shape[1] - 100) // 2, 30), 
                80, 
                (0, 255, 0)
            )
            
            # Add confidence
            conf_text = f"Confidence: {confidence:.1f}%"
            conf_size = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
            conf_x = (letter_panel.shape[1] - conf_size[0]) // 2
            cv2.putText(letter_panel, conf_text, (conf_x, 120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
        else:
            # Display "No prediction" when no letter is detected
            no_pred_text = "No prediction"
            text_size = cv2.getTextSize(no_pred_text, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 3)[0]
            text_x = (letter_panel.shape[1] - text_size[0]) // 2
            text_y = (letter_panel_height + text_size[1]) // 2
            cv2.putText(letter_panel, no_pred_text, (text_x, text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 3)
        
        # Combine frame and info panel horizontally
        combined_frame = np.hstack((frame, info_panel))
        
        # Combine frame and info panel horizontally
        combined_frame = np.hstack((frame, info_panel))
        
        # Combine with letter panel vertically
        final_frame = np.vstack((letter_panel, combined_frame))
        
        # Resize if too large for screen
        screen_width = 1280
        screen_height = 720
        if final_frame.shape[1] > screen_width or final_frame.shape[0] > screen_height:
            scale_w = screen_width / final_frame.shape[1]
            scale_h = screen_height / final_frame.shape[0]
            scale = min(scale_w, scale_h)
            final_frame = cv2.resize(final_frame, None, fx=scale, fy=scale)
        
        # Show the frame
        cv2.imshow('Kannada Sign Language Recognition', final_frame)
        
        # Check for key presses
        key = cv2.waitKey(1) & 0xFF
        
        # Exit on pressing 'q'
        if key == ord('q'):
            break
        
        # Toggle hand mirroring on pressing 'm'
        elif key == ord('m'):
            use_hand_mirroring = not use_hand_mirroring
            logging.info(f"Hand mirroring {'enabled' if use_hand_mirroring else 'disabled'}")
            # Clear prediction buffer when changing mirroring
            last_predictions = []
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    logging.info("Application closed")

if __name__ == "__main__":
    main()

