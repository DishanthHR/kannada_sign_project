# Manual testing script for Kannada sign recognition
import cv2
import torch
import numpy as np
import mediapipe as mp
import time
import os
import json
import csv
from datetime import datetime

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Path configuration
MODEL_DIR = os.path.join(os.getcwd(), "models", "kannada_sign_model_20250509-213234")
CLASS_MAPPING_PATH = os.path.join(MODEL_DIR, "class_mapping.json")
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pth")

class KannadaSignModel(torch.nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 512), torch.nn.BatchNorm1d(512), torch.nn.ReLU(), torch.nn.Dropout(0.3),
            torch.nn.Linear(512, 256), torch.nn.BatchNorm1d(256), torch.nn.ReLU(), torch.nn.Dropout(0.3),
            torch.nn.Linear(256, 128), torch.nn.BatchNorm1d(128), torch.nn.ReLU(), torch.nn.Dropout(0.2),
            torch.nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.net(x)

# Load class mapping and model
try:
    with open(CLASS_MAPPING_PATH, 'r', encoding='utf-8') as f:
        class_mapping = json.load(f)
    classes = [class_mapping[str(i)] for i in range(len(class_mapping))]
    num_classes = len(classes)
    
    model = KannadaSignModel(63, num_classes).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    
    print(f"Model loaded with {num_classes} classes")
    print(f"Classes: {classes}")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Try multiple preprocessing methods
def preprocess_method1(landmarks):
    """Original training preprocessing"""
    keypoints = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
    
    # Center
    hand_center = np.mean(keypoints, axis=0)
    centered = keypoints - hand_center
    
    # Scale
    wrist = keypoints[0]
    middle_tip = keypoints[12]
    scale = np.linalg.norm(middle_tip - wrist)
    
    if scale > 0:
        normalized = centered / scale
    else:
        normalized = centered
    
    return normalized.flatten()

def preprocess_method2(landmarks):
    """Zero Z method"""
    keypoints = np.array([[lm.x, lm.y, 0.0] for lm in landmarks], dtype=np.float32)
    
    # Center
    hand_center = np.mean(keypoints, axis=0)
    centered = keypoints - hand_center
    
    # Scale
    wrist = keypoints[0]
    middle_tip = keypoints[12]
    scale = np.linalg.norm(middle_tip - wrist)
    
    if scale > 0:
        normalized = centered / scale
    else:
        normalized = centered
    
    return normalized.flatten()

def preprocess_method3(landmarks):
    """XY only for calculations + Zero Z"""
    keypoints = np.array([[lm.x, lm.y, 0.0] for lm in landmarks], dtype=np.float32)
    
    # Center using only X,Y
    hand_center = np.mean(keypoints[:, :2], axis=0)
    for i in range(len(keypoints)):
        keypoints[i, 0] -= hand_center[0]
        keypoints[i, 1] -= hand_center[1]
    
    # Scale using only X,Y
    wrist = keypoints[0, :2]
    middle_tip = keypoints[12, :2]
    scale = np.sqrt(np.sum((middle_tip - wrist)**2))
    
    if scale > 0:
        for i in range(len(keypoints)):
            keypoints[i, 0] /= scale
            keypoints[i, 1] /= scale
    
    return keypoints.flatten()

def preprocess_method4(landmarks):
    """Method 3 + Clipping"""
    keypoints = np.array([[lm.x, lm.y, 0.0] for lm in landmarks], dtype=np.float32)
    
    # Center using only X,Y
    hand_center = np.mean(keypoints[:, :2], axis=0)
    for i in range(len(keypoints)):
        keypoints[i, 0] -= hand_center[0]
        keypoints[i, 1] -= hand_center[1]
    
    # Scale using only X,Y
    wrist = keypoints[0, :2]
    middle_tip = keypoints[12, :2]
    scale = np.sqrt(np.sum((middle_tip - wrist)**2))
    
    if scale > 0:
        for i in range(len(keypoints)):
            keypoints[i, 0] /= scale
            keypoints[i, 1] /= scale
    
    # Clip values
    keypoints = np.clip(keypoints, -1.0, 0.7)
    
    return keypoints.flatten()

def predict_with_all_methods(landmarks):
    results = []
    
    for method_num, preprocess_func in enumerate([
        preprocess_method1, preprocess_method2, preprocess_method3, preprocess_method4
    ], 1):
        try:
            keypoints = preprocess_func(landmarks)
            
            # Make prediction
            inputs = torch.FloatTensor(keypoints).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)[0]
                top3_probs, top3_idxs = torch.topk(probs, 3)
                top3 = [(classes[i], p.item()) for i, p in zip(top3_idxs, top3_probs)]
                
                conf, pred_idx = torch.max(probs, 0)
                confidence = conf.item()
                prediction = classes[pred_idx.item()]
            
            results.append({
                'method': f"Method {method_num}",
                'prediction': prediction,
                'confidence': confidence,
                'top3': top3
            })
            
        except Exception as e:
            print(f"Error with method {method_num}: {e}")
    
    return results

def main():
    # Create results directory
    results_dir = "manual_test_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Create CSV file for results
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    csv_path = os.path.join(results_dir, f"manual_test_{timestamp}.csv")
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['true_sign', 'method1_pred', 'method1_conf', 'method2_pred', 'method2_conf', 
                     'method3_pred', 'method3_conf', 'method4_pred', 'method4_conf']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Test each sign
        for true_sign in classes:
            print(f"\n=== Testing sign: {true_sign} ===")
            print(f"Please make the sign for '{true_sign}' and press SPACE when ready")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error reading frame")
                    break
                
                # Flip the frame horizontally
                frame = cv2.flip(frame, 1)
                
                # Process the frame with MediaPipe
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(frame_rgb)
                
                # Display instructions
                cv2.putText(frame, f"Make sign for: {true_sign}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, "Press SPACE when ready", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, "Press ESC to skip", (10, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # If hand landmarks are detected
                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    
                    # Draw landmarks on frame
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                        mp.solutions.drawing_styles.get_default_hand_connections_style())
                
                # Display the frame
                cv2.imshow("Manual Testing (SPACE=Capture, ESC=Skip, Q=Quit)", frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Quitting...")
                    cap.release()
                    cv2.destroyAllWindows()
                    return
                elif key == 27:  # ESC
                    print(f"Skipping {true_sign}")
                    break
                elif key == 32:  # SPACE
                    if not results.multi_hand_landmarks:
                        print("No hand detected. Please try again.")
                        continue
                    
                    # Save the frame
                    frame_path = os.path.join(results_dir, f"{true_sign}_{timestamp}.jpg")
                    cv2.imwrite(frame_path, frame)
                    print(f"Saved frame to {frame_path}")
                    
                    # Get predictions with all methods
                    hand_landmarks = results.multi_hand_landmarks[0]
                    predictions = predict_with_all_methods(hand_landmarks.landmark)
                    
                    # Display results
                    print("Predictions:")
                    for pred in predictions:
                        print(f"  {pred['method']}: {pred['prediction']} ({pred['confidence']:.2%})")
                    
                    # Write to CSV
                    row = {'true_sign': true_sign}
                    for i, pred in enumerate(predictions, 1):
                        row[f'method{i}_pred'] = pred['prediction']
                        row[f'method{i}_conf'] = pred['confidence']
                    
                    writer.writerow(row)
                    csvfile.flush()  # Ensure data is written immediately
                    
                    break
        
        cap.release()
        cv2.destroyAllWindows()
        print(f"\nTest completed. Results saved to {csv_path}")

if __name__ == "__main__":
    main()
