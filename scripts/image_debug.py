import cv2
import torch
import numpy as np
import mediapipe as mp
import os
import json
import matplotlib.pyplot as plt

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Path configuration
MODEL_DIR = r"C:\\Users\\savem\\OneDrive\\Desktop\\kannada_sign_project\\models\\kannada_sign_model_20250509-213234"
CLASS_MAPPING_PATH = os.path.join(MODEL_DIR, "class_mapping.json")
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pth")

# Load class mapping
with open(CLASS_MAPPING_PATH, 'r', encoding='utf-8') as f:
    class_mapping = json.load(f)
classes = [class_mapping[str(i)] for i in range(len(class_mapping))]
num_classes = len(classes)

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

# Load model
model = KannadaSignModel(63, num_classes).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# Try different preprocessing methods
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

# Capture a single frame
def capture_frame():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return None
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Error: Could not read frame")
        return None
    
    return frame

# Process frame with MediaPipe
def process_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    if not results.multi_hand_landmarks:
        print("No hand detected")
        return None
    
    # Draw landmarks on frame
    hand_landmarks = results.multi_hand_landmarks[0]
    frame_with_landmarks = frame.copy()
    mp_drawing.draw_landmarks(
        frame_with_landmarks, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    # Save frame with landmarks
    cv2.imwrite('debug_hand.jpg', frame_with_landmarks)
    print("Saved hand image to debug_hand.jpg")
    
    return hand_landmarks

# Make predictions with different preprocessing methods
def predict_with_methods(landmarks):
    results = []
    
    for method_num, preprocess_func in enumerate([
        preprocess_method1, preprocess_method2, preprocess_method3, preprocess_method4
    ], 1):
        try:
            keypoints = preprocess_func(landmarks.landmark)
            
            # Print keypoints range
            print(f"Method {method_num} - Keypoints min: {keypoints.min():.2f}, max: {keypoints.max():.2f}")
            
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
                'top3': top3,
                'keypoints': keypoints
            })
            
            print(f"Method {method_num} - Prediction: {prediction}, Confidence: {confidence:.2%}")
            print(f"Method {method_num} - Top 3: {top3}")
            
        except Exception as e:
            print(f"Error with method {method_num}: {e}")
    
    return results

# Visualize keypoints
def visualize_keypoints(results):
    plt.figure(figsize=(15, 10))
    
    for i, result in enumerate(results):
        keypoints = result['keypoints'].reshape(21, 3)
        
        # Plot X coordinates
        plt.subplot(4, 3, i*3 + 1)
        plt.title(f"{result['method']} - X coords\nPred: {result['prediction']} ({result['confidence']:.2%})")
        plt.plot(keypoints[:, 0])
        plt.grid(True)
        
        # Plot Y coordinates
        plt.subplot(4, 3, i*3 + 2)
        plt.title(f"{result['method']} - Y coords")
        plt.plot(keypoints[:, 1])
        plt.grid(True)
        
        # Plot Z coordinates
        plt.subplot(4, 3, i*3 + 3)
        plt.title(f"{result['method']} - Z coords")
        plt.plot(keypoints[:, 2])
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('keypoints_comparison.png')
    plt.close()
    print("Saved keypoints visualization to keypoints_comparison.png")

def main():
    print("Capturing frame...")
    frame = capture_frame()
    if frame is None:
        return
    
    print("Processing frame with MediaPipe...")
    landmarks = process_frame(frame)
    if landmarks is None:
        return
    
    print("Making predictions with different preprocessing methods...")
    results = predict_with_methods(landmarks)
    
    print("Visualizing keypoints...")
    visualize_keypoints(results)
    
    print("Done!")

if __name__ == "__main__":
    main()
