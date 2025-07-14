import cv2
import torch
import numpy as np
import mediapipe as mp
import os
import json
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd

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
RESULTS_DIR = "manual_test_results"

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
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

def preprocess_landmarks(landmarks):
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

def analyze_test_results():
    # Find the most recent CSV file
    csv_files = [f for f in os.listdir(RESULTS_DIR) if f.endswith('.csv')]
    if not csv_files:
        print("No test results found")
        return
    
    latest_csv = sorted(csv_files)[-1]
    csv_path = os.path.join(RESULTS_DIR, latest_csv)
    
    # Load the results
    df = pd.read_csv(csv_path)
    
    # Create confusion matrix for each method
    for method_num in range(1, 5):
        true_signs = df['true_sign'].tolist()
        pred_signs = df[f'method{method_num}_pred'].tolist()
        
        # Get unique labels in the correct order
        labels = sorted(list(set(true_signs)))
        
        # Create confusion matrix
        cm = confusion_matrix(true_signs, pred_signs, labels=labels)
        
        # Plot confusion matrix
        plt.figure(figsize=(16, 14))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.title(f'Confusion Matrix - Method {method_num}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f'confusion_matrix_method{method_num}.png'))
        plt.close()
        
        print(f"Saved confusion matrix for Method {method_num}")
    
    # Calculate accuracy for each method
    for method_num in range(1, 5):
        correct = sum(df['true_sign'] == df[f'method{method_num}_pred'])
        total = len(df)
        accuracy = correct / total
        print(f"Method {method_num} accuracy: {accuracy:.2%}")
    
    # Find most confused pairs
    for method_num in range(1, 5):
        confusion_pairs = []
        for i, row in df.iterrows():
            true_sign = row['true_sign']
            pred_sign = row[f'method{method_num}_pred']
            conf = row[f'method{method_num}_conf']
            if true_sign != pred_sign:
                confusion_pairs.append((true_sign, pred_sign, conf))
        
        # Sort by confidence (highest first)
        confusion_pairs.sort(key=lambda x: x[2], reverse=True)
        
        print(f"\nTop confused pairs for Method {method_num}:")
        for true, pred, conf in confusion_pairs[:10]:
            print(f"  True: {true}, Predicted: {pred}, Confidence: {conf:.2%}")
    
    # Analyze which signs are most often correctly recognized
    correct_counts = {}
    for sign in classes:
        correct_counts[sign] = 0
    
    for i, row in df.iterrows():
        true_sign = row['true_sign']
        for method_num in range(1, 5):
            pred_sign = row[f'method{method_num}_pred']
            if true_sign == pred_sign:
                correct_counts[true_sign] += 1
    
    # Sort by number of correct predictions
    sorted_correct = sorted(correct_counts.items(), key=lambda x: x[1], reverse=True)
    
    print("\nSigns by recognition success (across all methods):")
    for sign, count in sorted_correct:
        print(f"  {sign}: {count}/4 correct")

def visualize_hand_landmarks():
    # Find all image files
    image_files = [f for f in os.listdir(RESULTS_DIR) if f.endswith('.jpg')]
    if not image_files:
        print("No test images found")
        return
    
    # Create a directory for visualizations
    viz_dir = os.path.join(RESULTS_DIR, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    for img_file in image_files:
        # Extract the true sign from the filename
        true_sign = img_file.split('_')[0]
        
        # Load the image
        img_path = os.path.join(RESULTS_DIR, img_file)
        image = cv2.imread(img_path)
        
        # Process with MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        
        if not results.multi_hand_landmarks:
            print(f"No hand landmarks detected in {img_file}")
            continue
        
        # Get landmarks
        landmarks = results.multi_hand_landmarks[0].landmark
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Original image with landmarks
        axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0].set_title(f"Original Image - True Sign: {true_sign}")
        axes[0].axis('off')
        
        # 3D visualization of normalized landmarks
        keypoints = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
        
        # Center and normalize for visualization
        hand_center = np.mean(keypoints, axis=0)
        centered = keypoints - hand_center
        
        wrist = keypoints[0]
        middle_tip = keypoints[12]
        scale = np.linalg.norm(middle_tip - wrist)
        
        if scale > 0:
            normalized = centered / scale
        else:
            normalized = centered
        
        # Plot 3D points
        ax = axes[1]
        ax.scatter(normalized[:, 0], normalized[:, 1], c='blue', s=20)
        
        # Connect landmarks with lines
        connections = mp_hands.HAND_CONNECTIONS
        for connection in connections:
            start_idx = connection[0]
            end_idx = connection[1]
            ax.plot([normalized[start_idx, 0], normalized[end_idx, 0]],
                    [normalized[start_idx, 1], normalized[end_idx, 1]],
                    'r-', linewidth=1)
        
        # Add labels to key points
        for i, point in enumerate(normalized):
            ax.annotate(str(i), (point[0], point[1]), fontsize=8)
        
        ax.set_title("Normalized Hand Landmarks")
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.grid(True)
        
        # Make predictions with the model
        processed = preprocess_landmarks(landmarks)
        inputs = torch.FloatTensor(processed).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)[0]
            top3_probs, top3_idxs = torch.topk(probs, 3)
            top3 = [(classes[i], p.item()) for i, p in zip(top3_idxs, top3_probs)]
        
        # Add prediction info to the plot
        pred_text = f"Predictions:\n"
        for i, (letter, prob) in enumerate(top3):
            pred_text += f"{i+1}. {letter}: {prob:.2%}\n"
        
        fig.text(0.5, 0.01, pred_text, ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
        
        # Save the visualization
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, f"viz_{true_sign}.png"), bbox_inches='tight')
        plt.close()
        
        print(f"Saved visualization for {true_sign}")

def main():
    print("Analyzing test results...")
    analyze_test_results()
    
    print("\nVisualizing hand landmarks...")
    visualize_hand_landmarks()
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
