import cv2
import mediapipe as mp
import numpy as np
import os
from sklearn.model_selection import train_test_split

# Initialize MediaPipe Hands (configure for 1 or 2 hands)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# ======================== PATHS ========================
base_input_folder = r"C:\Users\savem\OneDrive\Desktop\kannada_sign_project\dataset\Dataset\raw_videos"
base_output_folder = r"C:\Users\savem\OneDrive\Desktop\kannada_sign_project\dataset\new_processed_video"

# Create main output directory and train/val subdirectories
os.makedirs(os.path.join(base_output_folder, "train"), exist_ok=True)
os.makedirs(os.path.join(base_output_folder, "val"), exist_ok=True)

# ======================== PROCESSING FUNCTION ========================
def extract_keypoints_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    keypoints_sequence = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        # Initialize keypoints for up to 2 hands (zero-padded if not detected)
        frame_keypoints = np.zeros((2, 21, 3))  # Shape: [hands, landmarks, xyz]
        
        if results.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks[:2]):  # Max 2 hands
                frame_keypoints[i] = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
                
        keypoints_sequence.append(frame_keypoints)
    
    cap.release()
    return np.array(keypoints_sequence)  # Shape: [frames, hands, landmarks, xyz]

# ======================== PROCESS & SPLIT ========================
for label in os.listdir(base_input_folder):
    label_folder = os.path.join(base_input_folder, label)
    if not os.path.isdir(label_folder):
        continue
    
    # Get all video files
    video_files = [f for f in os.listdir(label_folder) if f.lower().endswith((".mp4", ".avi", ".mov"))]
    if not video_files:
        continue
    
    # Split into train/val (stratified by label)
    train_files, val_files = train_test_split(video_files, test_size=0.2, random_state=42)
    
    # Process and save train videos
    os.makedirs(os.path.join(base_output_folder, "train", label), exist_ok=True)
    for video_file in train_files:
        video_path = os.path.join(label_folder, video_file)
        keypoints = extract_keypoints_from_video(video_path)
        output_path = os.path.join(base_output_folder, "train", label, f"{os.path.splitext(video_file)[0]}.npy")
        np.save(output_path, keypoints)
        print(f"✅ TRAIN: Processed {label}/{video_file}")
    
    # Process and save val videos
    os.makedirs(os.path.join(base_output_folder, "val", label), exist_ok=True)
    for video_file in val_files:
        video_path = os.path.join(label_folder, video_file)
        keypoints = extract_keypoints_from_video(video_path)
        output_path = os.path.join(base_output_folder, "val", label, f"{os.path.splitext(video_file)[0]}.npy")
        np.save(output_path, keypoints)
        print(f"✅ VAL: Processed {label}/{video_file}")

print("🎉 All videos processed and split into train/val!")