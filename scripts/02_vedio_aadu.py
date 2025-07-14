import cv2
import mediapipe as mp
import numpy as np
import os

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Folder containing videos for the word "ಆಡು"
video_folder = r"C:\Users\savem\OneDrive\Desktop\kannada_sign_project\dataset\vedio_augmented_data\ಹೋಗಿದ್ದೆ"
output_base_folder = r"C:\Users\savem\OneDrive\Desktop\kannada_sign_project\dataset\vedios_keypoints"

# Create the output folder for "ಆಡು" dynamically
output_folder = os.path.join(output_base_folder, "ಹೋಗಿದ್ದೆ")
os.makedirs(output_folder, exist_ok=True)

def extract_keypoints_from_video(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    keypoints_sequence = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR image to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Extract 21 keypoints (x, y, z coordinates)
                keypoints = []
                for landmark in hand_landmarks.landmark:
                    keypoints.extend([landmark.x, landmark.y, landmark.z])
                keypoints_sequence.append(keypoints)

                # Optional: Draw hand landmarks on the frame (for visualization)
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Save keypoints sequence as a NumPy array
    keypoints_sequence = np.array(keypoints_sequence)
    np.save(output_path, keypoints_sequence)
    cap.release()

# Process all videos in the folder
for video_file in os.listdir(video_folder):
    if video_file.endswith(".mp4"):  # Assuming videos are in .mp4 format
        video_path = os.path.join(video_folder, video_file)
        output_path = os.path.join(output_folder, video_file.replace(".mp4", ".npy"))
        extract_keypoints_from_video(video_path, output_path)
        print(f"Processed {video_file}")