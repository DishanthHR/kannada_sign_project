import cv2
import numpy as np
import mediapipe as mp
import os
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def extract_keypoints_from_video(video_path, visualize=False):
    """
    Extract hand keypoints from a video using MediaPipe.
    
    Args:
        video_path: Path to the video file
        visualize: Whether to show the video with keypoints
        
    Returns:
        keypoints_sequence: Array of shape (frames, hands, keypoints, coordinates)
    """
    # Open video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None
    
    # Get video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Initialize array to store keypoints
    # Shape: (frames, hands, keypoints, coordinates)
    keypoints_sequence = np.zeros((frame_count, 2, 21, 3))
    
    # Process each frame
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame with MediaPipe
        results = hands.process(rgb_frame)
        
        # If hands are detected, extract keypoints
        if results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                if hand_idx >= 2:  # Only process up to 2 hands
                    break
                
                # Draw hand landmarks on the frame if visualization is enabled
                if visualize:
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Extract keypoints
                for i, landmark in enumerate(hand_landmarks.landmark):
                    keypoints_sequence[frame_idx, hand_idx, i, 0] = landmark.x
                    keypoints_sequence[frame_idx, hand_idx, i, 1] = landmark.y
                    keypoints_sequence[frame_idx, hand_idx, i, 2] = landmark.z
        
        # Display the frame if visualization is enabled
        if visualize:
            cv2.imshow('MediaPipe Hands', frame)
            if cv2.waitKey(5) & 0xFF == 27:  # Exit on ESC
                break
        
        frame_idx += 1
    
    # Release resources
    cap.release()
    if visualize:
        cv2.destroyAllWindows()
    
    # Trim the array to the actual number of frames processed
    keypoints_sequence = keypoints_sequence[:frame_idx]
    
    return keypoints_sequence

# Example usage
if __name__ == "__main__":
    # Test on a single video
    video_path = input("Enter the path to a video file: ")
    if os.path.exists(video_path):
        print(f"Processing video: {video_path}")
        keypoints = extract_keypoints_from_video(video_path, visualize=True)
        print(f"Extracted keypoints shape: {keypoints.shape}")
    else:
        print(f"Error: File {video_path} does not exist")
