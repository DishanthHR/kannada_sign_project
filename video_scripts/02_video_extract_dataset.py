import os
import numpy as np
import time
from tqdm import tqdm
import cv2
from pathlib import Path
import sys

# Add the parent directory to the path to import the keypoints extraction function
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from video_scripts.video_keypoints import extract_keypoints_from_video

# Define paths
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Fix the path to use "raw_videos" instead of "raw_vedio"
dataset_dir = os.path.join(base_dir, 'dataset', 'Dataset', 'raw_videos')
output_dir = os.path.join(base_dir, 'dataset', 'vedio_processed_keypoints')

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Define the Kannada words (classes)
kannada_words = [
    "ಆಡು", "ಇದೀಯ", "ಇಲ್ಲ", "ಇಲ್ಲಿ", "ಇವತ್ತು", "ಎಲ್ಲಿ", "ಏಕೆ", "ಏನು", 
    "ಕುಳಿತುಕೊ", "ಕೇಳು", "ಗಲಾಟೆ", "ಜೊತೆ", "ತಿಂದೆ", "ತೆಗೆದುಕೋ", "ನಾನು", 
    "ನಿಧಾನವಾಗಿ", "ನಿನ್ನ", "ನೀನು", "ಪುಸ್ತಕ", "ಬಂದರು", "ಮಾಡಬೇಡಿ", "ಮಾತು", 
    "ಯಾರು", "ವಾಸವಾಗಿ", "ಸುಮ್ಮನೆ", "ಹೆಚ್ಚು", "ಹೋಗಿದ್ದೆ"
]

def process_all_videos():
    """Process all videos in the dataset and save keypoints"""
    # Count total videos for progress tracking
    total_videos = 0
    for word in kannada_words:
        word_dir = os.path.join(dataset_dir, word)
        if os.path.exists(word_dir) and os.path.isdir(word_dir):
            video_files = [f for f in os.listdir(word_dir) if f.endswith('.mp4')]
            total_videos += len(video_files)
    
    print(f"Found {total_videos} videos to process")
    
    # Process each word
    processed_count = 0
    for word in kannada_words:
        word_dir = os.path.join(dataset_dir, word)
        
        if not os.path.exists(word_dir):
            print(f"Warning: Directory for word '{word}' not found at {word_dir}")
            continue
        
        # Create output directory for this word
        word_output_dir = os.path.join(output_dir, word)
        os.makedirs(word_output_dir, exist_ok=True)
        
        # Process each video in this directory
        video_files = [f for f in os.listdir(word_dir) if f.endswith('.mp4')]
        
        for video_idx, video_file in enumerate(video_files):
            video_path = os.path.join(word_dir, video_file)
            output_path = os.path.join(word_output_dir, f"{Path(video_file).stem}.npy")
            
            # Skip if already processed
            if os.path.exists(output_path):
                print(f"Skipping already processed video: {video_path}")
                processed_count += 1
                continue
            
            print(f"Processing video {processed_count+1}/{total_videos}: {video_path}")
            
            try:
                # Extract keypoints
                keypoints = extract_keypoints_from_video(video_path, visualize=False)
                
                if keypoints is not None and keypoints.shape[0] > 0:
                    # Save keypoints
                    np.save(output_path, keypoints)
                    print(f"Saved keypoints to {output_path}, shape: {keypoints.shape}")
                else:
                    print(f"Warning: No keypoints extracted from {video_path}")
            
            except Exception as e:
                print(f"Error processing {video_path}: {str(e)}")
            
            processed_count += 1
    
    print(f"Processed {processed_count} videos")

if __name__ == "__main__":
    start_time = time.time()
    process_all_videos()
    elapsed_time = time.time() - start_time
    print(f"Total processing time: {elapsed_time:.2f} seconds")
