import os
import glob
from collections import Counter
import cv2

def check_video_formats():
    """Check the video formats in the dataset"""
    # Define paths
    base_dir = os.getcwd()
    dataset_dir = os.path.join(base_dir, 'dataset')
    processed_dir = os.path.join(dataset_dir, 'processed_data')
    
    train_dir = os.path.join(processed_dir, 'train')
    val_dir = os.path.join(processed_dir, 'val')
    
    # Function to check videos in a directory
    def check_directory(directory):
        if not os.path.exists(directory):
            print(f"Directory not found: {directory}")
            return {}, []
        
        extensions = []
        video_info = []
        
        # Get all video files
        for root, _, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                _, ext = os.path.splitext(file)
                
                if ext.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
                    extensions.append(ext.lower())
                    
                    # Try to get video properties
                    try:
                        cap = cv2.VideoCapture(file_path)
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        duration = frame_count / fps if fps > 0 else 0
                        
                        video_info.append({
                            'path': file_path,
                            'width': width,
                            'height': height,
                            'fps': fps,
                            'frames': frame_count,
                            'duration': duration
                        })
                        
                        cap.release()
                    except Exception as e:
                        print(f"Error reading video {file_path}: {e}")
        
        return Counter(extensions), video_info
    
    # Check train directory
    print(f"Checking videos in training directory: {train_dir}")
    train_extensions, train_info = check_directory(train_dir)
    
    # Check validation directory
    print(f"Checking videos in validation directory: {val_dir}")
    val_extensions, val_info = check_directory(val_dir)
    
    # Print results
    print("\nVideo formats in training set:")
    for ext, count in train_extensions.items():
        print(f"  {ext}: {count} files")
    
    print("\nVideo formats in validation set:")
    for ext, count in val_extensions.items():
        print(f"  {ext}: {count} files")
    
    # Print video statistics
    if train_info:
        widths = [info['width'] for info in train_info]
        heights = [info['height'] for info in train_info]
        fps_values = [info['fps'] for info in train_info]
        frame_counts = [info['frames'] for info in train_info]
        durations = [info['duration'] for info in train_info]
        
        print("\nTraining video statistics:")
        print(f"  Resolution: {min(widths)}x{min(heights)} to {max(widths)}x{max(heights)}")
        print(f"  FPS: {min(fps_values):.1f} to {max(fps_values):.1f}")
        print(f"  Frame count: {min(frame_counts)} to {max(frame_counts)}")
        print(f"  Duration: {min(durations):.1f}s to {max(durations):.1f}s")
    
    if val_info:
        widths = [info['width'] for info in val_info]
        heights = [info['height'] for info in val_info]
        fps_values = [info['fps'] for info in val_info]
        frame_counts = [info['frames'] for info in val_info]
        durations = [info['duration'] for info in val_info]
        
        print("\nValidation video statistics:")
        print(f"  Resolution: {min(widths)}x{min(heights)} to {max(widths)}x{max(heights)}")
        print(f"  FPS: {min(fps_values):.1f} to {max(fps_values):.1f}")
        print(f"  Frame count: {min(frame_counts)} to {max(frame_counts)}")
        print(f"  Duration: {min(durations):.1f}s to {max(durations):.1f}s")
    
    # Check if there are any image files
    train_images = len(glob.glob(os.path.join(train_dir, '**', '*.jpg'), recursive=True))
    train_images += len(glob.glob(os.path.join(train_dir, '**', '*.png'), recursive=True))
    
    val_images = len(glob.glob(os.path.join(val_dir, '**', '*.jpg'), recursive=True))
    val_images += len(glob.glob(os.path.join(val_dir, '**', '*.png'), recursive=True))
    
    print(f"\nImage files found in training directory: {train_images}")
    print(f"Image files found in validation directory: {val_images}")
    
    # Check if there are any other file types
    train_other = []
    val_other = []
    
    for root, _, files in os.walk(train_dir):
        for file in files:
            _, ext = os.path.splitext(file)
            if ext.lower() not in ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.jpg', '.png']:
                train_other.append(ext.lower())
    
    for root, _, files in os.walk(val_dir):
        for file in files:
            _, ext = os.path.splitext(file)
            if ext.lower() not in ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.jpg', '.png']:
                val_other.append(ext.lower())
    
    if train_other:
        print("\nOther file types in training directory:")
        for ext, count in Counter(train_other).items():
            print(f"  {ext}: {count} files")
    
    if val_other:
        print("\nOther file types in validation directory:")
        for ext, count in Counter(val_other).items():
            print(f"  {ext}: {count} files")

if __name__ == "__main__":
    check_video_formats()
