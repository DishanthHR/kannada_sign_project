import os
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import glob
import logging
import traceback
import time
import sys
from PIL import Image
import gc  # Garbage collection

# Set up logging
log_file = 'keypoint_extraction.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()  # Avoid sys.stdout handle issue
    ]
)

# Set up MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Define input and output paths
INPUT_DIR = r"C:\Users\savem\OneDrive\Desktop\kannada_sign_project\dataset\Dataset\images"
OUTPUT_DIR = r"C:\Users\savem\OneDrive\Desktop\kannada_sign_project\dataset\processed_keypoints"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'visualizations'), exist_ok=True)

def extract_keypoints(image_path, visualize=False):
    try:
        if not os.path.isfile(image_path):
            logging.warning(f"File does not exist: {image_path}")
            return None
        
        with Image.open(image_path) as pil_image:
            image = np.array(pil_image.convert('RGB'))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        with mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5) as hands:
            
            results = hands.process(image_rgb)

            if not results.multi_hand_landmarks:
                return None

            hand_landmarks = results.multi_hand_landmarks[0]
            keypoints = []
            for landmark in hand_landmarks.landmark:
                keypoints.extend([landmark.x, landmark.y, landmark.z])

            keypoints_array = np.array(keypoints).reshape(-1, 3)
            center = np.mean(keypoints_array, axis=0)
            centered_keypoints = keypoints_array - center

            wrist = keypoints_array[0]
            middle_tip = keypoints_array[12]
            scale = np.linalg.norm(middle_tip - wrist)
            if scale > 1e-6:
                normalized_keypoints = centered_keypoints / scale
            else:
                normalized_keypoints = centered_keypoints

            if visualize:
                vis_image = image.copy()
                mp_drawing.draw_landmarks(
                    vis_image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS)
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                class_name = os.path.basename(os.path.dirname(image_path))
                vis_path = os.path.join(OUTPUT_DIR, 'visualizations', f"{class_name}_{base_name}_landmarks.jpg")
                cv2.imwrite(vis_path, vis_image)

            return normalized_keypoints.flatten()
    except Exception as e:
        logging.error(f"Error processing {image_path}: {str(e)}")
        return None

def process_class(class_dir):
    class_data = []
    class_name = os.path.basename(class_dir)

    image_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']:
        image_paths.extend(glob.glob(os.path.join(class_dir, ext)))

    logging.info(f"Found {len(image_paths)} images for class {class_name}")

    successful = 0
    failed = 0

    for idx, image_path in enumerate(image_paths):
        keypoints = extract_keypoints(image_path, visualize=(idx < 5))
        if keypoints is not None:
            successful += 1
            class_data.append({
                'image_path': image_path,
                'class': class_name,
                'keypoints': keypoints
            })
        else:
            failed += 1

        if idx % 50 == 0:
            logging.info(f"{class_name}: Processed {idx}/{len(image_paths)} images")

    logging.info(f"Class {class_name}: {successful} successful, {failed} failed")
    return class_data

def process_dataset():
    all_data = []
    try:
        image_dirs = [d for d in glob.glob(os.path.join(INPUT_DIR, '*')) if os.path.isdir(d)]
        logging.info(f"Found {len(image_dirs)} directories in input path")

        if not image_dirs:
            logging.warning(f"No directories found in {INPUT_DIR}. Check path.")
            return
        
        for dir_idx, dir_path in enumerate(image_dirs):
            logging.info(f"Processing class: {os.path.basename(dir_path)} ({dir_idx+1}/{len(image_dirs)})")
            class_data = process_class(dir_path)
            all_data.extend(class_data)

            if class_data:
                temp_df = pd.DataFrame(all_data)
                temp_df.to_pickle(os.path.join(OUTPUT_DIR, 'keypoints_temp.pkl'))
                logging.info(f"Saved intermediate results: {len(all_data)} images")

            gc.collect()

        if not all_data:
            logging.warning("No data processed. Check input images.")
            return

        df = pd.DataFrame(all_data)
        df.to_pickle(os.path.join(OUTPUT_DIR, 'keypoints.pkl'))
        logging.info("Saved keypoints.pkl")

        keypoints_array = np.array([kp for kp in df['keypoints'].values])
        labels = np.array([cls for cls in df['class'].values])
        np.savez(os.path.join(OUTPUT_DIR, 'keypoints.npz'), keypoints=keypoints_array, labels=labels)
        logging.info("Saved keypoints.npz")

        class_counts = df['class'].value_counts()
        logging.info("\nClass distribution:")
        for class_name, count in class_counts.items():
            logging.info(f"{class_name}: {count} images")

    except Exception as e:
        logging.error(f"Error in process_dataset: {str(e)}")
        traceback.print_exc()

        if all_data:
            df = pd.DataFrame(all_data)
            df.to_pickle(os.path.join(OUTPUT_DIR, 'keypoints_partial.pkl'))
            logging.info("Saved partial results before crash.")

if __name__ == "__main__":
    start_time = time.time()
    logging.info("Starting keypoint extraction...")
    logging.info(f"Python version: {sys.version}")
    logging.info(f"OpenCV version: {cv2.__version__}")
    logging.info(f"PIL version: {Image.__version__}")

    process_dataset()

    end_time = time.time()
    logging.info(f"Keypoint extraction completed in {end_time - start_time:.2f} seconds.")
