# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
import random
import sys
from tqdm import tqdm
import shutil
from concurrent.futures import ThreadPoolExecutor
import logging

# Ensure UTF-8 encoding for logging Kannada
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('video_augmentation.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

class VideoAugmenter:
    def __init__(self, input_dir, output_dir, target_count=50):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.target_count = target_count
        os.makedirs(output_dir, exist_ok=True)
        self.aug_params = {
            'color_prob': 0.7,
            'geo_prob': 0.6,
            'noise_prob': 0.3,
            'speed_prob': 0.8,
            'drop_prob': 0.4,
            'max_rotation': 10,
            'brightness_range': (0.7, 1.3),
            'scale_range': (0.9, 1.1),
            'blur_prob': 0.2
        }

    def _apply_color_augmentation(self, frame):
        try:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hsv[..., 1] = np.clip(hsv[..., 1] * random.uniform(0.8, 1.2), 0, 255)
            hsv[..., 2] = np.clip(hsv[..., 2] * random.uniform(*self.aug_params['brightness_range']), 0, 255)
            frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            alpha = random.uniform(0.8, 1.2)
            frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=0)
            return frame
        except Exception as e:
            logging.error(f"Color augmentation failed: {str(e)}")
            return frame

    def _apply_geometric_augmentation(self, frame):
        try:
            h, w = frame.shape[:2]
            angle = random.uniform(-self.aug_params['max_rotation'], self.aug_params['max_rotation'])
            scale = random.uniform(*self.aug_params['scale_range'])
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, scale)
            frame = cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
            return frame
        except Exception as e:
            logging.error(f"Geometric augmentation failed: {str(e)}")
            return frame

    def _apply_noise(self, frame):
        try:
            noise = np.random.normal(0, random.uniform(1, 3), frame.shape)
            return np.clip(frame + noise, 0, 255).astype(np.uint8)
        except Exception as e:
            logging.error(f"Noise addition failed: {str(e)}")
            return frame

    def _apply_blur(self, frame):
        try:
            k = random.choice([3, 5, 7])
            kernel = np.ones((k, k)) / (k ** 2)
            return cv2.filter2D(frame, -1, kernel)
        except Exception as e:
            logging.error(f"Blur failed: {str(e)}")
            return frame

    def _read_video_frames(self, video_path):
        try:
            cap = cv2.VideoCapture(video_path)
            frames = []
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            cap.release()
            return frames, fps
        except Exception as e:
            logging.error(f"Failed to read video {video_path}: {str(e)}")
            return None, 30

    def _write_video(self, frames, output_path, fps=30):
        try:
            if not frames:
                return False
            h, w = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
            for frame in frames:
                out.write(frame)
            out.release()
            return True
        except Exception as e:
            logging.error(f"Failed to write video {output_path}: {str(e)}")
            return False

    def _temporal_augment(self, frames):
        try:
            if random.random() < self.aug_params['speed_prob']:
                factor = random.choice([0.8, 0.9, 1.1, 1.2])
                indices = np.linspace(0, len(frames)-1, int(len(frames)*factor)).astype(int)
                frames = [frames[i] for i in indices if i < len(frames)]
            if len(frames) > 10 and random.random() < self.aug_params['drop_prob']:
                for _ in range(random.randint(1, 3)):
                    del frames[random.randint(0, len(frames)-1)]
            return frames
        except Exception as e:
            logging.error(f"Temporal augmentation failed: {str(e)}")
            return frames

    def _apply_spatial_augmentations(self, frame):
        try:
            if random.random() < 0.5:
                frame = cv2.flip(frame, 1)
            if random.random() < self.aug_params['color_prob']:
                frame = self._apply_color_augmentation(frame)
            if random.random() < self.aug_params['geo_prob']:
                frame = self._apply_geometric_augmentation(frame)
            if random.random() < self.aug_params['noise_prob']:
                frame = self._apply_noise(frame)
            if random.random() < self.aug_params['blur_prob']:
                frame = self._apply_blur(frame)
            return frame
        except Exception as e:
            logging.error(f"Spatial augmentation failed: {str(e)}")
            return frame

    def augment_video(self, video_path, output_path):
        try:
            frames, fps = self._read_video_frames(video_path)
            if not frames:
                return False
            frames = self._temporal_augment(frames)
            if not frames:
                return False
            augmented = [self._apply_spatial_augmentations(f) for f in frames]
            return self._write_video(augmented, output_path, fps)
        except Exception as e:
            logging.error(f"Video augmentation failed for {video_path}: {str(e)}")
            return False

    def process_class(self, class_name):
        class_path = os.path.join(self.input_dir, class_name)
        out_path = os.path.join(self.output_dir, class_name)
        os.makedirs(out_path, exist_ok=True)
        vids = [f for f in os.listdir(class_path) if f.lower().endswith(('.mp4', '.avi', '.mov'))]
        if not vids:
            logging.warning(f"No videos in {class_path}")
            return
        logging.info(f"Processing class: {class_name}")
        for i, vid in enumerate(vids):
            shutil.copy2(os.path.join(class_path, vid), os.path.join(out_path, f"orig_{i:03d}.mp4"))
        need = max(0, self.target_count - len(vids))
        if need > 0:
            logging.info(f"Creating {need} augmented videos...")
            with ThreadPoolExecutor(max_workers=min(4, os.cpu_count() or 1)) as pool:
                tasks = []
                for i in range(need):
                    src = random.choice(vids)
                    src_path = os.path.join(class_path, src)
                    dst_path = os.path.join(out_path, f"aug_{i:03d}.mp4")
                    tasks.append(pool.submit(self.augment_video, src_path, dst_path))
                done = sum([t.result() for t in tqdm(tasks, desc=f"Augmenting {class_name}")])
                logging.info(f"Successfully created {done}/{need} augmented videos")

def validate_paths(input_dir, output_dir):
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input dir not found: {input_dir}")
    if not os.path.isdir(input_dir):
        raise NotADirectoryError(f"Not a directory: {input_dir}")
    os.makedirs(output_dir, exist_ok=True)

def main():
    input_dir = r"C:\Users\savem\OneDrive\Desktop\kannada_sign_project\dataset\Dataset\raw_videos"
    output_dir = r"C:\Users\savem\OneDrive\Desktop\kannada_sign_project\dataset\vedio_augmented_data"
    target_videos_per_class = 50
    try:
        validate_paths(input_dir, output_dir)
        class_dirs = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
        if not class_dirs:
            raise RuntimeError(f"No class folders found in {input_dir}")
        logging.info(f"Found {len(class_dirs)} classes to process")
        augmenter = VideoAugmenter(input_dir, output_dir, target_videos_per_class)
        for class_name in class_dirs:
            augmenter.process_class(class_name)
        logging.info("\n✅ Augmentation complete!")
        logging.info(f"✅ Videos saved to: {output_dir}")
    except Exception as e:
        logging.error(f"❌ Fatal error: {str(e)}", exc_info=True)
        return 1
    return 0

if __name__ == '__main__':
    exit(main())
