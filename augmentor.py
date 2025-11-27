import math
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class Pseudo3DAugmentor:
    """
    On-the-fly 3D View Generator for ML Pipelines.
    This class handles the geometric transformation logic in memory.
    """
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size
        self.object_height_const = 2.5
        self.perspective_strength = 0.3

    def _perspective_warp(self, img: np.ndarray, angle_deg: float, face_type: str) -> np.ndarray:
        h, w = img.shape[:2]
        theta = math.radians(angle_deg)

        if face_type == 'front':
            proj_w = int(w * math.cos(theta))
            shrink_ratio = 1 - math.sin(theta) * self.perspective_strength
        else:
            proj_w = int(w * math.sin(theta))
            shrink_ratio = 1 - math.cos(theta) * self.perspective_strength

        if proj_w <= 0: return np.zeros((h, 1, 3), dtype=np.uint8)

        # Coordinate calculation (Optimized)
        if face_type == 'front':
            dst_pts = np.float32([
                [0, 0], [proj_w, h * (1 - shrink_ratio) / 2],
                [proj_w, h - h * (1 - shrink_ratio) / 2], [0, h]
            ])
        else:
            dst_pts = np.float32([
                [0, h * (1 - shrink_ratio) / 2], [proj_w, 0],
                [proj_w, h], [0, h - h * (1 - shrink_ratio) / 2]
            ])

        src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        proj_w = max(1, proj_w)
        return cv2.warpPerspective(img, matrix, (proj_w, h))

    def _apply_vertical_perspective(self, img: np.ndarray, distance: float) -> np.ndarray:
        h, w = img.shape[:2]
        top_width_ratio = distance / math.sqrt(distance**2 + self.object_height_const**2)
        shrink_amount = min((1.0 - top_width_ratio) * 0.5, 0.45)

        src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst_pts = np.float32([
            [int(w * shrink_amount), 0], 
            [int(w - w * shrink_amount), 0],
            [w, h], [0, h]
        ])
        
        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        return cv2.warpPerspective(img, matrix, (w, h))

    def get_random_view(self, images: list) -> np.ndarray:
        """
        Generates a single random view from the 4 source images.
        Returns: numpy array (H, W, C) ready for ToTensor()
        """
        front, side, back, left = images
        
        # 1. Random Parameters
        angle = np.random.randint(0, 360)
        distance = np.random.uniform(1.0, 50.0) # Random continuous distance
        
        # 2. Face Selection Logic
        if 0 <= angle < 90:
            r, l, local_a = front, side, angle
        elif 90 <= angle < 180:
            r, l, local_a = side, back, angle - 90
        elif 180 <= angle < 270:
            r, l, local_a = back, left, angle - 180
        else:
            r, l, local_a = left, front, angle - 270

        # 3. Pipeline Processing
        # (Warping logic integrated here)
        warped_r = self._perspective_warp(r, local_a, 'front')
        warped_l = self._perspective_warp(l, local_a, 'side')
        
        # Canvas Composition
        h_canvas = r.shape[0] # Assume all inputs have same height
        total_w = warped_r.shape[1] + warped_l.shape[1]
        canvas = np.zeros((h_canvas, total_w, 3), dtype=np.uint8)
        
        if warped_l.shape[1] > 0:
            y_off = (h_canvas - warped_l.shape[0]) // 2
            canvas[y_off:y_off+warped_l.shape[0], :warped_l.shape[1]] = warped_l
            
        if warped_r.shape[1] > 0:
            y_off = (h_canvas - warped_r.shape[0]) // 2
            x_off = warped_l.shape[1]
            canvas[y_off:y_off+warped_r.shape[0], x_off:x_off+warped_r.shape[1]] = warped_r

        # 4. Vertical Distortion & Resize
        final_img = self._apply_vertical_perspective(canvas, distance)
        final_img = cv2.resize(final_img, self.target_size)
        
        return final_img

# =========================================================
# PyTorch Dataset Integration Example
# =========================================================
class Pseudo3DBuildingDataset(Dataset):
    def __init__(self, source_images, transform=None, epoch_len=1000):
        """
        Args:
            source_images (list): [front, side, back, left] loaded numpy arrays.
            transform (callable): PyTorch transforms (ToTensor, Normalize, etc.)
            epoch_len (int): Virtual length of dataset (how many variations to see per epoch).
        """
        self.augmentor = Pseudo3DAugmentor(target_size=(224, 224))
        self.images = source_images
        self.transform = transform
        self.epoch_len = epoch_len

    def __len__(self):
        return self.epoch_len

    def __getitem__(self, idx):
        # Generate a NEW random view every time this is called
        aug_img = self.augmentor.get_random_view(self.images)
        
        # Convert BGR to RGB (OpenCV is BGR, PyTorch needs RGB)
        aug_img = cv2.cvtColor(aug_img, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            return self.transform(aug_img)
        
        return aug_img