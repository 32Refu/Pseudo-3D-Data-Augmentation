import os
import math
import cv2
import numpy as np
from typing import Optional, Tuple

"""
[Basic Version]
This script generates only horizontal pseudo-3D rotation views.
It does NOT apply vertical perspective (vanishing point) effects.
"""

def load_image(path: str, size: Tuple[int, int] = (360, 360)) -> Optional[np.ndarray]:
    if not os.path.exists(path):
        print(f"[Error] File not found: {path}")
        return None
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None: return None
    return cv2.resize(img, size)

def perspective_warp(img: np.ndarray, angle_deg: int, face_type: str = 'front') -> np.ndarray:
    h, w = img.shape[:2]
    theta = math.radians(angle_deg)
    perspective_strength = 0.3 # Adjustable

    if face_type == 'front':
        proj_w = int(w * math.cos(theta))
        shrink_ratio = 1 - math.sin(theta) * perspective_strength
    else:
        proj_w = int(w * math.sin(theta))
        shrink_ratio = 1 - math.cos(theta) * perspective_strength

    if proj_w <= 0: return np.zeros((h, 1, 3), dtype=np.uint8)

    src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

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

    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    proj_w = max(1, proj_w)
    return cv2.warpPerspective(img, matrix, (proj_w, h))

def synthesize_view(front_img, side_img, angle_deg, output_h):
    warped_front = perspective_warp(front_img, angle_deg, 'front')
    warped_side = perspective_warp(side_img, angle_deg, 'side')
    
    h_s, w_s = warped_side.shape[:2]
    h_f, w_f = warped_front.shape[:2]
    
    canvas = np.zeros((output_h, w_f + w_s, 3), dtype=np.uint8)
    
    # Simple Center Alignment
    y_off_s = (output_h - h_s) // 2
    y_off_f = (output_h - h_f) // 2

    if w_s > 0: canvas[y_off_s:y_off_s+h_s, :w_s] = warped_side
    if w_f > 0: canvas[y_off_f:y_off_f+h_f, w_s:w_s+w_f] = warped_front
    
    return canvas

if __name__ == "__main__":
    # Settings
    INPUT_FILES = {'front': "front.png", 'side': "right.png", 'back': "back.png", 'left': "left.png"}
    OUTPUT_DIR = "generated_views_basic" # Creates a different folder
    TARGET_SIZE = (720, 720)

    # Load & Run
    imgs = [load_image(INPUT_FILES[k], TARGET_SIZE) for k in ['front', 'side', 'back', 'left']]
    if not all(x is not None for x in imgs):
        print("[Error] Check input images.")
    else:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        print(f"[*] Generating 360 basic views to '{OUTPUT_DIR}'...")
        
        for angle in range(360):
            if 0 <= angle < 90: r, l, a = imgs[0], imgs[1], angle
            elif 90 <= angle < 180: r, l, a = imgs[1], imgs[2], angle-90
            elif 180 <= angle < 270: r, l, a = imgs[2], imgs[3], angle-180
            else: r, l, a = imgs[3], imgs[0], angle-270
            
            res = synthesize_view(r, l, a, TARGET_SIZE[0])
            cv2.imwrite(f"{OUTPUT_DIR}/view_{angle:03d}.png", res)
            
        print("[Done]")