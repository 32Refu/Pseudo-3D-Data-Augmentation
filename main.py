import os
import math
import cv2
import numpy as np
from typing import Optional, Tuple

def load_image(path: str, size: Tuple[int, int] = (360, 360)) -> Optional[np.ndarray]:
    """
    Load an image from the specified path and resize it.
    
    Args:
        path (str): Path to the image file.
        size (tuple): Target size (width, height).
        
    Returns:
        np.ndarray: Loaded image or None if loading fails.
    """
    if not os.path.exists(path):
        print(f"[Error] File not found: {path}")
        return None
        
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"[Error] Invalid image format: {path}")
        return None
        
    return cv2.resize(img, size)

def perspective_warp(img: np.ndarray, angle_deg: int, face_type: str = 'front') -> np.ndarray:
    """
    Apply horizontal perspective warp to simulate rotation.
    Based on simple trigonometric projection (pseudo-3D).
    """
    h, w = img.shape[:2]
    theta = math.radians(angle_deg)
    
    # Perspective strength factor (0.3 ~ 0.5 recommended)
    perspective_strength = 0.3

    # Calculate projected width based on rotation angle
    if face_type == 'front':
        proj_w = int(w * math.cos(theta))
        shrink_ratio = 1 - math.sin(theta) * perspective_strength
    else:  # side
        proj_w = int(w * math.sin(theta))
        shrink_ratio = 1 - math.cos(theta) * perspective_strength

    if proj_w <= 0:
        return np.zeros((h, 1, 3), dtype=np.uint8)

    src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

    # Define destination points for perspective transform
    if face_type == 'front':
        # Front face: anchored at left (x=0), shrinks towards right
        dst_pts = np.float32([
            [0, 0],
            [proj_w, h * (1 - shrink_ratio) / 2],
            [proj_w, h - h * (1 - shrink_ratio) / 2],
            [0, h]
        ])
    else:
        # Side face: anchored at right, shrinks towards left
        dst_pts = np.float32([
            [0, h * (1 - shrink_ratio) / 2],
            [proj_w, 0],
            [proj_w, h],
            [0, h - h * (1 - shrink_ratio) / 2]
        ])

    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    proj_w = max(1, proj_w) # Prevent width from being 0
    
    return cv2.warpPerspective(img, matrix, (proj_w, h))

def synthesize_view(front_img: np.ndarray, side_img: np.ndarray, angle_deg: int, output_h: int) -> np.ndarray:
    """
    Combine front and side views into a single canvas.
    """
    warped_front = perspective_warp(front_img, angle_deg, face_type='front')
    warped_side = perspective_warp(side_img, angle_deg, face_type='side')

    h_f, w_f = warped_front.shape[:2]
    h_s, w_s = warped_side.shape[:2]
    
    total_w = w_f + w_s
    if total_w == 0:
        return np.zeros((output_h, 1, 3), dtype=np.uint8)

    canvas = np.zeros((output_h, total_w, 3), dtype=np.uint8)

    # Center alignment offsets
    y_off_s = (output_h - h_s) // 2
    y_off_f = (output_h - h_f) // 2

    # Layout: [Side View (Left)] + [Front View (Right)]
    if w_s > 0:
        canvas[y_off_s:y_off_s+h_s, :w_s] = warped_side
    if w_f > 0:
        canvas[y_off_f:y_off_f+h_f, w_s:w_s+w_f] = warped_front

    return canvas

def apply_vertical_perspective(img: np.ndarray, distance: int) -> np.ndarray:
    """
    Apply vertical perspective distortion based on distance.
    Simulates the vanishing point effect (buildings look narrower at the top).
    """
    h, w = img.shape[:2]
    
    # Constant representing the relative height of the object
    OBJECT_HEIGHT_CONST = 2.5
    
    # Calculate shrink ratio based on distance (Pin-hole camera model approximation)
    top_width_ratio = distance / math.sqrt(distance**2 + OBJECT_HEIGHT_CONST**2)
    shrink_amount = (1.0 - top_width_ratio) * 0.5
    
    # Limit maximum distortion
    shrink_amount = min(shrink_amount, 0.45)

    src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    
    # Pull top corners towards the center
    dst_tl_x = int(w * shrink_amount)
    dst_tr_x = int(w - w * shrink_amount)

    dst_pts = np.float32([
        [dst_tl_x, 0],   # Top-Left
        [dst_tr_x, 0],   # Top-Right
        [w, h],          # Bottom-Right
        [0, h]           # Bottom-Left
    ])

    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return cv2.warpPerspective(img, matrix, (w, h))

def generate_dataset(images: list, output_folder: str):
    """
    Main loop to generate multi-view dataset.
    Generates 360 views for multiple distances.
    """
    front, side, back, left = images
    input_h = front.shape[0]
    
    # Distance factors (smaller = closer/more distorted, larger = farther/flatter)
    distances = [1, 2, 5, 10, 20, 50, 100, 500]
    total_images = 360 * len(distances)

    print(f"[*] Starting generation: {total_images} images (Height: {input_h}px)")
    os.makedirs(output_folder, exist_ok=True)

    count = 0
    for angle in range(360):
        # Determine visible faces based on angle
        if 0 <= angle < 90:
            face_r, face_l = front, side
            local_angle = angle
        elif 90 <= angle < 180:
            face_r, face_l = side, back
            local_angle = angle - 90
        elif 180 <= angle < 270:
            face_r, face_l = back, left
            local_angle = angle - 180
        else:
            face_r, face_l = left, front
            local_angle = angle - 270

        # 1. Horizontal Rotation
        horizontal_view = synthesize_view(face_r, face_l, local_angle, output_h=input_h)

        # 2. Vertical Perspective (Distance variation)
        for d in distances:
            final_view = apply_vertical_perspective(horizontal_view, distance=d)
            
            filename = f"view_a{angle:03d}_d{d:03d}.png"
            save_path = os.path.join(output_folder, filename)
            cv2.imwrite(save_path, final_view)
            count += 1
            
        if angle % 45 == 0:
            print(f" >> Progress: {angle}/360 degrees processed...")
    
    print(f"[Done] All images saved to '{output_folder}'")

if __name__ == "__main__":
    # --- Configuration ---
    # Prepare 4 orthogonal view images
    INPUT_FILES = {
        'front': "front.png",
        'side':  "right.png",
        'back':  "back.png",
        'left':  "left.png"
    }
    OUTPUT_DIR = "generated_views"
    TARGET_SIZE = (720, 720)

    # Load Images
    loaded_imgs = []
    try:
        f_img = load_image(INPUT_FILES['front'], TARGET_SIZE)
        s_img = load_image(INPUT_FILES['side'], TARGET_SIZE)
        b_img = load_image(INPUT_FILES['back'], TARGET_SIZE)
        l_img = load_image(INPUT_FILES['left'], TARGET_SIZE)
        
        # Check if all images are loaded successfully
        if not all([f_img is not None, s_img is not None, b_img is not None, l_img is not None]):
            raise FileNotFoundError("One or more input images could not be loaded.")
            
        loaded_imgs = [f_img, s_img, b_img, l_img]
        
        # Run Generator
        generate_dataset(loaded_imgs, OUTPUT_DIR)
        
    except Exception as e:
        print(f"[Critical Error] {e}")