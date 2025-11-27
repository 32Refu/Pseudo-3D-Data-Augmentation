# Pseudo-3D Data Augmentation Tool

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)
![PyTorch](https://img.shields.io/badge/PyTorch-Compatible-orange)

A lightweight Python tool that generates **pseudo-3D rotating views** from 2D orthogonal images (Front, Side, Back, Left) using geometric transformations.

This project implements a custom projection algorithm without using 3D engines (like Unity or Blender), optimized for **Data Augmentation** in building classification or urban scene understanding tasks.

## ⚡ Quick Comparison (Script Modes)
This repository provides two modes of generation scripts.

| Script              | Feature                       | Output Characteristics |
|---------------------|-------------------------------|-------------------------|
| `basic_rotation.py` | Horizontal Rotation Only      | Simulates a camera rotating around the object at a fixed height. Best for simple view synthesis. |
| `main.py`           | Rotation + Vertical Perspective | Applies distance-based vanishing point effects. Simulates realistic camera physics where building tops appear narrower from a distance. |

## 🚀 Usage

### 1. Basic Generation (Simple Rotation)
Generates 360 views without vertical distortion.
```bash
python basic_rotation.py
# Output: 'generated_views_basic/'
```

### 2. Advanced Generation (Full Physics)
Generates a full dataset (360 degrees × 8 distances) with vertical perspective.
```bash
python main.py
# Output: 'generated_views/' (Contains 2,880 images)
```

### 3. Integration with PyTorch (Online Augmentation)
Use Pseudo3DAugmentor directly in your training loop to generate infinite unique views without storage overhead.
```python
from augmentor import Pseudo3DBuildingDataset
from torch.utils.data import DataLoader

# Initialize Dataset with your images
dataset = Pseudo3DBuildingDataset(
    source_images=[front_img, side_img, back_img, left_img],
    epoch_len=5000  # Virtual dataset length per epoch
)

loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Training Loop
for batch in loader:
    # batch shape: [32, 3, 224, 224] (Ready for Model)
    pass
```

## 🛠️ Installation
```bash
pip install opencv-python numpy
# (Optional) For ML pipeline integration
pip install torch torchvision
```

## 📂 Project Structure
```
.
├── main.py             # [Advanced] Generates views with vertical perspective (8 distances)
├── basic_rotation.py   # [Basic] Generates simple rotating views (No vertical distortion)
├── augmentor.py        # Online augmentation class for ML pipelines
├── requirements.txt    # Dependencies
├── README.md           # Documentation
└── sample_images/      # Place your input images here (front, right, back, left)
```

## ⚠️ Limitations (Constraints)
- **Cuboid Projection Only**: The algorithm is designed based on cuboid geometry. It works best for buildings, boxes, buses, or containers.
- **Planar Assumption**: It assumes the input images are orthogonal flat surfaces. Cylindrical or complex shapes may look distorted.

## 📝 Note
This project was implemented from scratch for educational purposes to study geometric transformation algorithms and data augmentation pipelines.
