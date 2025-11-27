# Pseudo-3D Data Augmentation Tool

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)
![PyTorch](https://img.shields.io/badge/PyTorch-Compatible-orange)

A lightweight Python tool that generates **pseudo-3D rotating views** from 2D orthogonal images (Front, Side, Back, Left) using geometric transformations. 

This project implements a custom projection algorithm without using 3D engines (like Unity or Blender), optimized for **Data Augmentation** in building classification or urban scene understanding tasks.

##  Key Features
- **Pseudo-3D Projection:** Simulates 3D rotation using `cv2.getPerspectiveTransform` and trigonometric calculations.
- **Vertical Perspective:** Applies distance-based vanishing point effects (pin-hole camera model).
- **Dual Modes:**
  -  **Offline Generation:** Save thousands of augmented images to disk (`main.py`).
  -  **Online Pipeline:** Generate views on-the-fly during training (`augmentor.py`).

##  Installation

```bash
pip install opencv-python numpy
# (Optional) For ML pipeline integration

pip install torch torchvision
