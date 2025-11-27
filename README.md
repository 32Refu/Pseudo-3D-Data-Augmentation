# Pseudo-3D Data Augmentation Tool

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)
![PyTorch](https://img.shields.io/badge/PyTorch-Compatible-orange)

A lightweight Python tool that generates **pseudo-3D rotating views** from 2D orthogonal images (Front, Side, Back, Left) using geometric transformations. 

This project implements a custom projection algorithm optimized for **Data Augmentation** in building classification tasks.

## ⚡ Quick Comparison (Script Modes)
This repository provides two modes of generation scripts.

| Script | Feature | Output Characteristics |
| :--- | :--- | :--- |
| **`basic_rotation.py`** | Horizontal Rotation Only | Simulates a camera rotating around the object at a fixed height. Best for simple view synthesis. |
| **`main.py`** | **Rotation + Vertical Perspective** | Applies **distance-based vanishing point effects**. Simulates realistic camera physics where building tops appear narrower from a distance. |

## 🛠️ Installation

```bash
pip install opencv-python numpy
# (Optional) For ML pipeline integration
pip install torch torchvision
