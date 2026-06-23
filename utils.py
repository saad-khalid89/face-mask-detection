"""
Utility functions for the face mask detection project.

These helpers can be used for loading the dataset, preprocessing images and
visualising results. They are not used directly by `app.py`, but are
available for experimentation and further development.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List


def read_image(path: str, img_size: Tuple[int, int] = (64, 64)) -> np.ndarray:
    """Read an image from disk, convert to grayscale, resize and flatten."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    resized = cv2.resize(img, img_size)
    return resized.flatten() / 255.0


def load_dataset(data_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load the entire dataset into memory (features and labels)."""
    X: List[np.ndarray] = []
    y: List[int] = []
    for label, subdir in [(1, "with_mask"), (0, "without_mask")]:
        class_dir = data_dir / subdir
        for img_file in class_dir.glob("*.png"):
            X.append(read_image(str(img_file)))
            y.append(label)
    return np.array(X), np.array(y)