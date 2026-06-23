"""
Train a face mask detection classifier.

This script reads images from the dataset directory, converts them to
grayscale, resizes them to a fixed size, flattens the pixel values and
trains a Support Vector Machine (SVM) classifier to distinguish between
faces with masks and without masks. The trained model is saved to disk
using joblib.

Example usage:
    python train.py --data-dir dataset --model-out models/mask_detector.pkl
"""

import argparse
from pathlib import Path
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib


def load_images(data_dir: Path, img_size: tuple = (64, 64)) -> tuple:
    """
    Load images from the dataset directory and return flattened arrays and labels.

    Args:
        data_dir: Path to dataset with subdirectories 'with_mask' and 'without_mask'.
        img_size: Size to resize images to (width, height).

    Returns:
        X: numpy array of shape (n_samples, img_size[0]*img_size[1])
        y: numpy array of labels (1 for with_mask, 0 for without_mask)
    """
    X = []
    y = []
    for label, subdir in [(1, "with_mask"), (0, "without_mask")]:
        class_dir = data_dir / subdir
        for img_path in class_dir.glob("*.png"):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            resized = cv2.resize(img, img_size)
            X.append(resized.flatten())
            y.append(label)
    return np.array(X), np.array(y)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train face mask detection model")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--model-out", type=str, default="models/mask_detector.pkl", help="Output model file path")
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    model_out = Path(args.model_out)

    X, y = load_images(data_dir)
    # Normalize pixel values
    X = X / 255.0
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    # Train SVM classifier with RBF kernel
    clf = SVC(kernel="rbf", probability=True, gamma="auto")
    clf.fit(X_train, y_train)
    # Evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.3f}")
    print("Classification report:\n", classification_report(y_test, y_pred))
    # Save model
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, model_out)
    print(f"Model saved to {model_out}")


if __name__ == "__main__":
    main()