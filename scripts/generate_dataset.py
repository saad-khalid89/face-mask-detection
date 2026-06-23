"""
Generate a synthetic face mask detection dataset.

This script creates two subdirectories within the specified output
directory: `with_mask` and `without_mask`. Each directory will contain
`num_samples` PNG images of simple cartoon faces drawn with OpenCV.
Faces in the `with_mask` class have a coloured rectangle over the lower
half of the face to simulate a mask, while those in the `without_mask`
class have a mouth drawn instead.
"""

import os
import argparse
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple


def draw_face(mask: bool, img_size: Tuple[int, int] = (64, 64)) -> np.ndarray:
    """Draw a simple face with or without a mask on a blank image."""
    w, h = img_size
    img = np.ones((h, w, 3), dtype=np.uint8) * 255  # white background
    # Draw face oval
    face_color = (224, 172, 105)  # light skin tone
    cv2.ellipse(img, (w // 2, h // 2), (int(w * 0.3), int(h * 0.4)), 0, 0, 360, face_color, -1)
    # Draw eyes
    eye_radius = 4
    eye_y = int(h * 0.35)
    eye_x_offset = int(w * 0.12)
    cv2.circle(img, (w // 2 - eye_x_offset, eye_y), eye_radius, (0, 0, 0), -1)
    cv2.circle(img, (w // 2 + eye_x_offset, eye_y), eye_radius, (0, 0, 0), -1)
    # Draw mask or mouth
    if mask:
        mask_color = random_color()
        top_left = (int(w * 0.2), int(h * 0.5))
        bottom_right = (int(w * 0.8), int(h * 0.7))
        cv2.rectangle(img, top_left, bottom_right, mask_color, -1)
    else:
        # Mouth (ellipse)
        mouth_center = (w // 2, int(h * 0.65))
        mouth_axes = (int(w * 0.15), int(h * 0.07))
        cv2.ellipse(img, mouth_center, mouth_axes, 0, 0, 180, (0, 0, 255), 2)
    return img


def random_color() -> Tuple[int, int, int]:
    """Generate a random bright colour for mask."""
    return tuple(np.random.randint(50, 200) for _ in range(3))


def generate_images(out_dir: Path, num_samples: int) -> None:
    np.random.seed(42)
    # Create class directories
    with_mask_dir = out_dir / "with_mask"
    without_mask_dir = out_dir / "without_mask"
    with_mask_dir.mkdir(parents=True, exist_ok=True)
    without_mask_dir.mkdir(parents=True, exist_ok=True)
    # Generate images
    for i in range(num_samples):
        img_mask = draw_face(mask=True)
        img_nomask = draw_face(mask=False)
        cv2.imwrite(str(with_mask_dir / f"mask_{i:04d}.png"), img_mask)
        cv2.imwrite(str(without_mask_dir / f"nomask_{i:04d}.png"), img_nomask)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic face mask dataset")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save generated images")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of images per class")
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    generate_images(out_dir, args.num_samples)
    print(f"Generated {args.num_samples * 2} images into {out_dir}")


if __name__ == "__main__":
    main()