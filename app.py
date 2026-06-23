"""
FastAPI server for face mask detection.

This API exposes two routes:

* `GET /health` – simple health check.
* `POST /predict` – accepts an image file and returns whether a mask is detected.

The model expects 64x64 grayscale images flattened into a vector. Ensure
the model file (`models/mask_detector.pkl`) exists before starting the server.
"""

import os
from io import BytesIO
from typing import Dict

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import joblib


MODEL_PATH = os.getenv("MODEL_PATH", "models/mask_detector.pkl")

try:
    model = joblib.load(MODEL_PATH)
except Exception as exc:
    raise RuntimeError(f"Could not load model from {MODEL_PATH}: {exc}")

app = FastAPI(title="Face Mask Detection API")


def preprocess_image(file_bytes: bytes, img_size: tuple = (64, 64)) -> np.ndarray:
    """Convert uploaded image bytes to a flattened, normalized grayscale array."""
    # Load image from bytes
    np_arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Invalid image data")
    img_resized = cv2.resize(img, img_size)
    # Normalize pixel values and flatten
    return (img_resized.flatten() / 255.0).reshape(1, -1)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> JSONResponse:
    """Predict whether the uploaded image contains a masked face."""
    try:
        contents = await file.read()
        features = preprocess_image(contents)
        pred = model.predict(features)[0]
        label = "with_mask" if pred == 1 else "without_mask"
        prob = model.predict_proba(features)[0].max()
        return JSONResponse({"prediction": label, "confidence": float(prob)})
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))