#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
IMAR Inference Pipeline — Trinder-1 Single-Task Version
==========================================================
Background subtraction (cv2.subtract + bitwise_not) -> YOLO droplet detection
-> ROI extraction -> CNN concentration regression

[Usage]
    1. Edit the "===== PARAMETERS =====" block below
    2. Run this file directly in your IDE (no CLI arguments needed)
    3. Results are printed to console; images saved to uploads/<timestamp>/
"""

import os
import cv2
import json
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO
from scipy.ndimage import zoom

# ============================================================================
# ===== PARAMETERS ===========================================================
# ============================================================================
# --- Paths ---
ORIGINAL_PATH  = r"D:\APPs\Pycharm\Python Projects\Drop_detection_py\trinder_1\1.jpg"
BACKGROUND_PATH = r"D:\APPs\Pycharm\Python Projects\Drop_detection_py\trinder_1\bg1.jpg"
MODEL_PATH      = r"D:\APPs\Pycharm\Python Projects\Drop_detection_py\trinder_1\results_reg_cnn\best_model.pth"
YOLO_PATH       = r"D:\APPs\Pycharm\Python Projects\Drop_detection_py\best2.pt"

# --- Inference ---
MODEL_NAME     = "trinder_1"   # model name (only trinder_1 is supported)
YOLO_CONF      = 0.6           # YOLO confidence threshold (0-1)
YOLO_IMGSZ     = 1024          # YOLO input image size (px)
YOLO_SAVE      = False         # whether to let YOLO save its own prediction images
BOX_Y_MIN      = 200           # valid droplet Y lower bound (px)
BOX_Y_MAX      = 1200          # valid droplet Y upper bound (px)
BOX_X_MIN      = 200           # valid droplet X lower bound (px)
BOX_X_MAX      = 1900          # valid droplet X upper bound (px)
ROI_HALF_WIDTH = 20            # ROI half-width (px), cropped to 40x40

# --- Output ---
SAVE_RESULT    = True          # save background-subtraction result image (result.jpg)
SAVE_ROI       = True          # save individual droplet ROI images (roi_*.jpg)
# ============================================================================

# ---------------------------------------------------------------------------
# Resolve paths: base dir = script location (overridable via env var)
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_BASE_DIR   = os.environ.get("IMAR_BASE_DIR", _SCRIPT_DIR)


def _resolve(rel: str) -> str:
    return os.path.join(_BASE_DIR, rel)


# ---------------------------------------------------------------------------
# CNN regression model (must match the training script CNNModel exactly)
# ---------------------------------------------------------------------------
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1    = nn.Conv2d(3, 64,   kernel_size=3, padding=1)
        self.conv2    = nn.Conv2d(64, 128,  kernel_size=3, padding=1)
        self.conv3    = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool     = nn.MaxPool2d(2, 2)
        self.dropout  = nn.Dropout(0.25)
        self.fc1      = nn.Linear(256 * 5 * 5, 512)
        self.fc2      = nn.Linear(512, 1)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 256 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# ---------------------------------------------------------------------------
# Model path mapping
# ---------------------------------------------------------------------------
MODEL_MAPPING = {
    "trinder_1": "trinder_1/results_reg_cnn/best_model.pth",
}

# Concentration conversion mapping (raw network output -> final concentration)
CONCENTRATION_MAPPING = {
    "trinder_1": {
        "formula":    lambda x: x,
        "unit":       "mmol/L",
        "description": "Glucose Concentration (Trinder-1)"
    },
}


# ---------------------------------------------------------------------------
# Model loading (cached after first load)
# ---------------------------------------------------------------------------
_model_cache: dict = {}
_yolo_model         = None


def load_yolo_model():
    global _yolo_model
    if _yolo_model is None:
        if not os.path.exists(YOLO_PATH):
            raise FileNotFoundError(f"YOLO model not found: {YOLO_PATH}")
        _yolo_model = YOLO(YOLO_PATH)
    return _yolo_model


def load_cnn_model(model_name: str = "trinder_1") -> torch.nn.Module:
    if model_name in _model_cache:
        return _model_cache[model_name]

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = CNNModel().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    _model_cache[model_name] = model
    return model


# ---------------------------------------------------------------------------
# Image preprocessing / prediction
# ---------------------------------------------------------------------------
_transforms = transforms.Compose([
    transforms.Resize((40, 40)),
    transforms.ToTensor(),
])


def predict_roi(model, roi_path: str):
    """
    Run CNN inference on a single ROI image file.
    Returns the raw network output value (not yet offset-adjusted).
    """
    try:
        img    = Image.open(roi_path).convert("RGB")
        tensor = _transforms(img).unsqueeze(0).to(next(model.parameters()).device)
        with torch.no_grad():
            return model(tensor).item()
    except Exception as e:
        print(f"[ERROR] predict_roi: {e}")
        return None


# ---------------------------------------------------------------------------
# Core inference pipeline
# ---------------------------------------------------------------------------
def process_images(original_path: str, background_path: str,
                   model_name: str = "trinder_1") -> dict:
    """
    Full inference pipeline: background subtraction -> YOLO detection
    -> ROI extraction -> CNN regression.

    Parameters
    ----------
    original_path   : str, path to the sample image
    background_path : str, path to the background reference image (captured
                      without sample under identical conditions)
    model_name      : str, model name (default "trinder_1")

    Returns
    -------
    dict — JSON-compatible result dictionary
    """
    try:
        # ---- 1. Load and resize images ------------------------------------
        original   = cv2.imread(original_path)
        background = cv2.imread(background_path)
        if original is None or background is None:
            raise ValueError("Failed to read image file(s). Check the paths.")

        if background.shape[:2] != original.shape[:2]:
            background = cv2.resize(background, (original.shape[1], original.shape[0]))

        # ---- 2. Background subtraction (bitwise_not makes droplets bright) --
        mask = cv2.subtract(background, original)
        mask = cv2.bitwise_not(mask)

        # Save background-subtraction result
        timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
        upload_dir   = _resolve(os.path.join("uploads", timestamp))
        os.makedirs(upload_dir, exist_ok=True)
        result_path  = os.path.join(upload_dir, "result.jpg")
        if SAVE_RESULT:
            cv2.imwrite(result_path, mask)

        # ---- 3. YOLO droplet detection ------------------------------------
        yolo = load_yolo_model()
        res  = yolo.predict(original_path, conf=YOLO_CONF, imgsz=YOLO_IMGSZ,
                            save=YOLO_SAVE)[0]

        xywh        = res.boxes.xywh.cpu().numpy()
        xyxy        = res.boxes.xyxy.cpu().numpy()
        confidences = res.boxes.conf.cpu().numpy()

        # Sort detections by position (row first, then column)
        sorted_boxes = sorted(enumerate(xywh),
                              key=lambda x: (int(x[1][1] / 100), int(x[1][0] / 100)))

        detections       = []
        roi_predictions  = []

        # ---- 4. ROI extraction and CNN prediction -------------------------
        if sorted_boxes:
            cnn_model = load_cnn_model(model_name)
            k = 0
            for idx, (j, box) in enumerate(sorted_boxes):
                # Position filter: only process boxes within the valid imaging region
                if not (BOX_Y_MIN < box[1] < BOX_Y_MAX and
                        BOX_X_MIN < box[0] < BOX_X_MAX):
                    continue
                k += 1

                cx = (xyxy[j][0] + xyxy[j][2]) / 2
                cy = (xyxy[j][1] + xyxy[j][3]) / 2
                x1 = max(0,                 int(cx - ROI_HALF_WIDTH))
                y1 = max(0,                 int(cy - ROI_HALF_WIDTH))
                x0 = min(original.shape[1], int(cx + ROI_HALF_WIDTH))
                y0 = min(original.shape[0], int(cy + ROI_HALF_WIDTH))

                roi = mask[y1:y0, x1:x0]
                if roi.size == 0:
                    continue

                # Resize ROI to 20x20 before feeding to CNN
                roi = zoom(roi, (20 / ROI_HALF_WIDTH, 20 / ROI_HALF_WIDTH, 1),
                           order=1).astype(np.uint8)

                roi_path = os.path.join(upload_dir, f"roi_{k}.jpg")
                if SAVE_ROI:
                    cv2.imwrite(roi_path, roi)

                raw_val = predict_roi(cnn_model, roi_path)
                if raw_val is not None:
                    roi_predictions.append(raw_val)

                detections.append({
                    "bbox":           [float(v) for v in xyxy[j]],
                    "confidence":      float(confidences[j]),
                    "class_name":     f"droplet_{k}",
                    "raw_prediction":  raw_val,
                    "roi_path":        roi_path,
                })

        # ---- 5. Aggregate statistics ----------------------------------------
        if roi_predictions:
            raw_mean = float(np.mean(roi_predictions))
            raw_std  = float(np.std(roi_predictions))
        else:
            raw_mean = None
            raw_std  = None

        conc_info = {}
        if raw_mean is not None:
            mapping = CONCENTRATION_MAPPING.get(model_name, {})
            formula = mapping.get("formula", lambda x: x)
            conc_info = {
                "raw_mean":           raw_mean,
                "final_value":        float(formula(raw_mean)),
                "std":                raw_std,
                "unit":               mapping.get("unit", ""),
                "description":        mapping.get("description", ""),
                "individual_values":  [float(v) for v in roi_predictions],
                "num_droplets":       len(roi_predictions),
            }

        return {
            "success":            True,
            "timestamp":          timestamp,
            "result_image_path":  result_path,
            "model":              model_name,
            "num_detections":     len(detections),
            "detections":         detections,
            "concentration":      conc_info,
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


# ---------------------------------------------------------------------------
# Public entry point (used by server.py FastAPI)
# ---------------------------------------------------------------------------
def main_process(original_path: str, background_path: str,
                model_name: str = "trinder_1") -> dict:
    return process_images(original_path, background_path, model_name)


# ---------------------------------------------------------------------------
# CLI entry point (IDE direct-run mode)
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    ORIGINAL_PATH  = _resolve("demo_data/original.jpg")
    BACKGROUND_PATH = _resolve("demo_data/background.jpg")

    # Validate all required files
    for path, label in [(ORIGINAL_PATH, "Sample image"),
                        (BACKGROUND_PATH, "Background image"),
                        (MODEL_PATH, "Model"),
                        (YOLO_PATH, "YOLO")]:
        if not os.path.exists(path):
            print(f"[ERROR] File not found: {label}\n  {path}")
            exit(1)

    print(f"[INFO] Sample image:    {ORIGINAL_PATH}")
    print(f"[INFO] Background image: {BACKGROUND_PATH}")
    print(f"[INFO] Model:           {MODEL_PATH}")
    print(f"[INFO] YOLO:            {YOLO_PATH}")

    result = process_images(ORIGINAL_PATH, BACKGROUND_PATH, MODEL_NAME)
    print(json.dumps(result, indent=2, ensure_ascii=False))