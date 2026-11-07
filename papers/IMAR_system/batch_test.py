#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Batch Test Script — Trinder-1 8-Group Regression Analysis
===========================================================
Runs inference on 8 groups of test images (original + background),
compares true concentrations (1-8 mmol/L) with predicted values
(raw_mean + 1, since training labels are 0-7), and generates
a regression plot.

Outputs (saved to ./batch_results/):
    batch_results/
        batch_test_<timestamp>.png   -- Regression plot
        batch_test_<timestamp>.csv   -- Raw data table
        batch_test_<timestamp>.json  -- Full run record
"""

import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy import stats

# Set matplotlib font for Chinese Windows (fallback to DejaVu if unavailable)
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

# ============================================================================
# [PARAMETERS — edit here]
# ============================================================================
ORIGINAL_DIR   = r"D:\APPs\Pycharm\Python Projects\Drop_detection_py\trinder_1"
BACKGROUND_DIR = r"D:\APPs\Pycharm\Python Projects\Drop_detection_py\trinder_1"
MODEL_PATH      = r"D:\APPs\Pycharm\Python Projects\Drop_detection_py\trinder_1\results_reg_cnn\best_model.pth"
YOLO_PATH       = r"D:\APPs\Pycharm\Python Projects\Drop_detection_py\best2.pt"

# True concentrations (mmol/L) for each group, 1-indexed
TRUE_CONCENTRATIONS = [1, 2, 3, 4, 5, 6, 7, 8]

# Inference parameters (must match infer.py)
YOLO_CONF       = 0.6
YOLO_IMGSZ      = 1024
BOX_Y_MIN, BOX_Y_MAX = 200, 1200
BOX_X_MIN, BOX_X_MAX = 200, 1900
ROI_HALF_WIDTH  = 20
# ============================================================================

import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from datetime import datetime
from ultralytics import YOLO
from scipy.ndimage import zoom

# ---------------------------------------------------------------------------
# Output folder
# ---------------------------------------------------------------------------
_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_OUTPUT_DIR  = os.path.join(_SCRIPT_DIR, "batch_results")
os.makedirs(_OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# CNN Model (must match the training architecture)
# ---------------------------------------------------------------------------
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1   = nn.Conv2d(3, 64,  kernel_size=3, padding=1)
        self.conv2   = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3   = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool    = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.fc1     = nn.Linear(256 * 5 * 5, 512)
        self.fc2     = nn.Linear(512, 1)

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
# Model loading (cached)
# ---------------------------------------------------------------------------
_yolo_model = None
_cnn_model  = None
_transforms = transforms.Compose([
    transforms.Resize((40, 40)),
    transforms.ToTensor(),
])


def load_yolo():
    global _yolo_model
    if _yolo_model is None:
        _yolo_model = YOLO(YOLO_PATH)
    return _yolo_model


def load_cnn():
    global _cnn_model
    if _cnn_model is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model  = CNNModel().to(device)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        _cnn_model = model
    return _cnn_model


def predict_roi(cnn_model, roi_np: np.ndarray) -> float:
    """Run CNN prediction on a single ROI array; returns raw network output."""
    roi_pil = Image.fromarray(cv2.cvtColor(roi_np, cv2.COLOR_BGR2RGB))
    tensor  = _transforms(roi_pil).unsqueeze(0).to(next(cnn_model.parameters()).device)
    with torch.no_grad():
        return cnn_model(tensor).item()


# ---------------------------------------------------------------------------
# Single image inference
# ---------------------------------------------------------------------------
def infer_one(idx: int) -> dict:
    """
    Run full inference on group idx (1-indexed).
    Returns dict with keys:
        num_droplets, raw_mean, raw_std, predicted_conc, individual_values
    """
    original_path   = os.path.join(ORIGINAL_DIR, f"{idx}.jpg")
    background_path = os.path.join(BACKGROUND_DIR, f"bg{idx}.jpg")

    original   = cv2.imread(original_path)
    background = cv2.imread(background_path)

    if original is None or background is None:
        raise FileNotFoundError(f"Failed to read image: {original_path} or {background_path}")

    if background.shape[:2] != original.shape[:2]:
        background = cv2.resize(background, (original.shape[1], original.shape[0]))

    # Background subtraction + bitwise_not to brighten droplet region
    mask = cv2.bitwise_not(cv2.subtract(background, original))

    # YOLO droplet detection
    yolo = load_yolo()
    res  = yolo.predict(original_path, conf=YOLO_CONF, imgsz=YOLO_IMGSZ, save=False)[0]

    xywh = res.boxes.xywh.cpu().numpy()
    xyxy = res.boxes.xyxy.cpu().numpy()

    # Sort by position (row first, then column)
    sorted_boxes = sorted(enumerate(xywh), key=lambda x: (int(x[1][1] / 100), int(x[1][0] / 100)))

    if not sorted_boxes:
        return {"num_droplets": 0, "raw_mean": None, "raw_std": None,
                "predicted_conc": None, "individual_values": []}

    cnn_model = load_cnn()
    predictions = []

    for j, box in sorted_boxes:
        # Filter out boxes outside the valid imaging region
        if not (BOX_Y_MIN < box[1] < BOX_Y_MAX and BOX_X_MIN < box[0] < BOX_X_MAX):
            continue

        cx = (xyxy[j][0] + xyxy[j][2]) / 2
        cy = (xyxy[j][1] + xyxy[j][3]) / 2
        x1 = max(0,                int(cx - ROI_HALF_WIDTH))
        y1 = max(0,                int(cy - ROI_HALF_WIDTH))
        x0 = min(original.shape[1], int(cx + ROI_HALF_WIDTH))
        y0 = min(original.shape[0], int(cy + ROI_HALF_WIDTH))
        roi = mask[y1:y0, x1:x0]
        if roi.size == 0:
            continue
        # Resize ROI to 20x20 before feeding to CNN
        roi = zoom(roi, (20 / ROI_HALF_WIDTH, 20 / ROI_HALF_WIDTH, 1), order=1).astype(np.uint8)
        predictions.append(predict_roi(cnn_model, roi))

    if not predictions:
        return {"num_droplets": 0, "raw_mean": None, "raw_std": None,
                "predicted_conc": None, "individual_values": []}

    raw_mean   = float(np.mean(predictions))
    raw_std    = float(np.std(predictions))
    # Training labels are 0-7, corresponding to true 1-8 mmol/L, so predicted = raw + 1
    predicted  = raw_mean + 1

    return {
        "num_droplets":      len(predictions),
        "raw_mean":          raw_mean,
        "raw_std":           raw_std,
        "predicted_conc":    predicted,
        "individual_values": [float(v) for v in predictions],
    }


# ---------------------------------------------------------------------------
# Run all groups
# ---------------------------------------------------------------------------
def run_batch():
    print("=" * 60)
    print("Batch Test — Trinder-1 8 Groups")
    print("=" * 60)

    true_vals      = []
    predicted_vals = []
    results        = []
    start_time     = time.time()

    for i in range(1, 9):
        print(f"[Group {i}/8] True = {TRUE_CONCENTRATIONS[i-1]} mmol/L", end=" ... ", flush=True)
        t0   = time.time()
        r    = infer_one(i)
        dt   = time.time() - t0

        if r["predicted_conc"] is not None:
            true_vals.append(TRUE_CONCENTRATIONS[i - 1])
            predicted_vals.append(r["predicted_conc"])
            print(f"{r['num_droplets']:2d} droplets detected | "
                  f"Predicted = {r['predicted_conc']:.3f} mmol/L | {dt:.1f}s")
        else:
            print("WARNING: no valid detections")

        results.append({
            "group":            i,
            "true_conc":        TRUE_CONCENTRATIONS[i - 1],
            "num_droplets":     r["num_droplets"],
            "raw_mean":         r["raw_mean"],
            "raw_std":          r["raw_std"],
            "predicted_conc":  r["predicted_conc"],
            "individual_values": r["individual_values"],
        })

    total_time = time.time() - start_time
    print(f"\nTotal time: {total_time:.1f}s")

    # Statistics
    true_arr      = np.array(true_vals,      dtype=float)
    predicted_arr = np.array(predicted_vals, dtype=float)
    mae  = float(np.mean(np.abs(true_arr - predicted_arr)))
    rmse = float(np.sqrt(np.mean((true_arr - predicted_arr) ** 2)))
    r2    = float(stats.pearsonr(true_arr, predicted_arr)[0] ** 2)

    print(f"\n{'='*40}")
    print(f"Regression Metrics:")
    print(f"  MAE  (Mean Absolute Error):       {mae:.3f} mmol/L")
    print(f"  RMSE (Root Mean Squared Error):   {rmse:.3f} mmol/L")
    print(f"  R^2  (Coefficient of Determination): {r2:.4f}")
    print(f"{'='*40}")

    return results, true_arr, predicted_arr, {"mae": mae, "rmse": rmse, "r2": r2, "total_time": total_time}


# ---------------------------------------------------------------------------
# Plot regression figure
# ---------------------------------------------------------------------------
def plot_regression(true_vals, predicted_vals, metrics: dict, results: list):
    """
    Generates a 2-panel regression figure:
      - Left: scatter + fit line + diagonal (perfect prediction)
      - Right: grouped bar chart (True vs Predicted per group)
    """
    true_arr = np.array(true_vals)
    pred_arr = np.array(predicted_vals)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ----- Left: regression scatter -----
    ax = axes[0]

    # Perfect prediction diagonal
    diag = np.linspace(0.5, 8.5, 100)
    ax.plot(diag, diag, "k--", linewidth=1.5, alpha=0.7, label="Perfect prediction (y=x)")

    # Scatter points, colored by group
    scatter = ax.scatter(true_arr, pred_arr, s=120, c=range(1, 9), cmap="viridis",
                        edgecolors="white", linewidths=1.5, zorder=5)

    # Vertical lines: true -> predicted (residuals)
    for t, p, g in zip(true_arr, pred_arr, range(1, 9)):
        ax.plot([t, t], [t, p], color="gray", linewidth=0.8, alpha=0.4, zorder=3)
        ax.annotate(f"G{g}", (t, p), textcoords="offset points",
                     xytext=(6, 6), fontsize=9, color="dimgray")

    # Linear fit
    slope, intercept, _, _, _ = stats.linregress(true_arr, pred_arr)
    fit_x = np.linspace(0.5, 8.5, 100)
    fit_y = slope * fit_x + intercept
    ax.plot(fit_x, fit_y, "r-", linewidth=2, alpha=0.8,
            label=f"Fit: y = {slope:.3f}x + {intercept:.3f}")

    ax.set_xlim(0.5, 8.5)
    ax.set_ylim(0.5, 8.5)
    ax.set_xlabel("True Concentration (mmol/L)", fontsize=12)
    ax.set_ylabel("Predicted Concentration (mmol/L)", fontsize=12)
    ax.set_title(f"True vs. Predicted Concentration\n"
                 f"MAE={metrics['mae']:.3f}  RMSE={metrics['rmse']:.3f}  R$^2$={metrics['r2']:.4f}",
                 fontsize=12, pad=10)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(1, 9))
    ax.set_yticks(range(1, 9))

    # ----- Right: grouped bar chart (True vs Predicted per group) -----
    ax2 = axes[1]
    groups        = [f"G{r['group']}" for r in results]
    true_vals_plot  = [r["true_conc"]      for r in results]
    pred_vals_plot = [r["predicted_conc"] for r in results]
    x_pos   = np.arange(1, 9)
    bar_w    = 0.35

    b_true  = ax2.bar(x_pos - bar_w / 2, true_vals_plot,  width=bar_w,
                       color="steelblue", alpha=0.85, edgecolor="white", linewidth=1.5,
                       label="True (mmol/L)")
    b_pred  = ax2.bar(x_pos + bar_w / 2, pred_vals_plot, width=bar_w,
                       color="orangered",  alpha=0.85, edgecolor="white", linewidth=1.5,
                       label="Predicted (mmol/L)")

    # Add value labels on top of each bar
    for bar in b_true:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.06,
                 f"{h:.1f}", ha="center", va="bottom", fontsize=8, color="steelblue")
    for bar in b_pred:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.06,
                 f"{h:.2f}", ha="center", va="bottom", fontsize=8, color="orangered")

    ax2.set_xlabel("Test Group", fontsize=11)
    ax2.set_ylabel("Concentration (mmol/L)", fontsize=11)
    ax2.set_title("True vs. Predicted by Group", fontsize=12)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f"G{g}" for g in range(1, 9)], fontsize=10)
    # Auto y-axis: max predicted value + 30% headroom
    max_val = max(max(true_vals_plot), max(pred_vals_plot))
    ax2.set_ylim(0, max_val * 1.35)
    ax2.legend(fontsize=10, framealpha=0.9)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig_path  = os.path.join(_OUTPUT_DIR, f"batch_test_{timestamp}.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig=None)
    print(f"\n[Saved] Figure: {fig_path}")
    return fig_path


# ---------------------------------------------------------------------------
# Save CSV and JSON
# ---------------------------------------------------------------------------
def save_outputs(results, true_vals, predicted_vals, metrics):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # CSV
    csv_path = os.path.join(_OUTPUT_DIR, f"batch_test_{timestamp}.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("group,true_conc,num_droplets,raw_mean,predicted_conc\n")
        for r in results:
            raw_str  = f"{r['raw_mean']:.4f}"      if r["raw_mean"]      is not None else "NaN"
            pred_str = f"{r['predicted_conc']:.4f}" if r["predicted_conc"] is not None else "NaN"
            f.write(f"{r['group']},{r['true_conc']},{r['num_droplets']},{raw_str},{pred_str}\n")
    print(f"[Saved] CSV: {csv_path}")

    # JSON
    json_path = os.path.join(_OUTPUT_DIR, f"batch_test_{timestamp}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": timestamp,
            "metrics":   metrics,
            "results":   results,
        }, f, indent=2, ensure_ascii=False)
    print(f"[Saved] JSON: {json_path}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"[INFO] All outputs will be saved to: {_OUTPUT_DIR}\n")
    results, true_vals, predicted_vals, metrics = run_batch()
    plot_regression(true_vals, predicted_vals, metrics, results)
    save_outputs(results, true_vals, predicted_vals, metrics)
