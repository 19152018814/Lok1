# IMAR — Intelligent Microdroplet Analytical Robotics

**What it does:** Background-subtracted image enhancement → YOLO droplet detection →
CNN concentration regression inference for colourmetric microdroplet analytics.

---

## 1. System Requirements

| Item       | Minimum                              | Recommended                         |
|------------|--------------------------------------|-------------------------------------|
| OS         | Windows 10 / Ubuntu 22.04 / macOS   | same                                |
| Python     | 3.10 – 3.12                          | 3.11                                |
| RAM        | 8 GB                                 | 16 GB                               |
| GPU        | Optional (CPU fallback supported)     | NVIDIA GPU, CUDA ≥ 11.8, VRAM ≥ 6 GB|
| Disk       | ~500 MB (code + model weights)       | same                                |

**Validated environment:** Windows 10, Python 3.11.9, NVIDIA RTX 3060 (12 GB),
CUDA 12.1, cuDNN 8.9.

No specialised hardware (microscope, industrial camera, etc.) is required; any
imaging system that produces standard image files is compatible.

---

## 2. Installation Guide

```bash
# 1. Navigate to the unpacked package directory
cd nature_code_submission

# 2. (Recommended) Create a virtual environment
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# 3. Upgrade pip
pip install --upgrade pip

# 4. Install all dependencies
pip install -r requirements.txt

# 5. (NVIDIA GPU only) Replace step 4 with a CUDA-enabled PyTorch build:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

> **Note:** `requirements.txt` installs the CPU-only PyTorch build by default.
> Run step 4 directly if no GPU is available (CPU inference is 3–5x slower).

**Typical installation time:** 5–15 min on a typical broadband connection;
PyTorch accounts for 3–8 min of the download.

---

## 3. Demo

### 3.1 Demo Data

`demo_data/` contains one image pair from the Trinder-1 test set:

| File                    | Description                                              |
|-------------------------|----------------------------------------------------------|
| `demo_data/original.jpg`    | Sample image (contains visible colourmetric reaction droplets) |
| `demo_data/background.jpg`  | Background reference (captured without sample under identical conditions) |

### 3.2 Run the Demo

```bash
# Working directory must be the package root
python infer.py --original demo_data/original.jpg --background demo_data/background.jpg
```

### 3.3 Expected Output

The program prints a JSON result to the console (truncated example):

```json
{
  "success": true,
  "timestamp": "20260410_200057",
  "result_image_path": "uploads/20260410_200057/result.jpg",
  "model": "trinder_1",
  "num_detections": 23,
  "detections": [
    {
      "bbox": [703.563, 931.985, 742.367, 973.864],
      "confidence": 0.812,
      "class_name": "droplet_1",
      "raw_prediction": 5.824,
      "roi_path": "uploads/20260410_200057/roi_1.jpg"
    }
    // ... 23 droplets total
  ],
  "concentration": {
    "raw_mean": 5.796,
    "final_value": 5.796,
    "std": 0.126,
    "unit": "mmol/L",
    "description": "Glucose Concentration (Trinder-1)",
    "individual_values": [5.824, 5.706, 5.783, ...],
    "num_droplets": 23
  }
}
```

A reference output file (`expected_output/demo_run_reference.json`) is included.

### 3.4 Expected Runtime

| Hardware       | Time per image |
|----------------|----------------|
| CPU (8 cores)  | 15 – 30 s      |
| GPU (RTX 3060) | 3 – 8 s        |

The YOLO detector and CNN model are loaded into memory on first call and
reused for subsequent requests.

---

## 4. Instructions for Use

### 4.1 Image Acquisition Requirements

| Property      | Requirement                                                    |
|--------------|----------------------------------------------------------------|
| Format       | JPEG or PNG, colour (RGB)                                      |
| Resolution   | ≥ 1920 × 1080 px (2048 × 2048 px recommended)               |
| Conditions   | Stable illumination; background and sample images captured without moving the camera or changing lighting |
| Camera       | Any device that outputs standard image files (flatbed scanner, microscope camera, industrial camera, etc.) |

**Capture procedure:**
1. Without placing any sample, capture the background image (`background.jpg`).
2. Place the sample and capture the sample image (`original.jpg`).
3. Do **not** move the camera or alter lighting between the two shots.

### 4.2 Running with Your Own Data

```bash
python infer.py --original /path/to/your_sample.jpg \
                --background /path/to/your_background.jpg
```

Both absolute paths and paths relative to the package root are supported.

### 4.3 Changing the Package Root Directory

To run the scripts from outside the package directory, set the `IMAR_BASE_DIR`
environment variable:

```powershell
# Windows PowerShell
$env:IMAR_BASE_DIR = "D:\path\to\nature_code_submission"
python infer.py --original demo_data/original.jpg --background demo_data/background.jpg
```

```bash
# Linux / macOS
export IMAR_BASE_DIR=/path/to/nature_code_submission
python infer.py --original demo_data/original.jpg --background demo_data/background.jpg
```

### 4.4 Output Files

After each inference run, results are written to `uploads/<timestamp>/`:

| File           | Contents                                                     |
|---------------|--------------------------------------------------------------|
| `result.jpg`    | Background-subtracted (enhanced) image                        |
| `roi_1.jpg … roi_N.jpg` | Cropped ROI of each detected droplet                 |

### 4.5 HTTP API Server (server.py)

```bash
python server.py
# Server starts at http://0.0.0.0:8000
# Interactive API docs: http://0.0.0.0:8000/docs
```

```bash
curl -X POST "http://localhost:8000/process_images" \
  -F "original_image=@demo_data/original.jpg" \
  -F "background_image=@demo_data/background.jpg" \
  -F "model_name=trinder_1"
```

### 4.6 Batch Testing

Use `batch_test.py` to run inference on all 8 groups of the Trinder-1 test set
and generate a regression plot:

```bash
python batch_test.py
# Outputs saved to batch_results/batch_test_<timestamp>/
```

Results include:
- `batch_test_<timestamp>.png` — regression plot (scatter + fit line + grouped bars)
- `batch_test_<timestamp>.csv` — raw data table
- `batch_test_<timestamp>.json` — complete run record

### 4.7 Training Code

This package contains **inference code and weights only**. The training scripts and
full training pipeline are available from the authors upon request.

---

## 5. File Structure

```
nature_code_submission/
├── README.md
├── requirements.txt
├── infer.py                          # CLI inference entry point
├── server.py                         # FastAPI HTTP server (optional)
├── batch_test.py                     # 8-group batch test + regression plot
├── models/
│   ├── yolo/
│   │   └── best2.pt                  # YOLO droplet detector (included)
│   └── trinder_1/results_reg_cnn/
│       └── best_model.pth             # CNN regression weights (included)
├── demo_data/
│   ├── original.jpg                  # Demo sample image (included)
│   └── background.jpg                # Demo background image (included)
├── expected_output/
│   └── demo_run_reference.json        # Demo reference output
├── batch_results/                    # Auto-created; holds batch test outputs
└── uploads/                          # Auto-created; holds per-run inference outputs
```

---

## 6. API Reference

### POST `/process_images`

**Parameters (multipart/form-data):**

| Field              | Type   | Required | Description                             |
|--------------------|--------|----------|-----------------------------------------|
| `original_image`   | File   | Yes      | Sample image file                       |
| `background_image` | File   | Yes      | Background reference image file          |
| `model_name`       | string | Yes      | Model identifier (only `"trinder_1"`)  |

**Response:** JSON object (same schema as the `infer.py` output).

### GET `/health`

Returns server health status and system resource usage.

### GET `/models`

Lists all available models.

---

*IMAR v1.0 — Nature Sensors 2026 Supplementary Software.*
