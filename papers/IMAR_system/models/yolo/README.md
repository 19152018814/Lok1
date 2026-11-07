# YOLO Droplet Detection Weights — best2.pt

Place your trained YOLO weight file at:

```
models/yolo/best2.pt
```

**How to obtain `best2.pt`:**

After Ultralytics YOLO training completes, the best weights are saved at:

```
runs/detect/train*/weights/best.pt
```

Copy that file into this directory and rename it `best2.pt` to match the path
used in `infer.py`.

**Model details:**

| Property | Value |
|----------|-------|
| Input | RGB image (recommended resolution >= 1024 x 1024 px) |
| Output | Bounding boxes of detected droplets |
| Task | Object detection (single-class: "droplet") |

**Note:** `infer.py` loads this model at runtime; no recompilation is needed.
