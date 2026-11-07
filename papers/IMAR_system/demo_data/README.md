# Demo Images

This directory contains a pair of demo images (from the Trinder-1 test set) for
running the inference pipeline without any additional setup:

```
demo_data/original.jpg     -- Sample image (contains visible droplets)
demo_data/background.jpg   -- Background reference (no sample; captured under
                               identical camera position, exposure and lighting)
```

## Image Requirements

- **Format**: JPEG or PNG, color (RGB)
- **Resolution**: recommended 2048 x 2048 px
- **Capture protocol**: Do NOT move the camera or alter lighting between the
  two shots. Capture the background image first, then place the sample and
  capture the original image.

## Running the Demo

```bash
# Working directory must be the package root
python infer.py --original demo_data/original.jpg --background demo_data/background.jpg
```

Replace either image with your own data at any time.
