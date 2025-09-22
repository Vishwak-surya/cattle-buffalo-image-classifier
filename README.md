# Cattle & Buffalo Image Classifier

Image-based Animal Type Classification for cattle and buffaloes.

## Overview
This project provides a Python-based image classification pipeline to distinguish between cattle and buffaloes. It includes a simple ResNet-based model, dataset loaders using torchvision, and a minimal training script.

## Project structure
- src/cattle_buffalo_classifier: Python package with data, model, and training utilities
- scripts: Helper scripts for common tasks
- tests: Minimal tests to verify the package imports
- data (you create): Place your dataset here in the following structure:
  - data/
    - train/
      - cattle/
      - buffalo/
    - val/
      - cattle/
      - buffalo/

## Quickstart (Windows PowerShell)
1) Create and activate a virtual environment:
   - python -m venv .venv
   - .\.venv\Scripts\Activate.ps1

2) Install dependencies:
   - pip install --upgrade pip
   - pip install -r requirements.txt

3) Prepare your dataset under data/ as shown above.

4) Train a model (example):
   - python -m cattle_buffalo_classifier.train --data-dir data --epochs 5 --batch-size 32 --lr 1e-3 --image-size 224

## Notes
- The default model uses a torchvision ResNet18 with randomly initialized weights (no internet required). You can switch to pretrained weights if your environment allows downloads.
- You can extend transforms, augmentations, and model architectures as needed.
