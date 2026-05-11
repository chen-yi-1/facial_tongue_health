# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

面部与舌象健康特征识别系统 — 基于 MobileNet 与 Flask 的轻量化多模态健康状态三分类（健康/亚健康/不健康）系统，支撑毕业论文实验。

## Key Commands

使用 conda 环境 `tf` 中的 Python（路径：`E:\miniconda3\envs\tf\python.exe`），已将 `E:\miniconda3\envs\tf` 加入 `PATH` 前置。

```bash
# Install dependencies (已有环境，无需重装)
pip install -r requirements.txt

# Train model (MobileNetV2 default, set MODEL_VERSION=v3 for V3Small)
python train.py           # default: 10 epochs
EPOCHS=20 python train.py
MODEL_VERSION=v3 python train.py

# Start Flask web service
python app/app.py         # http://127.0.0.1:5000

# Run comparative experiments (face vs tongue vs multimodal)
python scripts/run_experiments.py --epochs 5 --backbone_version v3

# Validate dataset structure
python scripts/smoke_check_dataset.py

# Test API endpoints (requires Flask running)
python scripts/test_api_cases.py
python scripts/test_predict.py
```

## Architecture

### Model (`models/mobilenet_multimodal.py`)
- **Dual-branch multimodal architecture**: face image → MobileNet backbone → features → concatenate → Dense(256) → Dropout(0.5) → Dense(3, softmax) ← tongue image
- Two backbone variants: `MobileNetV2` (default) and `MobileNetV3Small`, selected via `build_multimodal_mobilenet_by_version("v2"/"v3")`
- Backbone layers renamed with `{face/tongue}_` prefix to avoid Keras name collisions when sharing the same base class in two branches
- All backbone weights frozen by default (`trainable=False`), only the fusion head trains
- Image size: 224×224, pooling: avg

### Training (`train.py`)
- **Paired dataset strategy**: within each class, face and tongue images are paired by index after taking `min(len(face), len(tongue))` samples — not real paired captures but a reproducible experimental setup
- Custom `ManualValidation` callback computes validation metrics each epoch (avoids Keras validation_* quirks in some TF versions)
- Model exported as `tf.saved_model` + separate `.h5` weights
- Output path: `saved_models/multimodal_mobilenet_{v2/v3}_savedmodel`

### Flask Web Service (`app/app.py`)
- Two routes: `GET /` (index) and `POST /predict` (multipart form: face_image + tongue_image)
- Handles both Keras `Model.predict()` and raw `tf.saved_model` inference (serving_default signature)
- Returns JSON: `{success, label, face_image_url, tongue_image_url, probabilities}`
- Secure file serving via `send_from_directory` with `os.path.basename` sanitization
- Missing either image → 400; invalid image → 400

### Comparative Experiments (`scripts/run_experiments.py`)
- Self-contained experiment runner (does NOT import `models/` or `train.py` — duplicates model building code for reproducibility)
- Three modes: `face` (single-branch), `tongue` (single-branch), `multimodal` (dual-branch)
- Outputs per-mode: loss/accuracy curve, confusion matrix, classification report, metrics JSON
- Runs with `--backbone_version v2/v3` for paper comparison tables

### Data Preparation Scripts
- `scripts/prepare_dataset.py` — generic dataset reorganizer with `--map src=target` label mapping
- `scripts/prepare_face_from_kaggle.py` — FFHQ + acne + face-skin-disease → data/face/ structure
- `scripts/prepare_tongue_from_kaggle.py` — tongue datasets with augmentation when training samples < target
- `scripts/download_public_datasets.py` — download from Dryad (no login) and Kaggle (requires kaggle.json)

## Dataset Structure

```
data/
  face/ (or tongue/)
    train/
      healthy/
      subhealthy/
      unhealthy/
    val/
      healthy/
      subhealthy/
      unhealthy/
```

## Important Notes

- The paired dataset does NOT represent real face-tongue pairs from the same person — it's same-class index-based pairing for multimodal fusion experiments. This must be clearly stated in the thesis.
- SavedModel export may fail on some TF/Keras versions due to structure serialization bugs; the code falls back to `tf.saved_model.load()` + `serving_default` signature at inference time
- Base dependency versions: TensorFlow 2.15.0, Flask 3.0.2, numpy 1.26.4, OpenCV 4.9.0
