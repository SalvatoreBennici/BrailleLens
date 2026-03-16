# BrailleLens

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![YOLO](https://img.shields.io/badge/Ultralytics_YOLO-00FFFF?style=for-the-badge&logo=ultralytics&logoColor=black)
![OpenCV](https://img.shields.io/badge/OpenCV-%23white.svg?style=for-the-badge&logo=opencv&logoColor=black)
![FiftyOne](https://img.shields.io/badge/FiftyOne-FF6600?style=for-the-badge&logo=voxel51&logoColor=white)

**BrailleLens** is a Computer Vision pipeline designed for End-to-End Optical Braille Recognition (OBR). It recognises braille characters from raw images and real-time camera feeds of Braille text.
The system relies on a strictly decoupled two-stage architecture:

1. **Localization:** A fine-tuned YOLO detector identifies individual Braille characters.
2. **Classification:** A custom Convolutional Neural Network (CNN) classifies the specific dot patterns from the detected crops.

---

## Getting Started

[Pixi](https://prefix.dev/tools/pixi) is highly recommended for deterministic package management, though standard pip is fully supported.

### 1. Installation

**Option A: Using Pixi (Recommended)** Pixi automatically handles Python versions, CUDA dependencies, and system libraries without polluting the global environment.

```bash
pixi install
pixi shell
```

**Option B: Using Pip** Ensure a virtual environment is active before installing dependencies.

```bash
pip install -r requirements.txt
```

### 2. Dataset Preparation

Before running any evaluation or training scripts, the raw and processed datasets must be fetched:

```bash
python datasets/download_datasets.py
```

---

## Usage

### Notebooks (Quickstart & Visualization)

The `notebooks/` directory serves as the primary entry point to understand the project flow. **All notebooks are pre-executed** using the pre-trained weights stored in `runs/`. These can be reviewed directly on GitHub or executed locally:

* `01_EDA.ipynb`: Dataset analysis.
* `02_detector_evaluation.ipynb` & `03_classifier_evaluation.ipynb`: Isolated model metrics and errors analysis.
* `04_pipeline_evaluation.ipynb`: End-to-End diagnostic visualizations.
* `05_Inference_Translation.ipynb`: A simple example of a final reading-order translation applied to sample images.

### Dataset Exploration

To interactively explore the datasets, filter by confidence, or debug ground-truth annotations, the integrated FiftyOne viewer can be launched. The `--cache` flag enables database persistence for instantaneous future loading.

```bash
python src/data/explore_dataset.py --cache
```

### Training & Evaluation Scripts

The `src/` directory contains CLI-ready scripts for reproducibility. System behavior and hyperparameters are strictly controlled via `configs/*.yaml`.

* **Detector:** `src/detector/train.py` | `src/detector/evaluate.py`
* **Classifier:** `src/classifier/train.py` | `src/classifier/evaluate.py`
* **End-to-End:** `src/pipeline/evaluate.py`

### Real-Time Inference

For live demonstrations, BrailleLens supports real-time inference using an Android smartphone as a webcam via `scrcpy` and a V4L2 loopback device.

```bash
python src/demo_live.py
```

---

## Project Structure

The codebase is organized by separating data ingestion, model training, and the end-to-end pipeline.

```text
BrailleLens/
├── configs/            # Centralized YAML configurations (models, inference, data)
├── datasets/           # Download scripts, raw data, and processed YOLO/Crop datasets
├── notebooks/          # Pre-executed notebooks for EDA, and visualization
├── runs/               # Stored artifacts: pre-trained weights and metric reports
└── src/                # Core source code
    ├── classifier/     # CNN architecture, train, and validation scripts
    ├── detector/       # YOLO initialization, training, and validation scripts
    ├── data/           # Dataset generation tools and FiftyOne exploration scripts
    ├── pipeline/       # E2E integration
    └── demo_live.py    # Real-time V4L2 inference script (Android via scrcpy)

```
