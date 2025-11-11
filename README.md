# UCRS: Uncertainty-Calibrated Region Sampling for Efficient Small Object Detection in High-Resolution Images

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.6+](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.7+-orange.svg)](https://pytorch.org/)

This repository contains the official implementation of **UCRS (Uncertainty-Calibrated Region Sampling)**, a framework that employs evidential learning to estimate prediction confidence and allocate computation adaptively across image regions. UCRS achieves a 60-75% reduction in FLOPs while maintaining detection accuracy, making it ideal for UAV or surveillance systems operating under tight real-time constraints.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Testing and Evaluation](#testing-and-evaluation)
- [Inference](#inference)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Model Variants](#model-variants)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Overview

UCRS addresses the fundamental computational challenge of detecting small objects (occupying < 0.1% of image area) in high-resolution imagery (exceeding 2048×2048 pixels). By linking uncertainty quantification with spatial attention, UCRS adaptively focuses computational effort only where the information content is expected to be high.

The framework consists of three key components:

1. **Uncertainty Estimation Module**: Employs evidential learning with gamma distribution modeling to predict mean likelihood (μ) and confidence (σ²) at each spatial location
2. **RegionLocator**: Generates class-agnostic likelihood maps to identify areas of interest
3. **Uncertainty-Calibrated Region Selection**: Uses confidence-weighted metrics to make principled, probabilistic decisions about computational allocation

**Key Results:**
- **VisDrone**: 44.4% AP50 and 25.2% AP at 34.9 FPS with 52.5 GFLOPs (65-94% computation reduction)
- **TinyPerson**: 35.1% AP50 and 10.9% AP at 19.4 FPS
- **Efficiency**: 60-75% reduction in FLOPs while maintaining detection accuracy
- **Precision**: 5-8% improvement over deterministic region sampling at equivalent computational budgets

This implementation includes:
- Complete UCRS framework with evidential learning-based uncertainty estimation
- Full training and evaluation pipelines for TinyPerson and VisDrone datasets
- Comprehensive model comparison tools
- Automatic metrics logging and visualization

## Key Features

### Core Capabilities

- **Multi-Dataset Support**: Pre-configured for VisDrone and TinyPerson datasets
- **Uncertainty-Calibrated Region Sampling**: Probabilistic resource allocation using evidential learning
- **Model Variants**:
  - Base YOLOv5 (baseline)
  - UCRS (Uncertainty-Calibrated Region Sampling with evidential learning)
- **Automatic Metrics Logging**: CSV-based training metrics with epoch-by-epoch tracking
- **Structured Results**: Organized test results with quantitative and qualitative outputs
- **Model Comparison Tools**: Side-by-side evaluation of multiple trained models
- **Adaptive Hyperparameters**: Dynamic learning rate and loss weight scheduling
- **Multi-GPU Training**: Distributed Data Parallel (DDP) support
- **Ground Truth Validation**: Tools for verifying dataset annotation quality

### Technical Highlights

- **Evidential Learning**: Single-pass uncertainty estimation using gamma distribution modeling (α, β parameters)
- **Confidence-Weighted Selection**: Probabilistic region allocation based on E_conf = μ · (1 - σ²)
- **Computational Efficiency**: 60-75% FLOP reduction while maintaining competitive accuracy
- **Real-Time Performance**: Up to 34.9 FPS on VisDrone and 19.4 FPS on TinyPerson
- **Flexible Configuration**: YAML-based model and hyperparameter configuration
- **Comprehensive Evaluation**: Support for COCO-style metrics (mAP@0.5, mAP@0.5:0.95) and official dataset evaluation tools

## Installation

### Prerequisites

- **Python**: >= 3.6.0
- **PyTorch**: >= 1.7 (CUDA support recommended)
- **CUDA**: 11.1+ (optional, for GPU acceleration)

### Step-by-Step Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd esod
   ```

2. **Install base dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install PyTorch** (adjust CUDA version as needed):
   ```bash
   pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
   ```

4. **Install setuptools** (for compatibility):
   ```bash
   pip install setuptools==59.5.0
   ```

5. **Optional: Install Segment Anything Model (SAM)** for enhanced ground truth heatmaps:
   ```bash
   cd third_party/segment-anything
   pip install -e .
   cd ../..
   ```

### Verify Installation

Run the diagnostic script to verify your installation:

```bash
python scripts/check_esod.py
```

This will check:
- Dataset configuration files
- Model building capability
- Data loading functionality
- Forward pass execution

## Quick Start

### Minimal Training Command

Train with default settings:

```bash
python train.py
```

### Minimal Testing Command

Test with default settings:

```bash
python test.py
```

### Complete Workflow Example

For a complete training and evaluation workflow on TinyPerson:

```powershell
# 1. Prepare dataset
python scripts/data_prepare.py --dataset dataset/tinyperson

# 2. Train UCRS model
python train.py `
    --data data/tinyperson.yaml `
    --cfg models/cfg/esod/tinyperson_yolov5m_uncertainty.yaml `
    --weights weights/yolov5m.pt `
    --hyp data/hyps/hyp.tinyperson.finetune.yaml `
    --batch-size 8 `
    --img-size 2048 2048 `
    --epochs 50 `
    --device 0 `
    --name tinyperson_yolov5m_ucrs

# 3. Evaluate model
python test.py test `
    --data data/tinyperson.yaml `
    --weights runs/train/tinyperson_yolov5m_ucrs/weights/best.pt `
    --batch-size 8 `
    --img-size 2048 `
    --device 0 `
    --task test
```

## Data Preparation

### Supported Datasets

The framework supports three primary datasets:

1. **VisDrone**: Aerial surveillance dataset with 10 object classes
2. **UAVDT**: UAV-based detection and tracking dataset
3. **TinyPerson**: Small person detection dataset

### Dataset Organization

Datasets should be organized under the `dataset/` directory:

```
dataset/
├── VISDRONE/
├── uavdt/
└── tinyperson/
```

### Data Preparation Script

Use the unified data preparation script for all datasets:

```bash
# VisDrone
python scripts/data_prepare.py --dataset dataset/VISDRONE

# UAVDT
python scripts/data_prepare.py --dataset dataset/uavdt

# TinyPerson
python scripts/data_prepare.py --dataset dataset/tinyperson
```

### Dataset-Specific Instructions

#### VisDrone

1. Download the official VisDrone dataset
2. Organize splits as follows:
   ```
   dataset/VISDRONE/
     VisDrone2019-DET-train/
     VisDrone2019-DET-val/
     VisDrone2019-DET-test-dev/
     VisDrone2019-DET-test-challenge/
   ```
3. Run the preparation script (generates YOLO labels, heatmaps, and split files)

#### UAVDT

1. Download the UAVDT dataset
2. Run the preparation script to generate YOLO-format labels and split files

#### TinyPerson

1. Download the TinyPerson dataset
2. If you already have YOLO-format data in `train/valid/test` directories, you can skip preparation
3. Otherwise, run the preparation script to convert annotations

### Ground Truth Heatmaps (Optional)

For enhanced training, leverage the Segment Anything Model (SAM) to generate precise shape priors:

1. Install SAM (see [Installation](#installation))
2. Place SAM checkpoint in the appropriate location
3. The data preparation script will automatically use SAM if available

### Dataset Format

The framework expects Darknet/YOLO format:
- **Images**: `.jpg` or `.png` files
- **Labels**: `.txt` files with normalized coordinates (class_id x_center y_center width height)
- **Heatmaps**: `.npy` files (optional, generated during preparation)

Label files are automatically located by replacing `/images/` with `/labels/` in image paths.

## Training

### Training Prerequisites

Before training, ensure:
1. Pretrained weights are downloaded (e.g., `yolov5m.pt` in `weights/` directory)
2. Dataset is prepared (see [Data Preparation](#data-preparation))
3. All dependencies are installed (see [Installation](#installation))

### Model Variants

#### 1. Base YOLOv5 (Baseline)

Standard YOLOv5 model without UCRS modifications. Provides a baseline for comparison.

**Training Command**:
```powershell
python train.py `
    --data data/tinyperson.yaml `
    --cfg models/cfg/yolov5m.yaml `
    --weights weights/yolov5m.pt `
    --hyp data/hyps/hyp.tinyperson.finetune.yaml `
    --batch-size 8 `
    --img-size 2048 2048 `
    --epochs 50 `
    --device 0 `
    --name tinyperson_yolov5m_base `
    --project runs/train
```

#### 2. UCRS (Uncertainty-Calibrated Region Sampling)

UCRS framework with evidential learning-based uncertainty estimation, RegionLocator, and uncertainty-calibrated region selection.

**Training Command**:
```powershell
python train.py `
    --data data/tinyperson.yaml `
    --cfg models/cfg/esod/tinyperson_yolov5m_uncertainty.yaml `
    --weights weights/yolov5m.pt `
    --hyp data/hyps/hyp.tinyperson.finetune.yaml `
    --batch-size 8 `
    --img-size 2048 2048 `
    --epochs 50 `
    --device 0 `
    --name tinyperson_yolov5m_ucrs `
    --project runs/train
```

**Key Hyperparameters:**
- `τ_conf = 0.3`: Confidence threshold for region selection
- `λ1 = 0.2`: Weight for RegionLocator loss
- `λ2 = 0.1`: Weight for confidence estimation loss

### Multi-GPU Training

For distributed training across multiple GPUs:

```powershell
python -m torch.distributed.launch --nproc_per_node 4 train.py `
    --data data/tinyperson.yaml `
    --cfg models/cfg/esod/tinyperson_yolov5m_uncertainty.yaml `
    --weights weights/yolov5m.pt `
    --hyp data/hyps/hyp.tinyperson.finetune.yaml `
    --batch-size 32 `
    --img-size 2048 2048 `
    --epochs 50 `
    --device 0,1,2,3 `
    --name tinyperson_yolov5m_ucrs `
    --project runs/train
```

**Note**: Increase batch size proportionally with the number of GPUs (e.g., 4 GPUs → 4x batch size).

### Training Features

#### Automatic Metrics Logging

All training metrics are automatically saved to CSV files:
- **Location**: `runs/train/{experiment_name}/metrics.csv`
- **Contents**: Epoch-by-epoch train/val losses, mAP, precision, recall, learning rates
- **Analysis**: Open in Excel/pandas for plotting and analysis

#### Auto-Resume

Training automatically resumes from the last checkpoint if:
- A checkpoint exists at `runs/train/{experiment_name}/weights/last.pt`
- The checkpoint contains valid training state
- `--resume` is not explicitly set to `False`

#### Adaptive Hyperparameters

The framework includes adaptive hyperparameter scheduling (enabled by default):
- Dynamic learning rate adjustment
- Loss weight balancing
- Can be disabled with `--disable-adaptive-hyp`

### Training Tips

1. **Batch Size**: Adjust based on GPU memory. If OOM errors occur, reduce batch size (try 4 or 2) or reduce image size.

2. **Image Size**: 
   - TinyPerson: 2048x2048 (recommended)
   - VisDrone: 1536x1536
   - UAVDT: 1280x1280

3. **Epochs**: Start with 50 epochs. Monitor `metrics.csv` to see when validation metrics plateau.

4. **Resume Training**: 
   ```powershell
   python train.py ... --resume runs/train/tinyperson_yolov5m_base/weights/last.pt
   ```

5. **Check Logs**: Training logs are saved to `runs/train/{name}/training.log`

### Training on Other Datasets

#### VisDrone
```powershell
python train.py `
    --data data/visdrone.yaml `
    --cfg models/cfg/esod/visdrone_yolov5m_uncertainty.yaml `
    --weights weights/yolov5m.pt `
    --hyp data/hyps/hyp.visdrone.finetune.yaml `
    --batch-size 8 `
    --img-size 1536 1536 `
    --epochs 50 `
    --device 0 `
    --name visdrone_yolov5m_ucrs
```

## Testing and Evaluation

### Individual Model Evaluation

Evaluate a trained model on the test set:

```powershell
python test.py test `
    --data data/tinyperson.yaml `
    --weights runs/train/tinyperson_yolov5m_base/weights/best.pt `
    --batch-size 8 `
    --img-size 2048 `
    --device 0 `
    --task test

python test.py test `
    --data data/tinyperson.yaml `
    --weights runs/train/tinyperson_yolov5m_ucrs/weights/best.pt `
    --batch-size 8 `
    --img-size 2048 `
    --device 0 `
    --task test
```

### Evaluation Metrics

The framework reports comprehensive metrics:

- **Precision (P)**: Proportion of positive detections that are correct
- **Recall (R)**: Proportion of actual positives detected
- **mAP@0.5**: Mean Average Precision at IoU threshold 0.5
- **mAP@0.5:0.95**: Mean Average Precision averaged over IoU thresholds 0.5 to 0.95
- **BPR**: Best Possible Recall
- **Occupy**: Occupancy metric
- **Loss Components**: Box, objectness, classification, pixel, area, and distance losses

### Model Comparison

Compare multiple trained models side-by-side:

```powershell
python scripts/test_models.py `
    --data data/tinyperson.yaml `
    --weights-dir runs/train `
    --models tinyperson_yolov5m_base tinyperson_yolov5m_ucrs `
    --batch-size 8 `
    --img-size 2048 `
    --device 0 `
    --test-split test `
    --csv-output tinyperson_comparison.csv
```

This generates:
- Side-by-side comparison table in terminal
- Comparison CSV: `runs/test_comparison/tinyperson_comparison.csv`
- Detailed metrics for each model

### Computational Analysis

Measure GFLOPs and FPS for any model:

```powershell
python test.py test `
    --data data/tinyperson.yaml `
    --weights runs/train/tinyperson_yolov5m_base/weights/best.pt `
    --batch-size 1 `
    --img-size 2048 `
    --device 0 `
    --task measure
```

### Official Dataset Evaluation

For official evaluation on VisDrone test-dev set:

1. **Generate predictions in Darknet format**:
   ```powershell
   python test.py test `
       --data data/visdrone.yaml `
       --weights weights/yolov5m.pt `
       --batch-size 8 `
       --img-size 1536 `
       --device 0 `
       --task test `
       --save-txt `
       --save-conf
   ```

2. **Convert to official format**:
   ```bash
   python scripts/data_convert.py --dataset VisDrone --pred runs/test/exp/labels
   ```

3. **Run official evaluation**:
   ```bash
   cp -r ./evaluation/VisDrone2018-DET-toolkit ./VisDrone/
   cd ./VisDrone/VisDrone2018-DET-toolkit
   matlab -nodesktop -nosplash -r evalDET
   ```

### Structured Results

Test results are automatically organized in a structured format:

```
results/
├── {dataset_name}/
│   ├── {model_name}/
│   │   ├── test_summary.json
│   │   ├── quantitative_results.csv
│   │   └── individual_results/
│   │       ├── top5_best/
│   │       └── top5_worst/
```

## Inference

### Single Image or Directory

Run inference on images or directories:

```powershell
python detect.py `
    --weights weights/yolov5m.pt `
    --source data/images/visdrone.txt `
    --img-size 1536 `
    --device 0 `
    --view-cluster `
    --line-thickness 1
```

### Inference Options

- `--view-cluster`: Draw generated patches in green boxes and save heatmaps
- `--save-txt`: Save detection results in YOLO format
- `--save-conf`: Include confidence scores in saved results
- `--conf-thres`: Confidence threshold (default: 0.25)
- `--iou-thres`: IoU threshold for NMS (default: 0.45)

### Output

Results are saved to `runs/detect/exp/`:
- Annotated images with bounding boxes
- Optional: Patch visualizations and heatmaps
- Optional: Text files with detection coordinates

## Project Structure

```
esod/
├── data/                    # Dataset configuration files
│   ├── *.yaml              # Dataset YAML configs
│   └── hyps/              # Hyperparameter files
├── dataset/                # Dataset storage (user-provided)
│   ├── VISDRONE/
│   ├── uavdt/
│   └── tinyperson/
├── models/                 # Model definitions
│   ├── cfg/               # Model configuration files
│   │   ├── esod/         # RE-ScopeNet model configs
│   │   └── vanilla/      # Baseline model configs
│   ├── yolo.py           # YOLO model implementation
│   ├── common.py         # Common model components
│   ├── uncertainty_segmenter.py  # Confidence-guided segmenter
│   └── uncertainty_slicer.py    # Probabilistic slicer
├── utils/                  # Utility modules
│   ├── datasets.py       # Data loading utilities
│   ├── loss.py           # Loss computation
│   ├── metrics.py        # Evaluation metrics
│   ├── general.py        # General utilities
│   ├── csv_logger.py     # CSV metrics logging
│   └── adaptive_hyperparameters.py  # Adaptive hyperparameter scheduling
├── scripts/               # Helper scripts
│   ├── data_prepare.py   # Dataset preparation
│   ├── test_models.py    # Model comparison
│   ├── check_gt_alignment.py  # GT validation
│   └── ...
├── evaluation/            # Official evaluation tools
│   ├── VisDrone2018-DET-toolkit/
│   └── ...
├── runs/                  # Training and testing outputs
│   ├── train/            # Training experiments
│   └── test/             # Test results
├── results/              # Structured test results
├── train.py             # Training script
├── test.py              # Testing script
├── detect.py            # Inference script
└── requirements.txt     # Python dependencies
```

## Configuration

### Model Configuration

Model architectures are defined in YAML files under `models/cfg/`:

- **RE-ScopeNet Models**: `models/cfg/esod/*.yaml`
- **Baseline Models**: `models/cfg/vanilla/*.yaml` or `models/cfg/yolov5*.yaml`

Key configuration parameters:
- `nc`: Number of classes
- `depth_multiple`: Model depth multiplier
- `width_multiple`: Model width multiplier
- `anchors`: Anchor box definitions
- `backbone`: Backbone architecture definition
- `head`: Detection head definition

### Hyperparameter Configuration

Hyperparameters are defined in YAML files under `data/hyps/`:

- Dataset-specific: `hyp.{dataset}.{mode}.yaml`
- Example: `hyp.tinyperson.finetune.yaml`

Key hyperparameters:
- Learning rate schedules
- Loss weights (box, objectness, classification, etc.)
- Data augmentation parameters
- Confidence weight (for confidence-guided models)

### Dataset Configuration

Dataset configurations are in `data/*.yaml`:

```yaml
train: ./dataset/tinyperson/train/images
val: ./dataset/tinyperson/valid/images
test: ./dataset/tinyperson/test/images
nc: 1
names: ['person']
```

## Model Variants

### Base YOLOv5

- **Config**: `models/cfg/yolov5m.yaml`
- **Description**: Standard YOLOv5m without UCRS modifications
- **Use Case**: Baseline for comparison

### UCRS (Uncertainty-Calibrated Region Sampling)

- **Config**: `models/cfg/esod/{dataset}_yolov5m_uncertainty.yaml`
- **Components**: 
  - Uncertainty Estimation Module (evidential learning with gamma distribution)
  - RegionLocator (class-agnostic likelihood maps)
  - Uncertainty-Calibrated Region Selection (confidence-weighted sampling)
- **Description**: Framework that employs evidential learning to estimate prediction confidence and allocate computation adaptively
- **Benefits**: 
  - 60-75% reduction in FLOPs while maintaining detection accuracy
  - 5-8% precision improvement over deterministic region sampling
  - Real-time performance (up to 34.9 FPS on VisDrone)
- **Use Case**: Efficient small object detection in high-resolution imagery with computational constraints

### Key Differences

| Feature | Base YOLOv5 | UCRS |
|---------|-------------|------|
| Patch-based Processing | ❌ | ✅ |
| Region Likelihood Masking | ❌ | ✅ |
| Dynamic Patch Reduction | ❌ | ✅ |
| Evidential Learning | ❌ | ✅ |
| Uncertainty Quantification | ❌ | ✅ |
| Confidence-Weighted Selection | ❌ | ✅ |
| Probabilistic Resource Allocation | ❌ | ✅ |

## Troubleshooting

### Common Issues

#### 1. Out of Memory (OOM) Errors

**Solution**:
- Reduce batch size: `--batch-size 4` or `--batch-size 2`
- Reduce image size: `--img-size 1536` or `--img-size 1280`
- Use gradient accumulation (modify training script)

#### 2. Dataset Not Found

**Solution**:
- Verify dataset paths in `data/*.yaml` files
- Run `python scripts/data_prepare.py --dataset dataset/{dataset_name}`
- Check that image and label directories exist

#### 3. Missing Pretrained Weights

**Solution**:
- Download weights from [YOLOv5 releases](https://github.com/ultralytics/yolov5/releases)
- Place in `weights/` directory
- Or specify path with `--weights path/to/weights.pt`

#### 4. Ground Truth Alignment Issues

**Solution**:
- Run validation script: `python scripts/check_gt_alignment.py`
- Check coordinate format (should be normalized: 0-1)
- Verify label files match image files

#### 5. Training Metrics Not Saving

**Solution**:
- Check write permissions in `runs/train/` directory
- Verify CSV logger is initialized (check training logs)
- Ensure sufficient disk space

### Getting Help

1. Check existing documentation in `docs/` directory
2. Review training logs: `runs/train/{name}/training.log`
3. Run diagnostic script: `python scripts/check_esod.py`
4. Verify dataset format: `python scripts/check_gt_alignment.py`

## Citation

If you use this code in your research, please cite the UCRS paper:

```bibtex
@article{ucrs2025,
    title={UCRS: Uncertainty-Calibrated Region Sampling for Efficient Small Object Detection in High-Resolution Images},
    author={1st Author Name and 2nd Author Name and 3rd Author Name and 4th Author Name},
    journal={[Journal/Conference Name]},
    year={2025}
}
```

**Authors:**
- 1st Author Name, Department of Electronics and Communication Engineering, National Institute of Technology Rourkela
- 2nd Author Name, Department of Electronics and Communication Engineering, National Institute of Technology Rourkela
- 3rd Author Name, Department of Electronics and Communication Engineering, National Institute of Technology Rourkela
- 4th Author Name, Department of Electronics and Communication Engineering, National Institute of Technology Rourkela

## License

This project is licensed under the GNU General Public License v3.0. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- This implementation is based on [YOLOv5](https://github.com/ultralytics/yolov5) by Ultralytics
- Ground truth heatmap generation leverages [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything) by Meta AI Research
- We thank the authors of related work for their valuable contributions

---

**Note**: This repository contains the UCRS (Uncertainty-Calibrated Region Sampling) implementation. The framework employs evidential learning for uncertainty estimation and confidence-weighted region selection. Use the `*_uncertainty.yaml` config files to train UCRS models with uncertainty-calibrated region sampling.
