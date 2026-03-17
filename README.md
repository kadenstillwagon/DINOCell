# DINOCell: Self-supervised Pretraining of Cell Segmentation Models

[![Python 3.8+](https://img.shields.io/badge/python-3+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Paper-Arxiv-red.svg)](https://doi.org/10.1371/journal.pone.0319532)

DINOCell is an automated cell segmentation model for live cell microscopy images. Through initialization with DINOv2 weights, pretrained on 1.2B images, and domain-adaptation on 130k unlabeled cell images, DINOCell achieves unparalleled performance across a wide variety of cell types and microscope conditions.

![PBL HEK Segmentation](https://github.com/kadenstillwagon/DINOCell/blob/main/assets/PBL_HEK_example_out.png)

## 🌟 Key Features

- **State-of-the-art Performance**: Outperforms existing methods like Cellpose-SAM and SAMCell on both test-set and zero-shot cross-dataset evaluation
- **Zero-shot Generalization**: Works on novel cell types and microscopes not seen during training
- **Vision Transformer Architecture**: Leverages DINOv2's ViT-based encoder pretrained on 1.2B images for robust image representations, which are further tuned for microscopy through domain-adaptation on 130k unlabeled cell images.
- **Flows Regression**: Predicts Cellpose-style Flows instead of binary masks, enabling better separation of densely packed cells

## 📊 Performance

DINOCell demonstrates superior performance in both test-set and zero-shot cross-dataset evaluation:

### LIVECell Test-Set Performance

| Method | SEG | DET | MMA |
|--------|-----|-----|------|
|**DINOCell** | **0.784** | **0.926** | **0.876** |
| Cellpose-SAM | 0.710 | 0.852 | 0.807 |
| SAMCell-LIVECell | 0.744 | 0.911 | 0.834 |

### Zero-Shot Cross-Dataset Performance

| Dataset | Method | SEG | DET | MMA |
|---------|--------|-----|-----|--------|
| PBL-HEK | **DINOCell** | **0.553** | **0.818** | **0.734** |
| | Cellpose-SAM | 0.452 | 0.627 | 0.617 |
| | SAMCell-cyto | 0.458 | 0.765 | 0.628 |
| | SAMCell-LIVECell | 0.365 | 0.660 | 0.519 |
| PBL-N2A | **DINOCell** | **0.859** | **0.947** | **0.923** |
| | Cellpose-SAM | 0.778 | 0.899 | 0.801 |
| | SAMCell-cyto | 0.822 | 0.932 | 0.849 |
| | SAMCell-LIVECell | 0.610 | 0.699 | 0.668 |
| Glioma-C6 | **DINOCell** | **0.697** | **0.684** | **0.737** |
| | Cellpose-SAM | 0.315 | 0.387 | 0.360 |
| | SAMCell-cyto | 0.475 | 0.576 | 0.513 |
| | SAMCell-LIVECell | 0.000 | 0.000 | 0.000 |
