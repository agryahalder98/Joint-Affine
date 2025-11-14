## AGRYA HALDER, ED25D900
# CT-MRI Brain Image Registration

A deep learning-based medical image registration system that aligns CT and MRI brain scans using affine transformation and B-spline deformable registration optimized with mutual information.

## Overview

This project implements a two-phase registration pipeline:
1. **Phase 1**: Affine-only registration (global alignment)
2. **Phase 2**: Joint affine + B-spline deformable registration (local refinement)

The system uses **Mutual Information (MI)** as the similarity metric to handle multi-modal image registration between CT and MRI scans.

## Features

- Affine transformation with learnable parameters
- B-spline deformable registration via control point grid
- Mutual Information (MI) optimization for multi-modal alignment
- Automatic train/validation/test split (70%/15%/15%)
- Optional tumor image inclusion
- Real-time training metrics (loss, accuracy)
- Target Registration Error (TRE) evaluation
- Overlay alignment accuracy measurement
- Visual overlay generation for results
- Optimized for speed with GPU acceleration

## Requirements

```bash
pip install torch torchvision numpy pillow matplotlib
```

**Minimum Requirements:**
- Python 3.7+
- PyTorch 1.7+
- CUDA-capable GPU (recommended) or CPU
- 4GB RAM minimum
- 2GB disk space for dataset

## Dataset Structure

Organize your data as follows:

```
Dataset/
├── Brain Tumor CT scan Images/
│   ├── Healthy/
│   │   ├── image(1).jpg
│   │   ├── image(2).jpg
│   │   └── ...
│   └── Tumor/
│       ├── image(1).jpg
│       ├── image(2).jpg
│       └── ...
└── Brain Tumor MRI images/
    ├── Healthy/
    │   ├── image(1).jpg
    │   ├── image(2).jpg
    │   └── ...
    └── Tumor/
        ├── image(1).jpg
        ├── image(2).jpg
        └── ...
```

**Important:** 
- Images must be named with indices in parentheses: `image(1).jpg`, `image(2).jpg`, etc.
- CT and MRI pairs must have matching indices
- Supported format: JPG/JPEG

## Configuration

Edit these parameters in the code to customize:

```python
# Data paths
DATA_ROOT = "/path/to/your/dataset"

# Image settings
TARGET_SIZE = (128, 128)  # Image resolution (H, W)
BATCH_SIZE = 16           # Batch size for training
INCLUDE_TUMOR = False     # Set True to include tumor images

# Training parameters
NUM_EPOCHS_AFFINE = 30    # Epochs for affine-only phase
NUM_EPOCHS_JOINT = 60     # Epochs for joint phase
LR_AFFINE = 2e-3          # Learning rate for affine
LR_JOINT = 5e-4           # Learning rate for joint

# Model parameters
CONTROL_SHAPE = (6, 6)    # B-spline control grid size
MI_BINS = 32              # Histogram bins for MI calculation
MI_SIGMA = 0.03           # Gaussian kernel width for MI

# Hardware
DEVICE = 'cuda'           # 'cuda' for GPU, 'cpu' for CPU
NUM_WORKERS = 2           # Parallel data loading threads
```

## Usage

### Basic Training

```python
# Run the complete training pipeline
python ct_mri_registration.py
```

### Include Tumor Images

```python
# In the code, set:
INCLUDE_TUMOR = True
```

### Custom Dataset Path

```python
# Modify the DATA_ROOT variable:
DATA_ROOT = "/your/custom/path/Dataset"
```

## Training Process

The system will:

1. **Load and split dataset** into train/validation/test sets
2. **Phase 1 (Affine)**: Train affine transformation for global alignment
3. **Phase 2 (Joint)**: Train affine + deformable for local refinement
4. **Evaluate on test set**: Report final metrics
5. **Save models**: Export trained models to `.pth` file
6. **Generate visualizations**: Create overlay images showing results


## Evaluation Metrics

### 1. Registration Loss
- **Mutual Information (MI)**: Measures statistical dependence between CT and MRI
- **Lower is better** (we minimize negative MI)
- Regularization term prevents excessive deformation

### 2. Overlay Alignment Accuracy
- Percentage of pixels with intensity difference < threshold (0.1)
- **Higher is better** (0-100%)
- Indicates how well images align

### 3. Target Registration Error (TRE)
- Measures spatial alignment accuracy
- Based on test point grid
- **Lower is better**

## Model Architecture

### Affine Transformation
- 6 learnable parameters (2×3 affine matrix)
- Initialized to identity transformation
- Optimized with Adam optimizer

### B-spline Deformable Registration
- Control point grid (default: 6×6)
- Bilinear upsampling to dense displacement field
- Pixel-level deformation with smooth interpolation

### Loss Function
```
Total Loss = -MI(fixed, warped) + λ * ||deformation||²
```
- MI term: Encourages alignment
- Regularization (λ=1e-3): Prevents unrealistic deformations


If you use this code in your research, please cite:

```bibtex
@software{ct_mri_registration,
  title={CT-MRI Brain Image Registration with Mutual Information},
  author={Agrya Halder},
  year={2024},
  description={Deep learning-based multi-modal medical image registration}
}
```

## References

- Mutual Information for medical image registration: Mattes et al. (2003)
- B-spline deformable registration: Rueckert et al. (1999)
- Affine transformations in PyTorch: torch.nn.functional.affine_grid

## License

This project is for educational and research purposes.

## Support

For issues or questions:
- Check the Troubleshooting section above
- Review parameter configurations
- Ensure dataset is properly formatted
- Verify PyTorch and CUDA installation

## Acknowledgments

Built with PyTorch for efficient GPU-accelerated training and registration.


**Version:** 1.0  
**Last Updated:** November 2024  
**Status:** Optimized for speed and accuracy
