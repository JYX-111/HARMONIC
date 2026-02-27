# HARMONIC

![python](https://img.shields.io/badge/python-3.10%2B-blue)
![pytorch](https://img.shields.io/badge/PyTorch-2.9.0-orange)

---

## Overview

Cell-cell communication (CCC) is essential to how life forms and functions. Recent tools achieve single-cell-resolved CCC inference utilizing spatial transcriptomics (ST). However, most ignore the modeling of tissue contexts surrounding cells, causing high false-positive/negative rates. Here, we propose HARMONIC, a CCC inference method integrating multimodal ST and hematoxylin and eosin (H&E)-stained images. HARMONIC causally modeling the transcriptomic-to-contextual relationships for CCC inference. The state-of-the-art performance was verified across ST platforms, species and healthy/diseased status, on both synthetic and biological samples. HARMONIC was applied in various real-world scenarios, especially on tissues with clear morphological boundaries, including cortical layers in mouse brain, medullary-cortex structures in mouse kidney, as well as tumor-stromal/immune interface. Significant refinement of false-positive/negative predictions was observed compared to ST-only CCC tools.

---

## Installation

### Environment Setup

Create a conda environment using the provided configuration:

```bash
conda env create -f environment.yaml
conda activate ccc
```

### Requirements

- Python ≥ 3.10
- PyTorch 2.9.0 (CUDA 12.6)
- Key dependencies:
  - `numpy`, `scipy`, `pandas` — data processing / numerics
  - `pyyaml` — configuration management
  - `scikit-image` — morphology / image feature extraction
  - `timm`, `huggingface-hub` — pretrained vision encoder loading
  - `pillow`, `imageio`, `tifffile` — image I/O
