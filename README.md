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

## Data Preparation

### Directory Structure

Organize your data following this structure:

```text
data_root/
  lr_pairs.csv
  └── {sample_id}/
      ├── HE.npy
      ├── mask.npy
      ├── *_data.csv
      ├── *_meta.csv
      └── HE_global.npy(Possible)
```

### Required Files

**At dataset root:**
1. `lr_pairs.csv` — ligand–receptor pair dataset

**For each `{sample_id}/`:**
1. `HE.npy` — H&E image array
2. `mask.npy` — segmentation mask array  
3. `*_data.csv` — gene expression table
4. `*_meta.csv` — cell possition table
5. `HE_global.npy` — optional global H&E context array (if available)

### Data Preprocessing

Data should be log-normalized with gene-wise scaling. For H&E feature extraction, please refer to:

- [Prov-GigaPath](https://github.com/prov-gigapath/prov-gigapath)

## Usage

### Running

Adjust the key parameters in your config file and run the main entrypoint:

```yaml
# Example key parameters (edit in your config YAML)
seed: 7
device: cuda:0
R: 1
output:
    out_dir: /path/to/outputs/
dataset:
    dataset_id: dataset_id
    root: /path/to/data/
    sample_id: "sample_id"
    active_percentile: 98
causal:
    enabled: true
    tile_size_px: 1536
    global_he_path: "/path/to/HE_global.npy/" # optional: pre-built global HE mosaic (.npy). If empty, will stitch neighbors.
    patch_size_px: 100
    local_um: 250.0
    global_um: 1000.0
    node_patch_um: 80.0
train:
    epochs: 400
    lr: 0.001
    weight_decay: 0.0001
    grad_clip: 5.0
    print_every: 25
```

Launch running:
```launch
python main.py
```

### Output Format
Outputs are saved to OUT_ROOT/dataset_id/sample_id/:
- pred_edges.csv: Filtered results including src_cell_id、dst_cell_id、ligand、receptor、pathway、mechanism and score.

## Citation
If you use FOCUS in your research, please cite:
``` citation
@article{wang2026histology,
  title={Histology-Aware Graph for Modeling Intercellular Communication in Spatial Transcriptomics},
  author={Wang, Xiaofei and Tao, Chenyang and Jiang, Yixuan and Liu, Hanyu and Jiang, Zheng and Zhu, Pinan and Que, Ningfeng and Xi, Jianzhong and Price, Stephen and Mou, Yonggao and others},
  journal={bioRxiv},
  pages={2026--01},
  year={2026},
  publisher={Cold Spring Harbor Laboratory}
}
```

## Acknowledgments
HARMONIC builds upon several excellent open-source projects:
- GigaPath: [prov-gigapath/prov-gigapath](https://www.bing.com/search?q=prov-gigapath%2Fprov-gigapath&cvid=250c4ab19c454273bab25f880cc85e1a&gs_lcrp=EgRlZGdlKgYIABBFGDkyBggAEEUYOTIGCAEQRRg6MgYIAhBFGDwyBwgDEOsHGECoAgCwAgA&FORM=ANAB01&adppc=EDGEESS&PC=ASTS&mkt=zh-CN)
