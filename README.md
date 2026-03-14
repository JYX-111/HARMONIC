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

### Systhetic Data and Validation

#### Systhetic Data

We constructed a synthetic spatial transcriptomics benchmark with paired H&E-like images to evaluate cell–cell communication (CCC) inference under controlled settings. A total of 20,000 cells were sampled and assigned to one of three cell categories: sender cells, receiver cells, and non-interacting cells. Sender cells were placed at the centers of multiple circular niches, receiver cells were arranged as concentric rings around the sender populations, and non-interacting cells filled the remaining background tissue regions.

Gene expression was simulated for 50 genes in total, including 10 ligand genes, 10 receptor genes, and 40 noise genes. Baseline expression values were initialized from Gaussian noise. True ligand genes were activated in sender cells, and receiver cells within a spatial threshold of 50 pixels were allowed to interact with nearby sender cells. Once a sender cell was selected as an interacting partner, the corresponding receptor gene in the receiver cell was upregulated. For true ligand–receptor pairs, the expression effect size was sampled from `U(10, 20)`. In parallel, we generated a paired RGB H&E-like image with the same spatial layout, where sender cells, receiver cells, and non-interacting cells were represented by distinct colors to encode microenvironmental morphology.

- Base Simulation Setting
  The base simulation defines the canonical CCC setting without perturbation. Sender cells occupy niche centers, receiver cells form surrounding rings, and nearby sender–receiver pairs constitute candidate communication events. The synthetic H&E-like image preserves the same spatial organization as the simulated ST grid,          enabling joint evaluation of transcriptional and morphological information. This setting serves as the reference dataset for all downstream benchmark experiments.

- False-Positive Benchmark: Barrier-Blocked Spurious Communication
  To simulate false-positive CCC signals caused by spatial proximity but suppressed by microenvironmental barriers, we introduced barrier structures between sender and receiver regions in the H&E-like image. These barriers were visualized as dark boundaries, and their strength was sampled from `U(0, 1)`. As barrier strength       decreased, the barrier gradually faded and its inhibitory effect weakened.
  Sender cells were still allowed to exhibit elevated expression of CCC-related ligand genes, but stronger barriers reduced both the probability and the effective strength of communication reaching receiver cells. In addition, barrier strength exerted a nonlinear modulation on receptor expression in nearby receiver cells. The     final receptor expression was modeled as:

  `G_receiver = f1(G_sender) + β × f2(Strength_barrier) / d + noise`

  where `d` denotes the spatial distance between sender and receiver cells.

- False-Negative Benchmark: Communication Loss Under Weak Transcriptional Signal
  To simulate false-negative CCC events caused by weak molecular evidence, sender and receiver cells were each divided into two subclasses: normal-expression cells and low-expression cells. Both subclasses remained capable of communication, but low-expression cells carried weaker transcriptional signals.
  For normal-expression cells, highly expressed genes in true ligand–receptor pairs were sampled from `U(10, 20)`. For low-expression cells, the corresponding activated genes were sampled from `U(3, 7)`. In the paired H&E-like image, low-expression cells were displayed with reduced color intensity to reflect weakened molecular    activity.
  We further introduced permissive scenarios with different strengths by adding grid-like texture patterns to the H&E-like image. The size and density of the texture varied across settings, providing additional microenvironmental cues. Grid density also exerted a nonlinear regulatory effect on receptor expression in receiver      cells. The final receptor expression was modeled as:
  `G_receiver = f1(G_sender) + β × f2(Grid_density) + noise`

#### Validation

Based on the simulated datasets, we generated the corresponding ground-truth labels. For the False-Positive Benchmark and False-Negative Benchmark, we evaluated FPR and FNR, respectively. A total of 20 samples were used for evaluation, and the results were summarized using box plots.

#### Tutorial

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

### HARMONIC on one example

&nbsp;&nbsp;For the **Human Ovary Cancer** region shown in **Figure 3** of our paper, we provide one corresponding HARMONIC model weight file. <br>
&nbsp;&nbsp;The file is stored in Google One and can be accessed from the following link:[https://drive.google.com/drive/folders/1UAy6AT9znP8tWUr6OXLcdKpDKMHU0Xfm?usp=sharing].

## Citation
If you use HARMONIC in your research, please cite:
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
- GigaPath: [prov-gigapath/prov-gigapath](https://github.com/prov-gigapath/prov-gigapath)

## Contact
For questions and feedback:
- Open an issue on GitHub
- Email: xw405@cam.ac.uk

---

**Note:** This is research software. While we strive for correctness, please validate results for your specific application.  
Contributions and feedback are welcome!
