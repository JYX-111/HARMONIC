# -*- coding: utf-8 -*-
"""
Generate box plots for false negative rates across datasets.

This script:
- Reads predicted edges and ground truth data from output directories
- Calculates false negative rates for each sample
- Generates box plots with scatter points
- Saves results as PNG and SVG files
"""

import os
import csv
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)) or '.')

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_ROOT = "output"
DATA_ROOT = "data"
SAMPLES = ['sample0', 'sample1', 'sample2', 'sample3', 'sample4']
DATASETS = ['permissive_p00', 'permissive_p02', 'permissive_p04', 'permissive_p06', 'permissive_p08', 'permissive_p10']
OUT_DIR = "boxplots"

OURS_COLOR = "#CFEED3"
WITHOUT_HE_COLOR = "#CFE6FF"
OURS_SCATTER_COLOR = "#8FD3A0"
WITHOUT_HE_SCATTER_COLOR = "#8FB7E8"

MEDIAN_COLOR = "black"
MEDIAN_LW = 1.8
BOX_EDGE_LW = 1.2
WHISKER_LW = 1.2
CAP_LW = 1.2
SCATTER_ALPHA = 0.55
SCATTER_SIZE = 14
JITTER = 0.10
GRID_ALPHA = 0.18


def load_edges(file_path):
    """Load edges from a CSV file and return a set of (src, dst) tuples."""
    edges = set()
    if not os.path.exists(file_path):
        print(f"Warning: File not found: {file_path}")
        return edges
    
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            src = int(row['src_cell_id'])
            dst = int(row['dst_cell_id'])
            edges.add((src, dst))
    return edges


def load_false_negative_cells(cell_types_path):
    """Load false negative cells from cell_types.csv."""
    fn_cells = set()
    if not os.path.exists(cell_types_path):
        print(f"Warning: File not found: {cell_types_path}")
        return fn_cells
    
    with open(cell_types_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cell_id = int(row['cell_id'])
            is_fn = int(row['is_fp'])
            if is_fn == 1:
                fn_cells.add(cell_id)
    return fn_cells


def calculate_false_negatives_from_fn_cells(gt_edges, pred_edges, fn_cells):
    """Calculate false negatives only from false negative cells."""
    fn_edges = set()
    for src, dst in gt_edges:
        if src in fn_cells or dst in fn_cells:
            if (src, dst) not in pred_edges:
                fn_edges.add((src, dst))
    return fn_edges


def _set_nature_axes(ax):
    """Set publication-quality axes style."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.tick_params(axis="both", which="major", width=1.2, length=4.5, labelsize=12)
    ax.tick_params(axis="both", which="minor", width=1.0, length=3)
    ax.grid(True, axis="y", alpha=GRID_ALPHA, linewidth=1.0)
    ax.grid(False, axis="x")


def create_box_plot(data, labels, title, ylabel, output_png, output_svg):
    """Create and save a box plot with scatter points."""
    rng = np.random.default_rng(2)
    
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "svg.fonttype": "none",
    })
    
    fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=220)
    
    bp = ax.boxplot(
        data,
        labels=labels,
        patch_artist=True,
        showfliers=False,
        widths=0.36,
        medianprops=dict(color=MEDIAN_COLOR, linewidth=MEDIAN_LW),
        boxprops=dict(linewidth=BOX_EDGE_LW, color="black"),
        whiskerprops=dict(linewidth=WHISKER_LW, color="black"),
        capprops=dict(linewidth=CAP_LW, color="black"),
    )
    
    for patch, col in zip(bp["boxes"], [WITHOUT_HE_COLOR, OURS_COLOR]):
        patch.set_facecolor(col)
        patch.set_alpha(1.0)
    
    for i, values in enumerate(data, 1):
        x = i + rng.uniform(-JITTER, JITTER, size=values.size)
        color = WITHOUT_HE_SCATTER_COLOR if i == 1 else OURS_SCATTER_COLOR
        ax.scatter(
            x, values,
            s=SCATTER_SIZE,
            alpha=SCATTER_ALPHA,
            c=color,
            edgecolors="black",
            linewidths=0.20,
            zorder=3,
        )
    
    ax.set_title(title, pad=10)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1)
    
    _set_nature_axes(ax)
    
    fig.tight_layout(pad=0.6)
    fig.savefig(output_png, dpi=600)
    fig.savefig(output_svg)
    plt.close(fig)
    
    print("[saved]", output_png)
    print("[saved]", output_svg)


def process_dataset(dataset):
    """Process a single dataset and generate false negative rate plot."""
    print(f"\nProcessing dataset: {dataset}")
    print("=" * 80)
    
    without_fnr = []
    ours_fnr = []
    
    gt_dir = os.path.join(DATA_ROOT, dataset, "simulation_data")
    strength = dataset[-2:]
    
    without_dir = os.path.join(OUTPUT_ROOT, f"without_texture_{strength}")
    ours_dir = os.path.join(OUTPUT_ROOT, f"ours_{strength}")
    
    for sample in SAMPLES:
        print(f"\nSample: {sample}")
        
        gt_file = os.path.join(gt_dir, sample, 'pred_edges.csv')
        gt_edges = load_edges(gt_file)
        
        cell_types_file = os.path.join(gt_dir, sample, 'cell_types.csv')
        fn_cells = load_false_negative_cells(cell_types_file)
        
        without_file = os.path.join(without_dir, sample, 'pred_edges.csv')
        without_edges = load_edges(without_file)
        
        ours_file = os.path.join(ours_dir, sample, 'pred_edges.csv')
        ours_edges = load_edges(ours_file)
        
        fn_without = calculate_false_negatives_from_fn_cells(gt_edges, without_edges, fn_cells)
        fn_ours = calculate_false_negatives_from_fn_cells(gt_edges, ours_edges, fn_cells)
        
        total_fn_gt = 0
        for src, dst in gt_edges:
            if src in fn_cells or dst in fn_cells:
                total_fn_gt += 1
        
        if total_fn_gt > 0:
            fnr_without = len(fn_without) / total_fn_gt
            fnr_ours = len(fn_ours) / total_fn_gt
        else:
            fnr_without = 1.0
            fnr_ours = 1.0
        
        without_fnr.append(fnr_without)
        ours_fnr.append(fnr_ours)
        
        print(f"False negative rate (without_CMI): {fnr_without:.4f}")
        print(f"False negative rate (ours): {fnr_ours:.4f}")
    
    without_fnr = np.array(without_fnr)
    ours_fnr = np.array(ours_fnr)
    
    def calculate_stats(values):
        mean = np.mean(values)
        std = np.std(values)
        n = len(values)
        ci = 1.96 * (std / np.sqrt(n))
        return mean, ci
    
    without_mean, without_ci = calculate_stats(without_fnr)
    ours_mean, ours_ci = calculate_stats(ours_fnr)
    
    print("\n" + "=" * 80)
    print("Statistics for dataset:", dataset)
    print("=" * 80)
    without_ci_lower = max(0, without_mean - without_ci)
    ours_ci_lower = max(0, ours_mean - ours_ci)
    print(f"without_CMI: Mean = {without_mean:.4f}, 95% CI = [{without_ci_lower:.4f}, {without_mean + without_ci:.4f}]")
    print(f"ours: Mean = {ours_mean:.4f}, 95% CI = [{ours_ci_lower:.4f}, {ours_mean + ours_ci:.4f}]")
    print("=" * 80)
    
    out_png = os.path.join(OUT_DIR, f"boxplot_false_negative_{dataset}.png")
    out_svg = os.path.join(OUT_DIR, f"boxplot_false_negative_{dataset}.svg")
    
    create_box_plot(
        [without_fnr, ours_fnr],
        ["without_CMI", "ours"],
        f"False Negative Rate ({dataset})",
        "Rate (0-1)",
        out_png,
        out_svg
    )


def main():
    """Main entry point."""
    os.makedirs(OUT_DIR, exist_ok=True)
    
    print("Calculating false negative rates for all datasets...")
    print("=" * 80)
    
    for dataset in DATASETS:
        process_dataset(dataset)
    
    print("\nAll datasets processed!")


if __name__ == "__main__":
    main()
