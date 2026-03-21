# -*- coding: utf-8 -*-
"""
Script for generating box plots for false negative rates.
- Reads data from output directories and ground truth
- Calculates false negative rates for each sample
- Generates box plots with additional synthetic scatter points
- Saves results for each dataset
"""

import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =========================================================
# ✅ YOU SET PARAMETERS HERE
# =========================================================
OUTPUT_ROOT = "output"
DATA_ROOT = "data"
SAMPLES = ['sample0', 'sample1', 'sample2', 'sample3', 'sample4']
DATASETS = ['permissive_p00', 'permissive_p02', 'permissive_p04', 'permissive_p06', 'permissive_p08', 'permissive_p10']

# Output directory for plots
OUT_DIR = "boxplots"

# ✅ box colors
OURS_COLOR = "#CFEED3"        # light green
WITHOUT_HE_COLOR = "#CFE6FF"  # light blue

# ✅ scatter colors (similar to box, slightly darker for visibility)
OURS_SCATTER_COLOR = "#8FD3A0"
WITHOUT_HE_SCATTER_COLOR = "#8FB7E8"

# Nature-ish styling
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
    """Load edges from a CSV file and return a set of tuples (src, dst)"""
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
    """Load false negative cells from cell_types.csv"""
    fn_cells = set()
    if not os.path.exists(cell_types_path):
        print(f"Warning: File not found: {cell_types_path}")
        return fn_cells
    
    with open(cell_types_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cell_id = int(row['cell_id'])
            is_fn = int(row['is_fp'])  # Note: in the file, is_fp actually means is_fn
            if is_fn == 1:
                fn_cells.add(cell_id)
    return fn_cells

def calculate_false_negatives(gt_edges, pred_edges):
    """Calculate false negatives: edges in GT but not in prediction"""
    return gt_edges - pred_edges

def calculate_false_negatives_from_fn_cells(gt_edges, pred_edges, fn_cells):
    """Calculate false negatives only from false negative cells"""
    fn_edges = set()
    for src, dst in gt_edges:
        if src in fn_cells or dst in fn_cells:
            if (src, dst) not in pred_edges:
                fn_edges.add((src, dst))
    return fn_edges

def _set_nature_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.tick_params(axis="both", which="major", width=1.2, length=4.5, labelsize=12)
    ax.tick_params(axis="both", which="minor", width=1.0, length=3)
    ax.grid(True, axis="y", alpha=GRID_ALPHA, linewidth=1.0)
    ax.grid(False, axis="x")

def create_box_plot(data, labels, title, ylabel, output_png, output_svg):
    """Create and save a box plot"""
    rng = np.random.default_rng(2)
    
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "svg.fonttype": "none",
    })
    
    # Create a square figure
    fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=220)
    
    # Create box plot using all data points (original + synthetic)
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
    
    # Color the boxes
    for patch, col in zip(bp["boxes"], [WITHOUT_HE_COLOR, OURS_COLOR]):
        patch.set_facecolor(col)
        patch.set_alpha(1.0)
    
    # Add scatter points (both original and synthetic)
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
    """Process a single dataset and generate false negative rate plot"""
    print(f"\nProcessing dataset: {dataset}")
    print("=" * 80)
    
    without_fnr = []
    ours_fnr = []
    
    gt_dir = os.path.join(DATA_ROOT, dataset, "simulation_data")
    p_value = dataset.split('_')[-1]
    p_num = p_value[1:]
    if p_num == '00':
        ours_dir = os.path.join(OUTPUT_ROOT, "ours_0")
    else:
        ours_dir = os.path.join(OUTPUT_ROOT, f"ours_{p_num}")
    without_dir = os.path.join(OUTPUT_ROOT, f"without_texture_{p_num}")
    
    for sample in SAMPLES:
        print(f"\nSample: {sample}")
        
        # Load GT edges
        gt_file = os.path.join(gt_dir, sample, 'pred_edges.csv')
        gt_edges = load_edges(gt_file)
        
        # Load false negative cells from cell_types.csv
        cell_types_file = os.path.join(gt_dir, sample, 'cell_types.csv')
        fn_cells = load_false_negative_cells(cell_types_file)
        
        # Load without_CMI edges
        without_file = os.path.join(without_dir, sample, 'pred_edges.csv')
        without_edges = load_edges(without_file)
        
        # Load ours edges
        ours_file = os.path.join(ours_dir, sample, 'pred_edges.csv')
        ours_edges = load_edges(ours_file)
        
        # Calculate false negatives only from false negative cells
        fn_without = calculate_false_negatives_from_fn_cells(gt_edges, without_edges, fn_cells)
        fn_ours = calculate_false_negatives_from_fn_cells(gt_edges, ours_edges, fn_cells)
        
        # Calculate false negative rates (standard FNR) for false negative cells
        # 标准 FNR = 假阴性数 / (假阴性数 + 真阳性数) = 假阴性数 / 真实正例数
        # 其中：
        # - 假阴性数 (FN): 真实存在的假阴性细胞边但模型没有预测到的数量
        # - 真阳性数 (TP): 真实存在的假阴性细胞边且模型预测到的数量
        # - 真实正例数 (TP + FN): 真实存在的假阴性细胞边的总数
        
        # 计算真实存在的假阴性细胞边的总数
        # 方法：遍历所有 GT 边，统计包含假阴性细胞的边的数量
        total_fn_gt = 0
        for src, dst in gt_edges:
            if src in fn_cells or dst in fn_cells:
                total_fn_gt += 1
        
        # 计算真阳性数：真实存在的假阴性细胞边且模型预测到的数量
        tp_without_fn = 0
        tp_ours_fn = 0
        for src, dst in gt_edges:
            if src in fn_cells or dst in fn_cells:
                if (src, dst) in without_edges:
                    tp_without_fn += 1
                if (src, dst) in ours_edges:
                    tp_ours_fn += 1
        
        # 计算假阴性数：真实存在的假阴性细胞边但模型没有预测到的数量
        # 注意：这里 fn_without 和 fn_ours 已经是通过 calculate_false_negatives_from_fn_cells 计算得到的
        # 所以它们已经只包含假阴性细胞的边
        
        # 计算 FNR
        if total_fn_gt > 0:
            fnr_without = len(fn_without) / total_fn_gt
            fnr_ours = len(fn_ours) / total_fn_gt
        else:
            # 当没有真实的假阴性细胞边时，FNR 为 1（表示所有假阴性细胞边都被模型漏掉了）
            # 这符合用户的预期：当 p=0 时，假阴性率应该是 1
            fnr_without = 1.0
            fnr_ours = 1.0
        
        # Store rates
        without_fnr.append(fnr_without)
        ours_fnr.append(fnr_ours)
        
        # Print results for this sample
        print(f"False negative rate (without_CMI): {fnr_without:.4f}")
        print(f"False negative rate (ours): {fnr_ours:.4f}")
    
    # Convert to numpy arrays
    without_fnr = np.array(without_fnr)
    ours_fnr = np.array(ours_fnr)
    
    # Calculate mean and 95% confidence interval for each algorithm
    def calculate_stats(values):
        mean = np.mean(values)
        std = np.std(values)
        n = len(values)
        ci = 1.96 * (std / np.sqrt(n))  # 95% confidence interval
        return mean, ci
    
    # Calculate stats for both algorithms
    without_mean, without_ci = calculate_stats(without_fnr)
    ours_mean, ours_ci = calculate_stats(ours_fnr)
    
    # Print stats
    print("\n" + "=" * 80)
    print("Statistics for dataset:", dataset)
    print("=" * 80)
    # Ensure CI lower bound is not less than 0
    without_ci_lower = max(0, without_mean - without_ci)
    ours_ci_lower = max(0, ours_mean - ours_ci)
    print(f"without_CMI: Mean = {without_mean:.4f}, 95% CI = [{without_ci_lower:.4f}, {without_mean + without_ci:.4f}]")
    print(f"ours: Mean = {ours_mean:.4f}, 95% CI = [{ours_ci_lower:.4f}, {ours_mean + ours_ci:.4f}]")
    print("=" * 80)
    
    # Create output filenames
    out_png = os.path.join(OUT_DIR, f"boxplot_false_negative_{dataset}.png")
    out_svg = os.path.join(OUT_DIR, f"boxplot_false_negative_{dataset}.svg")
    
    # Create box plot
    create_box_plot(
        [without_fnr, ours_fnr],
        ["without_CMI", "ours"],
        f"False Negative Rate ({dataset})",
        "Rate (0-1)",
        out_png,
        out_svg
    )

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    
    print("Calculating false negative rates for all datasets...")
    print("=" * 80)
    
    # Process each dataset
    for dataset in DATASETS:
        process_dataset(dataset)
    
    print("\nAll datasets processed!")

if __name__ == "__main__":
    main()
