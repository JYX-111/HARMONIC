# -*- coding: utf-8 -*-
"""Script for generating box plots for false positive rates.
Reads data from output directories and ground truth, calculates false positive rates
for each sample, and generates square box plots with real scatter points.
"""

import os
import csv
import re
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Configuration parameters
OUT_BASE = "output"
DATA_ROOT = "data"
SAMPLES = ['sample0', 'sample1', 'sample2', 'sample3', 'sample4']

# Output directory for plots
OUT_DIR = "boxplots"

# Box plot colors
OURS_COLOR = "#CFEED3"
WITHOUT_HE_COLOR = "#CFE6FF"

# Scatter colors (similar to box, slightly darker for visibility)
OURS_SCATTER_COLOR = "#8FD3A0"
WITHOUT_HE_SCATTER_COLOR = "#8FB7E8"

# Nature-style plot settings
MEDIAN_COLOR = "black"
MEDIAN_LW = 1.8
BOX_EDGE_LW = 1.2
WHISKER_LW = 1.2
CAP_LW = 1.2
SCATTER_ALPHA = 0.55
SCATTER_SIZE = 14
JITTER = 0.10
GRID_ALPHA = 0.18

def load_edges(file_path, *, strict=True):
    """Load edges from a CSV file and return a set of tuples (src, dst)."""
    edges = set()
    if not os.path.exists(file_path):
        if strict:
            sample_dir = os.path.dirname(file_path)
            err_path = os.path.join(sample_dir, "error.txt")
            if os.path.exists(err_path):
                try:
                    with open(err_path, "r") as f:
                        err_text = f.read().strip()
                    if err_text:
                        print(f"Found error.txt for missing output:\n{err_text}")
                except Exception:
                    pass
            raise FileNotFoundError(file_path)
        print(f"Warning: File not found: {file_path}")
        return set()
    
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            src = int(row['src_cell_id'])
            dst = int(row['dst_cell_id'])
            if src == dst:
                continue
            edges.add((src, dst))
    return edges

def calculate_false_positives(edges1, edges2):
    """Calculate false positives: edges in edges1 but not in edges2."""
    return edges1 - edges2

def _set_nature_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.tick_params(axis="both", which="major", width=1.2, length=4.5, labelsize=12)
    ax.tick_params(axis="both", which="minor", width=1.0, length=3)
    ax.grid(True, axis="y", alpha=GRID_ALPHA, linewidth=1.0)
    ax.grid(False, axis="x")

def calculate_statistics(data, labels):
    """Calculate mean and 95% confidence interval for each algorithm."""
    t_crit_975 = {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
        21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
    }
    statistics = {}
    for i, (values, label) in enumerate(zip(data, labels)):
        mean = np.mean(values)
        n = len(values)
        if n < 2:
            ci_lower = float(mean)
            ci_upper = float(mean)
        else:
            std_err = np.std(values, ddof=1) / np.sqrt(n)
            t_value = t_crit_975.get(int(n - 1), 1.96)
            ci_lower = mean - t_value * std_err
            ci_upper = mean + t_value * std_err
        # Clamp to [0, 1] range
        ci_lower = max(0, ci_lower)
        ci_upper = min(1, ci_upper)
        statistics[label] = {
            "mean": mean,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper
        }
    return statistics

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
    
    # Create a square figure
    fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=220)
    
    # Add scatter points first
    for i, values in enumerate(data, 1):
        # Generate jittered x positions
        x = i + rng.uniform(-JITTER, JITTER, size=values.size)
        color = WITHOUT_HE_SCATTER_COLOR if i == 1 else OURS_SCATTER_COLOR
        ax.scatter(
            x, values,
            s=SCATTER_SIZE,
            alpha=SCATTER_ALPHA,
            c=color,
            edgecolors="black",
            linewidths=0.20,
            zorder=1,
        )
    
    # Create box plot
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
        patch.set_alpha(0.7)  # Slightly transparent to see scatter points
        patch.set_zorder(2)
    
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

def _discover_strengths(data_root):
    strengths = []
    if not os.path.isdir(data_root):
        raise NotADirectoryError(data_root)
    for name in os.listdir(data_root):
        m = re.fullmatch(r"barrier_s(\d+)", name)
        if m:
            strengths.append(m.group(1).zfill(2))
    strengths = sorted(set(strengths), key=lambda x: int(x))
    return strengths

def _resolve_without_dir(out_base, strength):
    candidates = [
        os.path.join(out_base, f"without_texture_{strength}"),
    ]
    for p in candidates:
        if os.path.isdir(p):
            return p
    return candidates[0]

def _count_cells(cell_types_csv):
    """Count number of cells in cell_types.csv file."""
    if not os.path.exists(cell_types_csv):
        raise FileNotFoundError(cell_types_csv)
    n = 0
    with open(cell_types_csv, "r") as f:
        reader = csv.DictReader(f)
        for _ in reader:
            n += 1
    return n

def _check_one_strength(*, strength, data_root, out_base, samples):
    gt_base = os.path.join(data_root, f"barrier_s{strength}", "simulation_data")
    gt0_base = os.path.join(data_root, "barrier_s00", "simulation_data")
    without_dir = _resolve_without_dir(out_base, strength)
    
    # Handle inconsistent naming: ours_0 vs ours_02, ours_04, etc.
    ours_dir = os.path.join(out_base, f"ours_{strength}")
    if not os.path.isdir(ours_dir) and strength == "00":
        ours_dir = os.path.join(out_base, "ours_0")
    
    missing = []
    for p in (gt_base, gt0_base, without_dir, ours_dir):
        if not os.path.isdir(p):
            missing.append(("dir", p))
    for sample in samples:
        gt0_file = os.path.join(gt0_base, sample, 'pred_edges.csv')
        cell_types0 = os.path.join(gt0_base, sample, 'cell_types.csv')
        gt_file = os.path.join(gt_base, sample, 'pred_edges.csv')
        without_file = os.path.join(without_dir, sample, 'pred_edges.csv')
        ours_file = os.path.join(ours_dir, sample, 'pred_edges.csv')
        for fp in (gt0_file, cell_types0, gt_file, without_file, ours_file):
            if not os.path.exists(fp):
                missing.append(("file", fp))
    if len(missing) == 0:
        print(f"s{strength}: OK")
        return True
    print(f"s{strength}: missing {len(missing)} paths")
    for kind, path in missing:
        print(f"  [{kind}] {path}")
    return False

def _evaluate_one_strength(*, strength, data_root, out_base, out_dir, samples, strict=True):
    gt_base = os.path.join(data_root, f"barrier_s{strength}", "simulation_data")
    gt0_base = os.path.join(data_root, "barrier_s00", "simulation_data")
    without_dir = _resolve_without_dir(out_base, strength)
    
    # Handle inconsistent naming: ours_0 vs ours_02, ours_04, etc.
    ours_dir = os.path.join(out_base, f"ours_{strength}")
    if not os.path.isdir(ours_dir) and strength == "00":
        ours_dir = os.path.join(out_base, "ours_0")
    
    for p in (gt_base, gt0_base, without_dir, ours_dir):
        if not os.path.isdir(p):
            raise NotADirectoryError(p)

    print(f"\nBarrier strength: s{strength} (p{strength})")
    print("=" * 80)
    
    print("Calculating false positive rates...")

    if strict:
        ok = _check_one_strength(
            strength=strength,
            data_root=data_root,
            out_base=out_base,
            samples=samples,
        )
        if not ok:
            raise FileNotFoundError(f"Missing required files for s{strength}. Run with --check-only for details.")
    
    # Store rates for each sample
    without_fpr = []
    ours_fpr = []
    
    for sample in samples:
        print(f"  Sample: {sample}")
        
        # Load GT edges
        gt0_file = os.path.join(gt0_base, sample, 'pred_edges.csv')
        cell_types0 = os.path.join(gt0_base, sample, 'cell_types.csv')
        gt_file = os.path.join(gt_base, sample, 'pred_edges.csv')
        without_file = os.path.join(without_dir, sample, 'pred_edges.csv')
        ours_file = os.path.join(ours_dir, sample, 'pred_edges.csv')
        try:
            gt0_edges = load_edges(gt0_file, strict=strict)
            n_cells0 = _count_cells(cell_types0)
            gt_edges = load_edges(gt_file, strict=strict)
            without_edges = load_edges(without_file, strict=strict)
            ours_edges = load_edges(ours_file, strict=strict)
        except FileNotFoundError as e:
            if strict:
                raise
            print(f"  Skipping sample due to missing file: {e}")
            continue
        
        # Calculate false positives: edges in prediction but not in GT
        fp_without = calculate_false_positives(without_edges, gt_edges)
        fp_ours = calculate_false_positives(ours_edges, gt_edges)
        
        # Calculate FPR = FP / (ours预测的edge数量 + 对比算法的预测的edge数量)
        # For without_CMI: FPR = FP_without / (len(ours_edges) + len(without_edges))
        # For ours: FPR = FP_ours / (len(ours_edges) + len(without_edges))
        total_pred_edges = len(ours_edges) + len(without_edges)
        
        if total_pred_edges > 0:
            fpr_without = len(fp_without) / total_pred_edges
            fpr_ours = len(fp_ours) / total_pred_edges
        else:
            fpr_without = 0.0
            fpr_ours = 0.0

        # Store rates
        without_fpr.append(fpr_without)
        ours_fpr.append(fpr_ours)
        
        # Print results for this sample
        print(f"    False positive rate (without_CMI): {fpr_without:.4f} (FP={len(fp_without)}, total_pred={total_pred_edges})")
        print(f"    False positive rate (ours): {fpr_ours:.4f} (FP={len(fp_ours)}, total_pred={total_pred_edges})")
    
    if len(without_fpr) == 0:
        print("No valid samples to evaluate for this strength.")
        return

    # Convert to numpy arrays
    without_fpr = np.array(without_fpr)
    ours_fpr = np.array(ours_fpr)
    
    # Calculate and print statistics (mean and 95% CI)
    print(f"\nCalculating statistics...")
    data = [without_fpr, ours_fpr]
    labels = ["without_CMI", "ours"]
    stats = calculate_statistics(data, labels)
    
    for label, values in stats.items():
        print(f"{label}:")
        print(f"  Mean: {values['mean']:.4f}")
        print(f"  95% CI: [{values['ci_lower']:.4f}, {values['ci_upper']:.4f}]")
    
    # Create box plot
    print(f"\nGenerating box plot...")
    
    os.makedirs(out_dir, exist_ok=True)
    output_png = os.path.join(out_dir, f"boxplot_false_positive_s{strength}.png")
    output_svg = os.path.join(out_dir, f"boxplot_false_positive_s{strength}.svg")
    
    create_box_plot(
        [without_fpr, ours_fpr],
        ["without_CMI", "ours"],
        f"False Positive Rate (FP/(ours+without)) (s{strength})",
        "Rate (0-1)",
        output_png,
        output_svg
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=DATA_ROOT)
    parser.add_argument("--out-base", default=OUT_BASE)
    parser.add_argument("--out-dir", default=OUT_DIR)
    parser.add_argument("--strengths", default="auto")
    parser.add_argument("--samples", default=",".join(SAMPLES))
    parser.add_argument("--skip-missing", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    data_root = args.data_root
    out_base = args.out_base
    out_dir = args.out_dir
    strict = not args.skip_missing
    samples = [s for s in args.samples.split(",") if s]

    if args.strengths.strip().lower() == "auto":
        strengths = _discover_strengths(data_root)
    else:
        strengths = [s.strip().zfill(2) for s in args.strengths.split(",") if s.strip()]
        strengths = sorted(set(strengths), key=lambda x: int(x))

    if len(strengths) == 0:
        raise ValueError("No strengths found/provided.")

    if args.check_only:
        all_ok = True
        for strength in strengths:
            ok = _check_one_strength(
                strength=strength,
                data_root=data_root,
                out_base=out_base,
                samples=samples,
            )
            all_ok = all_ok and ok
        if not all_ok:
            raise SystemExit(2)
        return

    for strength in strengths:
        try:
            _evaluate_one_strength(
                strength=strength,
                data_root=data_root,
                out_base=out_base,
                out_dir=out_dir,
                samples=samples,
                strict=strict,
            )
        except (FileNotFoundError, NotADirectoryError) as e:
            print(f"ERROR: {e}")
            print("Tip: use --check-only to list missing paths, or --skip-missing to skip incomplete samples.")
            raise SystemExit(2)

if __name__ == "__main__":
    main()
