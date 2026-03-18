# -*- coding: utf-8 -*-
"""
Global + per-niche CCC visualization (procedural background, step-by-step),
with niche-only pathology-like textures:
- Outside niches: near-white clean background (for "other cells" area)
- Inside niche:
  - sender disk + receiver ring share ONE single texture (either brick OR diamond)
- Optional: subtle low-frequency tissue texture ONLY within niche geometry

HE.npy is used ONLY for sampling cell colors (optional).
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from typing import Tuple, Optional


# =========================================================
# Paths
# =========================================================

strength = "10"
OUT_ROOT = f"output/without_texture_{strength}"
DATASET_ROOT = f"data/barrier_s{strength}/simulation_data"
NICHE_SAVE_ROOT = f"viz/without_texture_{strength}"
GLOBAL_CACHE = os.path.join(DATASET_ROOT, "_global_cache")
LAYOUT_CSV = os.path.join(GLOBAL_CACHE, "niche_layout.csv")

BIG_SAVE_DIR = "big_fig"


def set_paths(strength_val: str):
    """Set global paths based on strength value."""
    global OUT_ROOT, DATASET_ROOT, NICHE_SAVE_ROOT, GLOBAL_CACHE, LAYOUT_CSV, BIG_SAVE_DIR
    strength = strength_val
    OUT_ROOT = f"output/without_texture_{strength}"
    DATASET_ROOT = f"data/barrier_s{strength}/simulation_data"
    NICHE_SAVE_ROOT = f"viz/without_texture_{strength}"
    GLOBAL_CACHE = os.path.join(DATASET_ROOT, "_global_cache")
    LAYOUT_CSV = os.path.join(GLOBAL_CACHE, "niche_layout.csv")
    
    BIG_SAVE_DIR = f"big_fig_{strength}"


# =========================================================
# Visual style
# =========================================================
# arrows
ARROW_COLOR = "#545353"
ARROW_LINEWIDTH = 0.35
ARROW_ALPHA = 0.15
ARROW_MUTATION_SCALE = 2.0

# cells
NICHE_CELL_S = 25.0
NICHE_CELL_ALPHA = 0.95

OTHER_COLOR = (0.20, 0.20, 0.20)
OTHER_CELL_S = 25.0
OTHER_CELL_ALPHA = 0.90


# Background colors
OUTSIDE_BG = np.array([1.00, 1.00, 1.00], dtype=np.float32)

BASE_BG     = np.array([0.92, 0.88, 0.90], dtype=np.float32)
BASE_TISSUE = np.array([0.85, 0.78, 0.82], dtype=np.float32)

# Background texture settings (disabled for this version)
ADD_TISSUE_TEX_IN_NICHE = False
USE_GEOM_GRID_TEXTURE = False


# High-resolution export settings
EXPORT_DPI = 300
EXPORT_SCALE_GLOBAL = 3
EXPORT_SCALE_NICHE = 4
SCALE_STROKES_WITH_EXPORT = True


# Cell coloring settings
SAMPLE_CELL_COLOR_FROM_HE = True
FIXED_CELL_COLOR = (0.10, 0.10, 0.10)


# Helper functions
def _as_float_img(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr)
    if a.ndim == 2:
        a = a.astype(np.float32)
        mn, mx = float(a.min()), float(a.max())
        if mx > mn:
            a = (a - mn) / (mx - mn)
        a = np.stack([a, a, a], axis=-1)
        return a
    if a.ndim == 3:
        if a.shape[2] >= 3:
            rgb = a[..., :3].astype(np.float32)
        else:
            g = a[..., 0].astype(np.float32)
            mn, mx = float(g.min()), float(g.max())
            if mx > mn:
                g = (g - mn) / (mx - mn)
            rgb = np.stack([g, g, g], axis=-1)
            return rgb
        if rgb.max() <= 1.5 and rgb.min() >= -0.5:
            return np.clip(rgb, 0.0, 1.0)
        if rgb.max() <= 255.0 and rgb.min() >= 0.0:
            return np.clip(rgb / 255.0, 0.0, 1.0)
        mn, mx = float(rgb.min()), float(rgb.max())
        if mx > mn:
            rgb = (rgb - mn) / (mx - mn)
        return np.clip(rgb, 0.0, 1.0)
    raise ValueError(f"Unsupported HE shape: {a.shape}")


def guess_coord_cols(df: pd.DataFrame) -> Tuple[str, str]:
    cols = [c.lower() for c in df.columns]
    cand_pairs = [
        ("x", "y"),
        ("coord_x", "coord_y"),
        ("cx", "cy"),
        ("px", "py"),
        ("pos_x", "pos_y"),
        ("center_x", "center_y"),
    ]
    for a, b in cand_pairs:
        if a in cols and b in cols:
            return df.columns[cols.index(a)], df.columns[cols.index(b)]
    raise ValueError(f"Cannot infer coord columns from meta.csv columns={list(df.columns)}")


def guess_cellid_col(df: pd.DataFrame) -> str:
    cols = [c.lower() for c in df.columns]
    for cand in ["cell_id", "cellid", "cell", "id"]:
        if cand in cols:
            return df.columns[cols.index(cand)]
    return df.columns[0]


def _box_blur_2d(img: np.ndarray, k: int) -> np.ndarray:
    k = int(k)
    if k < 3:
        return img.astype(np.float32, copy=False)
    if k % 2 == 0:
        k += 1
    r = k // 2

    img = np.asarray(img, dtype=np.float32)
    H, W = img.shape
    pad = np.pad(img, ((r, r), (r, r)), mode="reflect")
    Hp, Wp = pad.shape

    ii = np.zeros((Hp + 1, Wp + 1), dtype=np.float32)
    ii[1:, 1:] = pad.cumsum(axis=0).cumsum(axis=1)

    D = ii[k : k + H, k : k + W]
    B = ii[0 : H,     k : k + W]
    C = ii[k : k + H, 0 : W]
    A = ii[0 : H,     0 : W]
    return (D - B - C + A) / float(k * k)


def _scaled_stroke(base: float, scale: int) -> float:
    if not SCALE_STROKES_WITH_EXPORT:
        return float(base)
    return float(base) * (0.75 + 0.45 * float(scale))


def _scaled_mutation(base: float, scale: int) -> float:
    if not SCALE_STROKES_WITH_EXPORT:
        return float(base)
    return float(base) * (0.80 + 0.50 * float(scale))


def make_global_background_with_niches(H: int, W: int, layout_df: pd.DataFrame) -> np.ndarray:
    bg = np.zeros((H, W, 3), dtype=np.float32)
    bg[:] = OUTSIDE_BG[None, None, :]
    return bg


def make_niche_background(h_loc: int, w_loc: int, nid: int, layout_row: pd.Series) -> np.ndarray:
    bg = np.zeros((h_loc, w_loc, 3), dtype=np.float32)
    bg[:] = OUTSIDE_BG[None, None, :]
    return bg


def _load_pred_csv(sample_out_dir: str) -> Optional[str]:
    p = os.path.join(sample_out_dir, "pred_edges.csv")
    return p if os.path.exists(p) else None


# =========================================================
# FIX: layout compatibility layer
# =========================================================
def _ensure_layout_bbox(layout: pd.DataFrame) -> pd.DataFrame:
    """Support both old layout (x0,y0,x1,y1) and new layout (cx,cy,r_in,r_out,tile_size_px,tile_pad).
    Always ends with x0,y0,x1,y1 columns.
    """
    layout = layout.copy()

    need_bbox = any(c not in layout.columns for c in ["x0", "y0", "x1", "y1"])
    if not need_bbox:
        return layout

    required = ["cx", "cy", "tile_size_px", "tile_pad"]
    miss = [c for c in required if c not in layout.columns]
    if len(miss) > 0:
        raise ValueError(
            f"niche_layout.csv missing bbox cols and also missing {miss}. "
            f"cols={list(layout.columns)}"
        )

    cx = pd.to_numeric(layout["cx"], errors="coerce").astype(np.float64)
    cy = pd.to_numeric(layout["cy"], errors="coerce").astype(np.float64)
    tile = pd.to_numeric(layout["tile_size_px"], errors="coerce").astype(np.float64)
    pad  = pd.to_numeric(layout["tile_pad"], errors="coerce").astype(np.float64)

    # generator logic: gx0 = cx - half - pad; gx1=gx0+tile
    half = np.floor(tile / 2.0)
    x0 = cx - half - pad
    y0 = cy - half - pad
    x1 = x0 + tile
    y1 = y0 + tile

    layout["x0"] = x0.astype(np.float32)
    layout["y0"] = y0.astype(np.float32)
    layout["x1"] = x1.astype(np.float32)
    layout["y1"] = y1.astype(np.float32)

    return layout


def _read_global_size_from_cache(layout_df: pd.DataFrame) -> Tuple[int, int]:
    he_global_npy = os.path.join(GLOBAL_CACHE, "HE_global.npy")
    if os.path.exists(he_global_npy):
        he = np.load(he_global_npy, allow_pickle=True)
        return int(he.shape[0]), int(he.shape[1])

    # if no HE_global.npy, try bbox columns
    if all(c in layout_df.columns for c in ["x1", "y1"]):
        W = int(np.ceil(layout_df["x1"].max()))
        H = int(np.ceil(layout_df["y1"].max()))
        return H, W

    raise RuntimeError("Cannot infer global H,W: missing HE_global.npy and missing layout bbox x1/y1.")


def _boxes_union_mask(x: np.ndarray, y: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    inside = np.zeros_like(x, dtype=bool)
    for (x0, y0, x1, y1) in boxes:
        inside |= (x >= x0) & (x < x1) & (y >= y0) & (y < y1)
    return inside


def _other_cells_from_mask_global(H: int, W: int, layout_df: pd.DataFrame, max_other: int = 60000):
    mask_p = os.path.join(GLOBAL_CACHE, "mask_global.npy")
    if not os.path.exists(mask_p):
        return None

    seg = np.load(mask_p, allow_pickle=True).astype(np.int32)
    if seg.ndim != 2:
        return None

    lab = seg.ravel()
    max_lab = int(lab.max())
    if max_lab <= 0:
        return None

    idx = np.arange(lab.size, dtype=np.int64)
    counts = np.bincount(lab, minlength=max_lab + 1).astype(np.int64)
    valid = np.where(counts[1:] > 0)[0] + 1
    if valid.size == 0:
        return None

    xs = (idx % W).astype(np.float64)
    ys = (idx // W).astype(np.float64)
    sumx = np.bincount(lab, weights=xs, minlength=max_lab + 1)
    sumy = np.bincount(lab, weights=ys, minlength=max_lab + 1)

    cx = (sumx[valid] / counts[valid]).astype(np.float32)
    cy = (sumy[valid] / counts[valid]).astype(np.float32)

    boxes = layout_df[["x0", "y0", "x1", "y1"]].to_numpy(dtype=np.float32)
    inside = _boxes_union_mask(cx, cy, boxes)

    cx = cx[~inside]
    cy = cy[~inside]

    if cx.size > max_other:
        rng = np.random.default_rng(7)
        sel = rng.choice(cx.size, size=max_other, replace=False)
        cx = cx[sel]; cy = cy[sel]

    return cx, cy


def _other_cells_random(H: int, W: int, layout_df: pd.DataFrame, n_other: int = 25000):
    rng = np.random.default_rng(7)
    boxes = layout_df[["x0", "y0", "x1", "y1"]].to_numpy(dtype=np.float32)
    xs = rng.uniform(0, W, size=n_other * 3).astype(np.float32)
    ys = rng.uniform(0, H, size=n_other * 3).astype(np.float32)
    inside = _boxes_union_mask(xs, ys, boxes)
    xs = xs[~inside][:n_other]
    ys = ys[~inside][:n_other]
    return xs, ys


def _sample_cell_rgb_from_he(he_img: np.ndarray, xl: np.ndarray, yl: np.ndarray) -> np.ndarray:
    h_loc, w_loc = int(he_img.shape[0]), int(he_img.shape[1])
    xi = np.clip(np.rint(xl).astype(np.int32), 0, w_loc - 1)
    yi = np.clip(np.rint(yl).astype(np.int32), 0, h_loc - 1)
    return he_img[yi, xi, :]


# Main function
def main(strength_val: str = "10"):
    set_paths(strength_val)
    
    if not os.path.exists(LAYOUT_CSV):
        raise FileNotFoundError(f"layout not found: {LAYOUT_CSV}")

    os.makedirs(BIG_SAVE_DIR, exist_ok=True)
    os.makedirs(NICHE_SAVE_ROOT, exist_ok=True)

    layout = pd.read_csv(LAYOUT_CSV)
    layout = _ensure_layout_bbox(layout)
    layout = layout.sort_values("nid").reset_index(drop=True)

    # now bbox columns must exist
    for c in ["nid", "x0", "y0", "x1", "y1"]:
        if c not in layout.columns:
            raise ValueError(f"niche_layout.csv missing column {c}, cols={list(layout.columns)}")

    H, W = _read_global_size_from_cache(layout)
    print(f"[global] canvas HxW = {H}x{W}")

    all_cell_x, all_cell_y, all_cell_rgb = [], [], []
    arrow_src, arrow_dst = [], []

    # ---------- per-niche loop ----------
    for _, row in layout.iterrows():
        nid = int(row["nid"])
        x0 = float(row["x0"])
        y0 = float(row["y0"])
        x1 = float(row["x1"])
        y1 = float(row["y1"])

        sample_name = f"sample{nid}"
        sample_dir = os.path.join(DATASET_ROOT, sample_name)
        sample_out_dir = os.path.join(OUT_ROOT, sample_name)

        meta_csv = os.path.join(sample_dir, "simulation_meta.csv")
        he_npy = os.path.join(sample_dir, "HE.npy")  # optional for sampling color
        pred_csv = _load_pred_csv(sample_out_dir)

        if (not os.path.exists(meta_csv)) or (pred_csv is None):
            print(f"[skip] {sample_name}: missing meta/pred")
            continue
        if SAMPLE_CELL_COLOR_FROM_HE and (not os.path.exists(he_npy)):
            print(f"[skip] {sample_name}: missing HE.npy for sampling colors")
            continue

        meta = pd.read_csv(meta_csv)
        cellid_col = guess_cellid_col(meta)
        xcol, ycol = guess_coord_cols(meta)

        meta_cell = meta[[cellid_col, xcol, ycol]].copy()
        meta_cell[cellid_col] = meta_cell[cellid_col].astype(str)
        meta_cell[xcol] = pd.to_numeric(meta_cell[xcol], errors="coerce")
        meta_cell[ycol] = pd.to_numeric(meta_cell[ycol], errors="coerce")
        meta_cell = meta_cell.dropna(subset=[xcol, ycol]).reset_index(drop=True)

        cids = meta_cell[cellid_col].to_numpy(dtype=str)
        xl = meta_cell[xcol].to_numpy(dtype=np.float32)
        yl = meta_cell[ycol].to_numpy(dtype=np.float32)

        # local -> global (still uses bbox x0/y0)
        xg = xl + x0
        yg = yl + y0

        # cell colors
        if SAMPLE_CELL_COLOR_FROM_HE:
            he = np.load(he_npy, allow_pickle=True)
            he_img = _as_float_img(he)
            rgb = _sample_cell_rgb_from_he(he_img, xl, yl)
            h_loc, w_loc = int(he_img.shape[0]), int(he_img.shape[1])
        else:
            w_loc = int(round(x1 - x0))
            h_loc = int(round(y1 - y0))
            rgb = np.tile(np.array(FIXED_CELL_COLOR, dtype=np.float32)[None, :], (xl.size, 1))

        all_cell_x.append(xg)
        all_cell_y.append(yg)
        all_cell_rgb.append(rgb)

        cell2xy_g = {cid: (float(xx), float(yy)) for cid, xx, yy in zip(cids, xg, yg)}
        cell2xy_l = {cid: (float(xx), float(yy)) for cid, xx, yy in zip(cids, xl, yl)}

        pred = pd.read_csv(pred_csv)
        if ("src_cell_id" not in pred.columns) or ("dst_cell_id" not in pred.columns):
            print(f"[skip] {sample_name}: pred_edges.csv missing src/dst_cell_id")
            continue

        pred["src_cell_id"] = pred["src_cell_id"].astype(str)
        pred["dst_cell_id"] = pred["dst_cell_id"].astype(str)
        pred = pred[pred["src_cell_id"].isin(cell2xy_g.keys()) & pred["dst_cell_id"].isin(cell2xy_g.keys())]

        for s, t in zip(pred["src_cell_id"].to_numpy(), pred["dst_cell_id"].to_numpy()):
            xS, yS = cell2xy_g[s]
            xT, yT = cell2xy_g[t]
            arrow_src.append((xS, yS))
            arrow_dst.append((xT, yT))

        print(f"[ok] {sample_name}: cells={len(cids)}, edges={len(pred)}")

        # =========================================================
        # Per-niche figure (background = pure white)
        # =========================================================
        niche_out_dir = os.path.join(NICHE_SAVE_ROOT, sample_name)
        os.makedirs(niche_out_dir, exist_ok=True)
        niche_png = os.path.join(niche_out_dir, "ccc_niche.png")
        niche_svg = os.path.join(niche_out_dir, "ccc_niche.svg")

        niche_src, niche_dst = [], []
        for s, t in zip(pred["src_cell_id"].to_numpy(), pred["dst_cell_id"].to_numpy()):
            xS, yS = cell2xy_l[s]
            xT, yT = cell2xy_l[t]
            niche_src.append((xS, yS))
            niche_dst.append((xT, yT))

        scale_n = int(EXPORT_SCALE_NICHE)
        dpi_n = int(EXPORT_DPI)
        fig_w_in = (w_loc * scale_n) / dpi_n
        fig_h_in = (h_loc * scale_n) / dpi_n

        lw_n = _scaled_stroke(ARROW_LINEWIDTH, scale_n)
        ms_n = _scaled_mutation(ARROW_MUTATION_SCALE, scale_n)

        fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in), dpi=dpi_n)
        ax.set_position([0, 0, 1, 1])

        bg = make_niche_background(h_loc, w_loc, nid, row)
        ax.imshow(bg, origin="upper", interpolation="nearest", resample=False, rasterized=True, zorder=0)

        ax.scatter(
            xl, yl,
            s=NICHE_CELL_S,
            c=rgb,
            alpha=NICHE_CELL_ALPHA,
            marker="o",
            edgecolors="none",
            linewidths=0.0,
            zorder=3,
        )

        for (xS, yS), (xT, yT) in zip(niche_src, niche_dst):
            ax.add_patch(FancyArrowPatch(
                (xS, yS),
                (xT, yT),
                arrowstyle="-|>",
                mutation_scale=ms_n,
                linewidth=lw_n,
                alpha=ARROW_ALPHA,
                color=ARROW_COLOR,
                zorder=4,
                clip_on=False,
            ))

        ax.set_aspect("equal")
        ax.set_xlim(-0.5, w_loc - 0.5)
        ax.set_ylim(h_loc - 0.5, -0.5)
        ax.set_axis_off()

        fig.savefig(niche_png, dpi=dpi_n)
        fig.savefig(niche_svg)
        plt.close(fig)

    if len(all_cell_x) == 0:
        raise RuntimeError("No niche samples loaded. Check paths & nid mapping.")

    X = np.concatenate(all_cell_x, axis=0)
    Y = np.concatenate(all_cell_y, axis=0)
    C = np.concatenate(all_cell_rgb, axis=0)

    print(f"[global] total niche cells: {X.size}")
    print(f"[global] total arrows: {len(arrow_src)}")

    other = _other_cells_from_mask_global(H, W, layout, max_other=80000)
    if other is None:
        ox, oy = _other_cells_random(H, W, layout, n_other=30000)
        print(f"[global] other cells (random): {ox.size}")
    else:
        ox, oy = other
        print(f"[global] other cells (from mask_global): {ox.size}")

    # =========================================================
    # Global figure (background = pure white)
    # =========================================================
    scale_g = int(EXPORT_SCALE_GLOBAL)
    dpi_g = int(EXPORT_DPI)
    fig_w_in = (W * scale_g) / dpi_g
    fig_h_in = (H * scale_g) / dpi_g

    lw_g = _scaled_stroke(ARROW_LINEWIDTH, scale_g)
    ms_g = _scaled_mutation(ARROW_MUTATION_SCALE, scale_g)

    fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in), dpi=dpi_g)
    ax.set_position([0, 0, 1, 1])

    bg = make_global_background_with_niches(H, W, layout)
    ax.imshow(bg, origin="upper", interpolation="nearest", resample=False, rasterized=True, zorder=0)

    ax.scatter(
        ox, oy,
        s=OTHER_CELL_S,
        c=[OTHER_COLOR],
        alpha=OTHER_CELL_ALPHA,
        marker="o",
        edgecolors="none",
        linewidths=0.0,
        zorder=2.5,
    )

    ax.scatter(
        X, Y,
        s=NICHE_CELL_S,
        c=C,
        alpha=NICHE_CELL_ALPHA,
        marker="o",
        edgecolors="none",
        linewidths=0.0,
        zorder=3,
    )

    for (xS, yS), (xT, yT) in zip(arrow_src, arrow_dst):
        ax.add_patch(FancyArrowPatch(
            (xS, yS),
            (xT, yT),
            arrowstyle="-|>",
            mutation_scale=ms_g,
            linewidth=lw_g,
            alpha=ARROW_ALPHA,
            color=ARROW_COLOR,
            zorder=4,
            clip_on=False,
        ))

    ax.set_aspect("equal")
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)
    ax.set_axis_off()

    # fig.savefig(OUT_PNG, dpi=dpi_g)
    # fig.savefig(OUT_SVG)
    plt.close(fig)

    print("[saved]", OUT_PNG)
    print("[saved]", OUT_SVG)
    print("[hires] global pixels =", (W * scale_g, H * scale_g), "dpi =", dpi_g)


if __name__ == "__main__":
    strengths = ["10", "08", "06", "04", "02", "00"]
    for strength_val in strengths:
        print(f"\n{'='*60}")
        print(f"Processing strength = {strength_val}")
        print(f"{'='*60}")
        try:
            main(strength_val)
        except FileNotFoundError as e:
            print(f"[skip] strength {strength_val}: {e}")
            continue
        except Exception as e:
            print(f"[error] strength {strength_val}: {e}")
            continue
        print(f"[done] strength {strength_val} completed successfully")
    print(f"\n{'='*60}")
    print("All strengths processed!")
    print(f"{'='*60}\n")
