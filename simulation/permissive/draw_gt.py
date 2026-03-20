# -*- coding: utf-8 -*-
"""
Cell-cell communication (CCC) visualization for ground truth data.

This script generates visualizations of CCC networks with:
- Procedural background textures
- Cell positions colored from HE images
- Arrow connections between cells
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

ARROW_COLOR = "#404040"
ARROW_LINEWIDTH = 0.30
ARROW_ALPHA = 0.1
ARROW_MUTATION_SCALE = 1.5

NICHE_CELL_S = 25.0
NICHE_CELL_ALPHA = 0.95

OTHER_COLOR = (0.20, 0.20, 0.20)
OTHER_CELL_S = 25.0
OTHER_CELL_ALPHA = 0.90

OUTSIDE_BG = np.array([1.00, 1.00, 1.00], dtype=np.float32)
BASE_BG = np.array([0.98, 0.96, 0.97], dtype=np.float32)
BASE_TISSUE = np.array([0.95, 0.92, 0.94], dtype=np.float32)

ADD_TISSUE_TEX_IN_NICHE = True
TEXTURE_DOWNSAMPLE = 4
TEX_SIGMA_LO = 18.0
TEX_SIGMA_MD = 6.0
TEX_STRENGTH = 0.02

USE_GEOM_GRID_TEXTURE = True
ONE_NICHE_TEXTURE = "diamond"

BRICK_W = 30
BRICK_H = 16
MORTAR = 2
BRICK_ANGLE = 0.10

DIAM_PERIOD = 16
DIAM_THICK = 2
DIAM_ANGLE = -0.15

GRID_FILL_COLOR = np.array([1.00, 0.96, 0.97], dtype=np.float32)
GRID_LINE_COLOR = np.array([1.00, 1.00, 1.00], dtype=np.float32)
GRID_FILL_ALPHA = 0.08
GRID_LINE_ALPHA = 0.50

MAX_GRID_SCALE = 3.0

EXPORT_DPI = 300
EXPORT_SCALE_GLOBAL = 3
EXPORT_SCALE_NICHE = 4

SAMPLE_CELL_COLOR_FROM_HE = True
FIXED_CELL_COLOR = (0.10, 0.10, 0.10)


def ensure_layout_xy01(layout: pd.DataFrame) -> pd.DataFrame:
    """Ensure x0, y0, x1, y1 columns exist in layout DataFrame."""
    df = layout.copy()
    cols = set(df.columns)

    if ("x0" in cols) and ("y0" in cols):
        if "x1" not in cols:
            if "tile_size_px" in cols:
                df["x1"] = df["x0"].astype(float) + df["tile_size_px"].astype(float)
            else:
                raise ValueError("layout has x0/y0 but missing x1 and tile_size_px")
        if "y1" not in cols:
            if "tile_size_px" in cols:
                df["y1"] = df["y0"].astype(float) + df["tile_size_px"].astype(float)
            else:
                raise ValueError("layout has x0/y0 but missing y1 and tile_size_px")
        return df

    need = {"cx", "cy", "tile_size_px"}
    if not need.issubset(cols):
        raise ValueError(
            f"layout missing columns for inference. Need either (x0,y0,...) or (cx,cy,tile_size_px). "
            f"cols={list(df.columns)}"
        )

    half = df["tile_size_px"].astype(float) / 2.0
    df["x0"] = (df["cx"].astype(float) - half).round().astype(int)
    df["y0"] = (df["cy"].astype(float) - half).round().astype(int)
    df["x1"] = df["x0"].astype(float) + df["tile_size_px"].astype(float)
    df["y1"] = df["y0"].astype(float) + df["tile_size_px"].astype(float)
    return df


def _as_float_img(arr: np.ndarray) -> np.ndarray:
    """Convert array to float32 image in range [0, 1]."""
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
    """Guess coordinate column names from DataFrame."""
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
    """Guess cell ID column name from DataFrame."""
    cols = [c.lower() for c in df.columns]
    for cand in ["cell_id", "cellid", "cell", "id"]:
        if cand in cols:
            return df.columns[cols.index(cand)]
    return df.columns[0]


def _box_blur_2d(img: np.ndarray, k: int) -> np.ndarray:
    """Apply box blur to 2D image using integral image."""
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
    B = ii[0 : H, k : k + W]
    C = ii[k : k + H, 0 : W]
    A = ii[0 : H, 0 : W]
    return (D - B - C + A) / float(k * k)


def _draw_arrows(ax, src_pts, dst_pts, *, linewidth: float, mutation_scale: float) -> None:
    """Draw arrow patches on axes."""
    for (xS, yS), (xT, yT) in zip(src_pts, dst_pts):
        ax.add_patch(
            FancyArrowPatch(
                (xS, yS),
                (xT, yT),
                arrowstyle="-|>",
                connectionstyle="arc3,rad=0.0",
                shrinkA=0.0,
                shrinkB=0.0,
                mutation_scale=float(mutation_scale),
                linewidth=float(linewidth),
                alpha=ARROW_ALPHA,
                color=ARROW_COLOR,
                zorder=4,
                clip_on=False,
            )
        )


def _clip01(x: float) -> float:
    """Clip value to [0, 1] range."""
    return float(max(0.0, min(1.0, x)))


def _grid_scale_from_p(p: float) -> float:
    """Calculate grid scale from probability parameter."""
    p = _clip01(p)
    return 1.0 + (1.0 - p) * (float(MAX_GRID_SCALE) - 1.0)


def _scaled_geom_params(p: float):
    """Calculate scaled geometry parameters based on probability."""
    s = _grid_scale_from_p(p)
    diam_period = int(max(4, round(float(DIAM_PERIOD) * s)))
    diam_thick = int(max(1, round(float(DIAM_THICK) * (s ** 0.5))))
    brick_w = int(max(6, round(float(BRICK_W) * s)))
    brick_h = int(max(6, round(float(BRICK_H) * s)))
    mortar = int(max(1, round(float(MORTAR) * (s ** 0.5))))
    return brick_w, brick_h, mortar, diam_period, diam_thick


def make_diamond_texture(Hh: int, Ww: int, period: int, thickness: int, angle: float) -> np.ndarray:
    """Generate diamond pattern texture."""
    yy, xx = np.mgrid[0:Hh, 0:Ww].astype(np.float32)
    ca = float(math.cos(angle))
    sa = float(math.sin(angle))
    xr = ca * xx + sa * yy
    yr = -sa * xx + ca * yy
    gx = (np.mod(xr, period) < thickness).astype(np.float32)
    gy = (np.mod(yr, period) < thickness).astype(np.float32)
    return np.clip(gx + gy, 0.0, 1.0).astype(np.float32)


def make_brick_texture(Hh: int, Ww: int, brick_w: int, brick_h: int, mortar: int, angle: float) -> np.ndarray:
    """Generate brick pattern texture."""
    yy, xx = np.mgrid[0:Hh, 0:Ww].astype(np.float32)
    ca = float(math.cos(angle))
    sa = float(math.sin(angle))
    xr = ca * xx + sa * yy
    yr = -sa * xx + ca * yy

    row = np.floor(yr / float(brick_h)).astype(np.int32)
    offset = (row & 1).astype(np.float32) * (0.5 * float(brick_w))
    xr2 = xr + offset

    m_h = (np.mod(yr, float(brick_h)) < float(mortar))
    m_v = (np.mod(xr2, float(brick_w)) < float(mortar))
    return (m_h | m_v).astype(np.float32)


def _make_lowfreq_tissue_tex(H: int, W: int, seed: int) -> np.ndarray:
    """Generate low-frequency tissue texture using noise."""
    rng = np.random.default_rng(seed)
    ds = max(1, int(TEXTURE_DOWNSAMPLE))
    h2 = int(math.ceil(H / ds))
    w2 = int(math.ceil(W / ds))
    noise1 = rng.normal(0, 1.0, size=(h2, w2)).astype(np.float32)
    noise2 = rng.normal(0, 1.0, size=(h2, w2)).astype(np.float32)

    tex_lo = _box_blur_2d(noise1, int(max(3, (TEX_SIGMA_LO / ds) * 2 + 1)))
    tex_md = _box_blur_2d(noise2, int(max(3, (TEX_SIGMA_MD / ds) * 2 + 1)))
    tex2 = 0.70 * tex_lo + 0.30 * tex_md
    mn, mx = float(tex2.min()), float(tex2.max())
    if mx > mn:
        tex2 = (tex2 - mn) / (mx - mn)

    tex = np.repeat(np.repeat(tex2, ds, axis=0), ds, axis=1)
    tex = tex[:H, :W].astype(np.float32)
    return tex


def _apply_grid_colors(bg: np.ndarray, mN: np.ndarray, grid_mask: np.ndarray) -> None:
    """Apply grid colors to background based on mask."""
    if not np.any(mN):
        return

    g = grid_mask[mN].astype(np.float32)
    inv = (1.0 - g).astype(np.float32)

    if GRID_FILL_ALPHA > 0.0:
        a_fill = (GRID_FILL_ALPHA * inv)[:, None]
        bm = bg[mN]
        bm = bm * (1.0 - a_fill) + GRID_FILL_COLOR[None, :] * a_fill
        bg[mN] = np.clip(bm, 0.0, 1.0)

    if GRID_LINE_ALPHA > 0.0:
        a_line = (GRID_LINE_ALPHA * g)[:, None]
        bm = bg[mN]
        bm = bm * (1.0 - a_line) + GRID_LINE_COLOR[None, :] * a_line
        bg[mN] = np.clip(bm, 0.0, 1.0)


def make_global_background_with_niches(H: int, W: int, layout_df: pd.DataFrame, bg_grid_p: float) -> np.ndarray:
    """Generate global background image with niche regions."""
    bg = np.zeros((H, W, 3), dtype=np.float32)
    bg[:] = OUTSIDE_BG[None, None, :]

    need_cols = ["cx", "cy", "r_in", "r_out"]
    if not all(c in layout_df.columns for c in need_cols):
        return bg

    tex = _make_lowfreq_tissue_tex(H, W, seed=12345) if ADD_TISSUE_TEX_IN_NICHE else None

    if USE_GEOM_GRID_TEXTURE:
        bw, bh, mo, dp, dt = _scaled_geom_params(bg_grid_p)
        if ONE_NICHE_TEXTURE.lower() == "brick":
            grid_niche = make_brick_texture(H, W, bw, bh, mo, BRICK_ANGLE)
        else:
            grid_niche = make_diamond_texture(H, W, dp, dt, DIAM_ANGLE)
    else:
        grid_niche = None

    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)

    for _, row in layout_df.iterrows():
        cx = float(row["cx"])
        cy = float(row["cy"])
        rin = float(row["r_in"])
        rout = float(row["r_out"])
        rin2 = rin * rin
        rout2 = rout * rout

        dx = xx - cx
        dy = yy - cy
        rr = dx * dx + dy * dy

        mS = rr <= rin2
        mR = (rr > rin2) & (rr <= rout2)
        mN = mS | mR

        base = (BASE_BG * 0.4 + BASE_TISSUE * 0.6).astype(np.float32)
        if np.any(mN):
            bg[mN] = base[None, :]

            if tex is not None:
                t = tex[mN]
                bg[mN, 0] = np.clip(bg[mN, 0] - 0.02 + TEX_STRENGTH * t, 0, 1)
                bg[mN, 1] = np.clip(bg[mN, 1] - 0.02 + 0.80 * TEX_STRENGTH * t, 0, 1)
                bg[mN, 2] = np.clip(bg[mN, 2] - 0.02 + 0.90 * TEX_STRENGTH * t, 0, 1)

            if grid_niche is not None:
                _apply_grid_colors(bg, mN, grid_niche)

    return bg


def make_niche_background(h_loc: int, w_loc: int, nid: int, layout_row: pd.Series, bg_grid_p: float) -> np.ndarray:
    """Generate niche background image."""
    bg = np.zeros((h_loc, w_loc, 3), dtype=np.float32)
    bg[:] = OUTSIDE_BG[None, None, :]

    for c in ["cx", "cy", "r_in", "r_out", "x0", "y0"]:
        if c not in layout_row.index:
            return bg

    cxg = float(layout_row["cx"])
    cyg = float(layout_row["cy"])
    x0 = float(layout_row["x0"])
    y0 = float(layout_row["y0"])
    cxt = cxg - x0
    cyt = cyg - y0

    rin = float(layout_row["r_in"])
    rout = float(layout_row["r_out"])
    rin2 = rin * rin
    rout2 = rout * rout

    tex = _make_lowfreq_tissue_tex(h_loc, w_loc, seed=10000 + nid) if ADD_TISSUE_TEX_IN_NICHE else None

    if USE_GEOM_GRID_TEXTURE:
        bw, bh, mo, dp, dt = _scaled_geom_params(bg_grid_p)
        if ONE_NICHE_TEXTURE.lower() == "brick":
            grid_niche = make_brick_texture(h_loc, w_loc, bw, bh, mo, BRICK_ANGLE)
        else:
            grid_niche = make_diamond_texture(h_loc, w_loc, dp, dt, DIAM_ANGLE)
    else:
        grid_niche = None

    yy, xx = np.mgrid[0:h_loc, 0:w_loc].astype(np.float32)
    dx = xx - float(cxt)
    dy = yy - float(cyt)
    rr = dx * dx + dy * dy

    mS = rr <= rin2
    mR = (rr > rin2) & (rr <= rout2)
    mN = mS | mR

    base = (BASE_BG * 0.4 + BASE_TISSUE * 0.6).astype(np.float32)
    if np.any(mN):
        bg[mN] = base[None, :]

        if tex is not None:
            t = tex[mN]
            bg[mN, 0] = np.clip(bg[mN, 0] - 0.02 + TEX_STRENGTH * t, 0, 1)
            bg[mN, 1] = np.clip(bg[mN, 1] - 0.02 + 0.80 * TEX_STRENGTH * t, 0, 1)
            bg[mN, 2] = np.clip(bg[mN, 2] - 0.02 + 0.90 * TEX_STRENGTH * t, 0, 1)

        if grid_niche is not None:
            _apply_grid_colors(bg, mN, grid_niche)

    return bg


def _load_pred_csv(sample_dir: str) -> Optional[str]:
    """Load prediction CSV path if exists."""
    p = os.path.join(sample_dir, "pred_edges.csv")
    return p if os.path.exists(p) else None


def _read_global_size_from_cache(layout_df: pd.DataFrame, global_cache: str) -> Tuple[int, int]:
    """Read global canvas size from cache or infer from layout."""
    he_global_npy = os.path.join(global_cache, "HE_global.npy")
    if os.path.exists(he_global_npy):
        he = np.load(he_global_npy, allow_pickle=True)
        return int(he.shape[0]), int(he.shape[1])

    if all(c in layout_df.columns for c in ["x1", "y1"]):
        W = int(np.ceil(layout_df["x1"].astype(float).max()))
        H = int(np.ceil(layout_df["y1"].astype(float).max()))
        return H, W

    if all(c in layout_df.columns for c in ["cx", "cy", "tile_size_px"]):
        half = layout_df["tile_size_px"].astype(float) / 2.0
        W = int(np.ceil((layout_df["cx"].astype(float) + half).max()))
        H = int(np.ceil((layout_df["cy"].astype(float) + half).max()))
        return H, W

    raise ValueError(f"Cannot infer global H,W. layout cols={list(layout_df.columns)}")


def _boxes_union_mask(x: np.ndarray, y: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Create mask for points inside any of the boxes."""
    inside = np.zeros_like(x, dtype=bool)
    for (x0, y0, x1, y1) in boxes:
        inside |= (x >= x0) & (x < x1) & (y >= y0) & (y < y1)
    return inside


def _other_cells_from_mask_global(H: int, W: int, layout_df: pd.DataFrame, global_cache: str, max_other: int = 60000):
    """Extract other cell positions from global mask."""
    mask_p = os.path.join(global_cache, "mask_global.npy")
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
        cx = cx[sel]
        cy = cy[sel]

    return cx, cy


def _other_cells_random(H: int, W: int, layout_df: pd.DataFrame, n_other: int = 25000):
    """Generate random cell positions outside niche regions."""
    rng = np.random.default_rng(7)
    boxes = layout_df[["x0", "y0", "x1", "y1"]].to_numpy(dtype=np.float32)
    xs = rng.uniform(0, W, size=n_other * 3).astype(np.float32)
    ys = rng.uniform(0, H, size=n_other * 3).astype(np.float32)
    inside = _boxes_union_mask(xs, ys, boxes)
    xs = xs[~inside][:n_other]
    ys = ys[~inside][:n_other]
    return xs, ys


def sample_cell_rgb_by_mask_mean(
    he_img: np.ndarray,
    mask_local: np.ndarray,
    cell_ids: np.ndarray,
    xl: np.ndarray,
    yl: np.ndarray
) -> np.ndarray:
    """Sample cell RGB colors by averaging HE image over mask regions."""
    h_loc, w_loc = int(he_img.shape[0]), int(he_img.shape[1])
    rgb = np.zeros((len(cell_ids), 3), dtype=np.float32)

    xi = np.clip(np.rint(xl).astype(np.int32), 0, w_loc - 1)
    yi = np.clip(np.rint(yl).astype(np.int32), 0, h_loc - 1)

    ids_int = np.array([int(x) for x in cell_ids], dtype=np.int32)

    for i, cid in enumerate(ids_int.tolist()):
        m = mask_local == cid
        if np.any(m):
            rgb[i] = he_img[m].mean(axis=0)
        else:
            rgb[i] = he_img[yi[i], xi[i], :]

    return np.clip(rgb, 0.0, 1.0)


def main(strength_val: str):
    """Main function to process a single strength value."""
    strength = strength_val

    dataset_root = f"data/permissive_p{strength}/simulation_data"
    out_root = dataset_root
    niche_save_root = f"viz/gt_{strength}"

    bg_grid_p = float(strength) / 10.0
    global_cache = os.path.join(dataset_root, "_global_cache")
    layout_csv = os.path.join(global_cache, "niche_layout.csv")

    if not os.path.exists(layout_csv):
        raise FileNotFoundError(f"layout not found: {layout_csv}")

    os.makedirs(niche_save_root, exist_ok=True)

    layout = pd.read_csv(layout_csv)
    layout = ensure_layout_xy01(layout)

    if "nid" not in layout.columns:
        raise ValueError(f"niche_layout.csv missing column nid, cols={list(layout.columns)}")

    H, W = _read_global_size_from_cache(layout, global_cache)
    print(f"[global] canvas HxW = {H}x{W}")

    all_cell_x, all_cell_y, all_cell_rgb = [], [], []
    arrow_src, arrow_dst = [], []

    layout = layout.sort_values("nid").reset_index(drop=True)

    for _, row in layout.iterrows():
        nid = int(row["nid"])
        x0 = float(row["x0"])
        y0 = float(row["y0"])

        sample_name = f"sample{nid}"
        sample_dir = os.path.join(dataset_root, sample_name)
        out_dir = os.path.join(out_root, sample_name)

        meta_csv = os.path.join(sample_dir, "simulation_meta.csv")
        he_npy = os.path.join(sample_dir, "HE.npy")
        mask_npy = os.path.join(sample_dir, "mask.npy")
        pred_csv = _load_pred_csv(out_dir)

        if (not os.path.exists(meta_csv)) or (pred_csv is None):
            print(f"[skip] {sample_name}: missing meta/pred")
            continue
        if SAMPLE_CELL_COLOR_FROM_HE and ((not os.path.exists(he_npy)) or (not os.path.exists(mask_npy))):
            print(f"[skip] {sample_name}: missing HE.npy or mask.npy for robust sampling")
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

        xg = xl + x0
        yg = yl + y0

        if SAMPLE_CELL_COLOR_FROM_HE:
            he = np.load(he_npy, allow_pickle=True)
            he_img = _as_float_img(he)
            mask_local = np.load(mask_npy, allow_pickle=True).astype(np.int32)

            if (mask_local.shape[0] != he_img.shape[0]) or (mask_local.shape[1] != he_img.shape[1]):
                raise RuntimeError(f"shape mismatch: HE={he_img.shape} mask={mask_local.shape} in {sample_name}")

            ids_int = np.array([int(x) for x in cids], dtype=np.int32)
            pix_counts = np.array([(mask_local == cid).sum() for cid in ids_int], dtype=np.int64)
            print(f"[debug] {sample_name} mask hits: zero={int((pix_counts==0).sum())}/{pix_counts.size}, "
                  f"min={int(pix_counts.min())}, median={int(np.median(pix_counts))}, max={int(pix_counts.max())}")

            rgb = sample_cell_rgb_by_mask_mean(he_img, mask_local, cids, xl, yl)
            h_loc, w_loc = int(he_img.shape[0]), int(he_img.shape[1])
        else:
            w_loc = int(row["x1"] - row["x0"])
            h_loc = int(row["y1"] - row["y0"])
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

        niche_out_dir = os.path.join(niche_save_root, sample_name)
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

        lw_arrow = float(ARROW_LINEWIDTH)
        ms_arrow = float(ARROW_MUTATION_SCALE)

        fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in), dpi=dpi_n)
        ax.set_position([0, 0, 1, 1])

        bg = make_niche_background(h_loc, w_loc, nid, row, bg_grid_p)
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

        _draw_arrows(ax, niche_src, niche_dst, linewidth=lw_arrow, mutation_scale=ms_arrow)

        ax.set_aspect("equal")
        ax.set_xlim(-0.5, w_loc - 0.5)
        ax.set_ylim(h_loc - 0.5, -0.5)
        ax.set_axis_off()

        fig.savefig(niche_png, dpi=dpi_n)
        fig.savefig(niche_svg)
        plt.close(fig)

    if len(all_cell_x) == 0:
        raise RuntimeError("No niche samples loaded. Check out_root and niche_layout.csv nid mapping.")

    X = np.concatenate(all_cell_x, axis=0)
    Y = np.concatenate(all_cell_y, axis=0)
    C = np.concatenate(all_cell_rgb, axis=0)

    print(f"[global] total niche cells: {X.size}")
    print(f"[global] total arrows: {len(arrow_src)}")

    other = _other_cells_from_mask_global(H, W, layout, global_cache, max_other=80000)
    if other is None:
        ox, oy = _other_cells_random(H, W, layout, n_other=30000)
        print(f"[global] other cells (random): {ox.size}")
    else:
        ox, oy = other
        print(f"[global] other cells (from mask_global): {ox.size}")

    scale_g = int(EXPORT_SCALE_GLOBAL)
    dpi_g = int(EXPORT_DPI)
    fig_w_in = (W * scale_g) / dpi_g
    fig_h_in = (H * scale_g) / dpi_g

    lw_arrow = float(ARROW_LINEWIDTH)
    ms_arrow = float(ARROW_MUTATION_SCALE)

    fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in), dpi=dpi_g)
    ax.set_position([0, 0, 1, 1])

    bg = make_global_background_with_niches(H, W, layout, bg_grid_p)
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

    _draw_arrows(ax, arrow_src, arrow_dst, linewidth=lw_arrow, mutation_scale=ms_arrow)

    ax.set_aspect("equal")
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)
    ax.set_axis_off()

    plt.close(fig)


if __name__ == "__main__":
    strengths = ["00", "02", "04", "06", "08", "10"]

    for strength_val in strengths:
        print(f"\n{'='*80}")
        print(f"Processing strength = {strength_val}")
        print(f"{'='*80}")
        try:
            main(strength_val)
            print(f"[done] strength {strength_val} completed successfully")
        except FileNotFoundError as e:
            print(f"[skip] strength {strength_val}: {e}")
            continue
        except Exception as e:
            print(f"[error] strength {strength_val}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'='*80}")
    print("All strengths processed!")
    print(f"{'='*80}\n")