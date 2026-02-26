# -*- coding: utf-8 -*-
"""
dataset.py
ROOT/
  lr_pairs.csv
  sample1/
    HE.npy
    mask.npy
    *_data.csv
    *_meta.csv
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

try:
    import torch
    from torch.utils.data import Dataset
except Exception as e:
    raise ImportError(
        "Failed to import torch. If you see NumPy 2.x ABI warnings, "
        "please downgrade numpy to <2 in your environment."
    ) from e

try:
    from scipy.spatial import cKDTree
except Exception as e:
    raise ImportError("scipy is required for radius/neighbor graph construction (scipy.spatial.cKDTree).") from e


# -----------------------------
# Helpers
# -----------------------------
def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    suf = path.suffix.lower()
    if suf == ".csv":
        return pd.read_csv(path)
    if suf in (".xlsx", ".xls"):
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file type: {path.suffix} ({path})")


def _drop_unnamed(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in df.columns if not str(c).lower().startswith("unnamed")]
    return df[cols].copy()



def _find_data_meta_tables(sample_dir: Path) -> Tuple[Path, Path]:
    """
    Find matched <prefix>_data.* and <prefix>_meta.* tables inside a sample folder.

    Supported extensions: .csv / .xlsx / .xls

    Matching rule (recommended):
      - <prefix>_data(.csv|.xlsx|.xls)
      - <prefix>_meta(.csv|.xlsx|.xls)

    If multiple prefixes exist, picks the first in sorted order.

    Returns:
      (data_path, meta_path)
    """
    exts = [".csv", ".xlsx", ".xls"]

    data_map: Dict[str, Path] = {}
    meta_map: Dict[str, Path] = {}

    for ext in exts:
        # glob like: human_ovary_data.csv
        for p in sample_dir.glob(f"*{ext}"):
            stem = p.stem  # filename without suffix
            if stem.endswith("_data"):
                pref = stem[: -len("_data")]
                if pref:
                    data_map[pref] = p
            elif stem.endswith("_meta"):
                pref = stem[: -len("_meta")]
                if pref:
                    meta_map[pref] = p

    common = sorted(set(data_map.keys()) & set(meta_map.keys()))
    if not common:
        raise FileNotFoundError(
            f"Cannot find matched <prefix>_data and <prefix>_meta tables under {sample_dir}. "
            f"Supported: <prefix>_data(.csv|.xlsx|.xls) and <prefix>_meta(.csv|.xlsx|.xls). "
            f"Found data prefixes={sorted(data_map.keys())}, meta prefixes={sorted(meta_map.keys())}."
        )

    pref = common[0]
    return data_map[pref], meta_map[pref]


def _infer_id_col(df: pd.DataFrame) -> Optional[str]:
    for c in ["ID", "id", "cell_id", "cellID", "barcode", "CellID", "CellId"]:
        if c in df.columns:
            return c
    return None


def _infer_cell_type_col(meta: pd.DataFrame) -> Optional[str]:
    for c in ["cell_type", "celltype", "CellType", "cellType", "type", "Type"]:
        if c in meta.columns:
            return c
    if len(meta.columns) > 0:
        last = meta.columns[-1]
        if isinstance(last, str) and ("type" in last.lower()):
            return last
    return None


def _infer_coord_cols(meta: pd.DataFrame) -> Tuple[str, str]:
    candidates = [
        ("x", "y"),
        ("X", "Y"),
        ("cx", "cy"),
        ("center_x", "center_y"),
        ("centroid_x", "centroid_y"),
        ("row", "col"),
        ("Row", "Col"),
        ("pos_x", "pos_y"),
        ("PosX", "PosY"),
    ]
    cols = set(meta.columns)
    for a, b in candidates:
        if a in cols and b in cols:
            return a, b

    numeric_cols = [c for c in meta.columns if pd.api.types.is_numeric_dtype(meta[c])]
    if len(numeric_cols) >= 2:
        return numeric_cols[0], numeric_cols[1]

    raise ValueError(
        f"Cannot infer coordinate columns from meta columns: {list(meta.columns)}. "
        "Please rename your coordinate columns to X/Y or x/y."
    )


def _safe_str(x: Any) -> str:
    return "" if x is None else str(x)


def _split_complex(g: str) -> List[str]:
    g = _safe_str(g).strip()
    if not g:
        return []
    parts = []
    for chunk in g.replace("&", "_").split("_"):
        chunk = chunk.strip()
        if chunk:
            parts.append(chunk)
    return parts if parts else [g]


@dataclass(frozen=True)
class LRSpec:
    pair_id: int
    ligand_raw: str
    receptor_raw: str
    ligand_parts: Tuple[str, ...]
    receptor_parts: Tuple[str, ...]
    pathway: str = ""
    mechanism: str = ""


def _load_lr_pairs(lr_pairs_path: Path) -> List[LRSpec]:
    df = _read_table(lr_pairs_path)
    col_map = {str(c).lower(): c for c in df.columns}
    if "ligand" not in col_map or "receptor" not in col_map:
        raise ValueError(
            f"lr_pairs.csv must contain columns 'ligand' and 'receptor'. Got: {list(df.columns)}"
        )

    ligand_c = col_map["ligand"]
    receptor_c = col_map["receptor"]
    pathway_c = col_map.get("pathway", None)
    type_c = col_map.get("type", None)

    specs: List[LRSpec] = []
    for _, row in df.iterrows():
        lig = _safe_str(row[ligand_c]).strip()
        rec = _safe_str(row[receptor_c]).strip()
        if not lig or not rec:
            continue
        pathway = _safe_str(row[pathway_c]).strip() if pathway_c else ""
        mech = _safe_str(row[type_c]).strip() if type_c else ""
        lig_parts = tuple(_split_complex(lig))
        rec_parts = tuple(_split_complex(rec))
        specs.append(
            LRSpec(
                pair_id=len(specs),
                ligand_raw=lig,
                receptor_raw=rec,
                ligand_parts=lig_parts,
                receptor_parts=rec_parts,
                pathway=pathway,
                mechanism=mech,
            )
        )
    if len(specs) == 0:
        raise ValueError(f"No valid LR pairs found in {lr_pairs_path}")
    return specs


def _normalize_id_series(s: pd.Series) -> pd.Series:
    s_str = s.astype(str).str.strip()
    num = pd.to_numeric(s_str, errors="coerce")
    num_int = num.astype("Int64")  # allows NA
    return num_int.astype(str)


def _align_data_meta(
    data_df: pd.DataFrame, meta_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    data_df = _drop_unnamed(data_df)
    meta_df = _drop_unnamed(meta_df)

    data_id = _infer_id_col(data_df)
    meta_id = _infer_id_col(meta_df)

    if (data_id is None) or (meta_id is None):
        if len(data_df) != len(meta_df):
            raise ValueError(
                f"Cannot align data/meta: missing ID col and row counts differ: "
                f"data={len(data_df)}, meta={len(meta_df)}."
            )
        cell_ids = [str(i) for i in range(len(data_df))]
        return data_df.copy(), meta_df.copy(), cell_ids

    meta_id_str = meta_df[meta_id].astype(str).str.strip()
    meta_df = meta_df[meta_id_str.str.lower() != "total"].copy()

    data_df = data_df.copy()
    meta_df = meta_df.copy()
    data_df["_ID_NORM_"] = _normalize_id_series(data_df[data_id])
    meta_df["_ID_NORM_"] = _normalize_id_series(meta_df[meta_id])

    data_df = data_df[data_df["_ID_NORM_"] != "<NA>"].copy()
    meta_df = meta_df[meta_df["_ID_NORM_"] != "<NA>"].copy()

    data_ids = data_df["_ID_NORM_"]
    meta_ids = meta_df["_ID_NORM_"]
    common = set(data_ids) & set(meta_ids)

    if len(common) == 0:
        if len(data_df) == len(meta_df):
            print(
                "[WARN] data/meta ID columns exist but still no overlap after normalization. "
                "Falling back to row-order alignment."
            )
            cell_ids = [str(i) for i in range(len(data_df))]
            data_df = data_df.drop(columns=["_ID_NORM_"])
            meta_df = meta_df.drop(columns=["_ID_NORM_"])
            return data_df.reset_index(drop=True), meta_df.reset_index(drop=True), cell_ids

        raise ValueError(
            f"data/meta have id columns ({data_id},{meta_id}) but no overlap after normalization. "
            f"Row counts: data={len(data_df)}, meta={len(meta_df)}."
        )

    data_df2 = data_df[data_ids.isin(common)].copy()
    meta_df2 = meta_df[meta_ids.isin(common)].copy()
    meta_df2 = meta_df2.set_index("_ID_NORM_").loc[data_df2["_ID_NORM_"].values].reset_index()

    cell_ids = data_df2["_ID_NORM_"].astype(str).tolist()

    data_df2 = data_df2.drop(columns=["_ID_NORM_"])
    meta_df2 = meta_df2.drop(columns=["_ID_NORM_"])

    return data_df2.reset_index(drop=True), meta_df2.reset_index(drop=True), cell_ids


def _build_active_mask(expr_log: np.ndarray, percentile: float) -> np.ndarray:
    if not (0.0 < percentile < 100.0):
        raise ValueError(f"percentile must be (0,100), got {percentile}")
    thr = np.percentile(expr_log, percentile, axis=1, keepdims=True)  # [N,1]
    return expr_log >= thr


def _quantile_normalize_rows(x: np.ndarray) -> np.ndarray:
    if x.ndim != 2:
        raise ValueError(f"quantile_normalize expects 2D array, got shape={x.shape}")
    if x.size == 0:
        return x.astype(np.float32, copy=True)

    order = np.argsort(x, axis=1)
    sorted_x = np.take_along_axis(x, order, axis=1)
    mean_sorted = sorted_x.mean(axis=0)
    inv = np.argsort(order, axis=1)
    out = mean_sorted[inv]
    return out.astype(np.float32, copy=False)


def _compute_radius_neighbors(
    coords: np.ndarray, radius_px: float, include_self: bool = False
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Build a radius graph using cKDTree (ball query).
    Returns per-node neighbor indices and distances (in pixels).
    """
    if radius_px <= 0:
        raise ValueError(f"radius_px must be > 0, got {radius_px}")
    tree = cKDTree(coords)
    nbr_idx_list = tree.query_ball_point(coords, r=float(radius_px))
    out_idx: List[np.ndarray] = []
    out_dist: List[np.ndarray] = []
    for i, nbrs in enumerate(nbr_idx_list):
        if not include_self:
            nbrs = [j for j in nbrs if j != i]
        if len(nbrs) == 0:
            out_idx.append(np.zeros((0,), dtype=np.int64))
            out_dist.append(np.zeros((0,), dtype=np.float32))
            continue
        js = np.asarray(nbrs, dtype=np.int64)
        ds = np.linalg.norm(coords[js] - coords[i], axis=1).astype(np.float32)
        out_idx.append(js)
        out_dist.append(ds)
    return out_idx, out_dist


class CCCGraphDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        sample_id: Optional[str] = None,
        sample_ids: Optional[List[str]] = None,
        lr_pairs_path: Optional[str] = None,
        mpp: float = 0.5,
        neighborhood_um: float = 300.0,
        neighborhood_px: Optional[float] = None,
        quantile_norm: bool = True,
        active_percentile: float = 98.5,
        cache_graph: bool = True,
        mmap_he: bool = True,
        device: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.root_dir = Path(root_dir)
        if not self.root_dir.exists():
            raise FileNotFoundError(f"root_dir not found: {self.root_dir}")

        if lr_pairs_path is None:
            for name in ["lr_pairs.csv", "Lr_pairs.csv", "LR_pairs.csv"]:
                cand = self.root_dir / name
                if cand.exists():
                    lr_pairs_path = str(cand)
                    break
        if lr_pairs_path is None:
            raise FileNotFoundError(
                "lr_pairs_path not provided and not found under root_dir (expected lr_pairs.csv)."
            )

        self.lr_pairs_path = Path(lr_pairs_path)
        self.lr_specs = _load_lr_pairs(self.lr_pairs_path)

        self.active_percentile = float(active_percentile)
        self.mpp = float(mpp)
        self.neighborhood_um = float(neighborhood_um)
        self.neighborhood_px = float(neighborhood_px) if neighborhood_px is not None else (self.neighborhood_um / self.mpp)
        self.quantile_norm = bool(quantile_norm)
        self.cache_graph = bool(cache_graph)
        self.mmap_he = bool(mmap_he)
        self.device = device

        # find sample folders
        # find sample folders
        # Support two layouts:
        #   (1) dataset root contains multiple sample subfolders: root_dir/<sample_id>/{HE.npy,mask.npy,<prefix>_data.*,<prefix>_meta.*}
        #   (2) root_dir itself is a single sample folder (so you can run only one sample by setting root_dir to that folder)
        self.sample_dirs: List[Path] = []

        def _is_valid_sample_dir(p: Path) -> bool:
            if not p.is_dir():
                return False
            if not (p / "HE.npy").exists():
                return False
            if not (p / "mask.npy").exists():
                return False
            try:
                _find_data_meta_tables(p)
            except FileNotFoundError:
                return False
            return True

        # case (2): root_dir is itself a sample folder
        if _is_valid_sample_dir(self.root_dir):
            self.sample_dirs = [self.root_dir]
        else:
            # case (1): scan subfolders
            for p in sorted(self.root_dir.iterdir()):
                if _is_valid_sample_dir(p):
                    self.sample_dirs.append(p)

        # optional filtering by sample_id / sample_ids
        if sample_id is not None and str(sample_id).strip() != "":
            want = str(sample_id).strip()
            self.sample_dirs = [p for p in self.sample_dirs if p.name == want]
        
        if sample_ids is not None and len(sample_ids) > 0:
            want_set = {str(s).strip() for s in sample_ids if str(s).strip() != ""}
            self.sample_dirs = [p for p in self.sample_dirs if p.name in want_set]

        if len(self.sample_dirs) == 0:
            raise FileNotFoundError(
                f"No valid sample folders found under {self.root_dir}. "
                "Expect: <sample_dir>/HE.npy, <sample_dir>/mask.npy, <prefix>_data.(csv|xlsx|xls), <prefix>_meta.(csv|xlsx|xls). "
                "Tip: if you only want to run one sample, you can set root_dir directly to that sample folder."
            )
        self._cache: Dict[int, Dict[str, Any]] = {}

    def __len__(self) -> int:
        return len(self.sample_dirs)

    # -----------------------------
    # Convenience interfaces you asked for
    # -----------------------------
    def get_he(self, idx: int) -> np.ndarray:
        return self[idx]["he"]

    def get_mask(self, idx: int) -> np.ndarray:
        return self[idx]["mask"]

    def get_graph(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self[idx]
        return {
            "node_ids": item["node_ids"],
            "coords": item["coords"],
            "edge_index": item["edge_index"],
            "edge_attr": item["edge_attr"],
        }

    # -----------------------------
    # Main loader
    # -----------------------------
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.cache_graph and idx in self._cache:
            return self._cache[idx]

        sample_dir = self.sample_dirs[idx]
        sample_id = sample_dir.name

        he_path = sample_dir / "HE.npy"
        mask_path = sample_dir / "mask.npy"
        he = np.load(he_path, mmap_mode="r" if self.mmap_he else None)
        mask = np.load(mask_path, mmap_mode="r" if self.mmap_he else None)
        data_path, meta_path = _find_data_meta_tables(sample_dir)


        data_df = _read_table(data_path)
        meta_df = _read_table(meta_path)

        # align
        data_df, meta_df, cell_ids = _align_data_meta(data_df, meta_df)

        # genes
        data_id_col = _infer_id_col(data_df)
        gene_cols = [c for c in data_df.columns if c != data_id_col]
        if len(gene_cols) == 0:
            raise ValueError(f"No gene columns found in {data_path}. Columns: {list(data_df.columns)}")

        # coords
        xcol, ycol = _infer_coord_cols(meta_df)
        coords_np = meta_df[[xcol, ycol]].to_numpy(dtype=np.float32)  # [N,2]
        n = coords_np.shape[0]

        # optional cell_type
        ct_col = _infer_cell_type_col(meta_df)
        cell_types = None
        if ct_col is not None and ct_col in meta_df.columns:
            cell_types = meta_df[ct_col].astype(str).fillna("").tolist()

        # expression
        expr_raw = data_df[gene_cols].to_numpy(dtype=np.float32)   # [N,G]
        expr_log = np.log1p(expr_raw)                              # [N,G]
        if self.quantile_norm:
            expr_log = _quantile_normalize_rows(expr_log)

        active_mask = _build_active_mask(expr_log, self.active_percentile)  # [N,G]
        gene_to_idx = {str(g): i for i, g in enumerate(gene_cols)}

        # split LR into simple & complex
        simple_pairs = []   # (pair_id, lig_idx, rec_idx)
        complex_pairs = []  # (pair_id, lig_idx_list, rec_idx_list)
        kept_specs: List[LRSpec] = []

        for spec in self.lr_specs:
            lig_parts = list(spec.ligand_parts)
            rec_parts = list(spec.receptor_parts)
            if any(p not in gene_to_idx for p in lig_parts) or any(p not in gene_to_idx for p in rec_parts):
                continue
            kept_specs.append(spec)
            lig_idx_list = [gene_to_idx[p] for p in lig_parts]
            rec_idx_list = [gene_to_idx[p] for p in rec_parts]
            if len(lig_idx_list) == 1 and len(rec_idx_list) == 1:
                simple_pairs.append((spec.pair_id, lig_idx_list[0], rec_idx_list[0]))
            else:
                complex_pairs.append((spec.pair_id, lig_idx_list, rec_idx_list))

        if len(kept_specs) == 0:
            raise ValueError("No LR pairs remain after matching gene names with expression columns.")

        # radius neighbors
        nbr_idx_list, nbr_dist_list = _compute_radius_neighbors(coords_np, radius_px=self.neighborhood_px)

        # build multi-edges
        src_list: List[int] = []
        dst_list: List[int] = []
        lrid_list: List[int] = []
        dist_list: List[float] = []
        coexpr_list: List[float] = []

        if simple_pairs:
            pair_ids = np.array([p[0] for p in simple_pairs], dtype=np.int64)
            lig_idx = np.array([p[1] for p in simple_pairs], dtype=np.int64)
            rec_idx = np.array([p[2] for p in simple_pairs], dtype=np.int64)

        for rcv in range(n):  # receiver
            senders = nbr_idx_list[rcv]
            dists = nbr_dist_list[rcv]
            if senders.size == 0:
                continue

            # simple pairs
            if simple_pairs:
                rec_active = active_mask[rcv, rec_idx]          # [K]
                rec_expr = expr_log[rcv, rec_idx]               # [K]
                lig_active = active_mask[senders][:, lig_idx]   # [S,K]
                keep = lig_active & rec_active[None, :]         # [S,K]

                s_slot, k_slot = np.nonzero(keep)
                if s_slot.size > 0:
                    picked_s = senders[s_slot]
                    picked_d = dists[s_slot]
                    picked_pid = pair_ids[k_slot]

                    lig_expr = expr_log[picked_s, lig_idx[k_slot]]
                    rec_expr_p = rec_expr[k_slot]
                    coexpr = lig_expr * rec_expr_p

                    src_list.extend(picked_s.tolist())
                    dst_list.extend([rcv] * int(s_slot.size))
                    dist_list.extend(picked_d.astype(float).tolist())
                    coexpr_list.extend(coexpr.astype(float).tolist())
                    lrid_list.extend(picked_pid.tolist())

            # complex pairs (AND across subunits)
            for pid, lig_comp, rec_comp in complex_pairs:
                if not bool(active_mask[rcv, rec_comp].all()):
                    continue
                send_ok = active_mask[senders][:, lig_comp].all(axis=1)
                if not np.any(send_ok):
                    continue
                ok_slots = np.nonzero(send_ok)[0]
                picked_s = senders[ok_slots]
                picked_d = dists[ok_slots]

                lig_expr = np.min(expr_log[picked_s][:, lig_comp], axis=1)
                rec_expr_c = float(np.min(expr_log[rcv, rec_comp]))
                coexpr = lig_expr * rec_expr_c

                src_list.extend(picked_s.tolist())
                dst_list.extend([rcv] * len(ok_slots))
                dist_list.extend(picked_d.astype(float).tolist())
                coexpr_list.extend(coexpr.astype(float).tolist())
                lrid_list.extend([pid] * len(ok_slots))

        # tensors
        if len(src_list) == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 3), dtype=torch.float32)
        else:
            edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
            edge_attr = torch.tensor(
                np.stack(
                    [
                        np.array(dist_list, np.float32),
                        np.array(coexpr_list, np.float32),
                        np.array(lrid_list, np.float32),
                    ],
                    axis=1,
                ),
                dtype=torch.float32,
            )

        node_ids = torch.arange(n, dtype=torch.long)

        out: Dict[str, Any] = {
            "sample_id": sample_id,
            "he": he,
            "mask": mask,
            "cell_ids": cell_ids,
            "gene_names": [str(g) for g in gene_cols],
            "expr_raw": torch.from_numpy(expr_raw).float(),
            "expr_log": torch.from_numpy(expr_log).float(),
            "active_mask": torch.from_numpy(active_mask),
            "coords": torch.from_numpy(coords_np).float(),
            "node_ids": node_ids,
            "edge_index": edge_index,
            "edge_attr": edge_attr,  # [distance_px, coexpr, lr_id]
            "lr_pairs": {
                "lr_pairs_path": str(self.lr_pairs_path),
                "active_percentile": float(self.active_percentile),
                "pairs_kept": [
                    {
                        "pair_id": spec.pair_id,
                        "ligand": spec.ligand_raw,
                        "receptor": spec.receptor_raw,
                        "ligand_parts": list(spec.ligand_parts),
                        "receptor_parts": list(spec.receptor_parts),
                        "pathway": spec.pathway,
                        "mechanism": spec.mechanism,
                    }
                    for spec in kept_specs
                ],
            },
            "paths": {
                "sample_dir": str(sample_dir),
                "he_path": str(he_path),
                "mask_path": str(mask_path),
                "data_path": str(data_path),
                "meta_path": str(meta_path),
            },
        }

        if cell_types is not None:
            out["cell_types"] = cell_types

        if self.device is not None:
            dev = torch.device(self.device)
            out["expr_raw"] = out["expr_raw"].to(dev, non_blocking=True)
            out["expr_log"] = out["expr_log"].to(dev, non_blocking=True)
            out["active_mask"] = out["active_mask"].to(dev, non_blocking=True)
            out["coords"] = out["coords"].to(dev, non_blocking=True)
            out["node_ids"] = out["node_ids"].to(dev, non_blocking=True)
            out["edge_index"] = out["edge_index"].to(dev, non_blocking=True)
            out["edge_attr"] = out["edge_attr"].to(dev, non_blocking=True)

        if self.cache_graph:
            self._cache[idx] = out
        return out