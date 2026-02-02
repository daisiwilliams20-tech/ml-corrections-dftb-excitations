#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build Coulomb-matrix features with molml, safely coerce to 2-D float32,
and save to NPZ + CSV (CSV written without pandas).

Run:  python3 make_cm_molml.py
"""

import os
import re
import glob
import numpy as np
from molml.features import CoulombMatrix

# ========= CONFIG =========
XYZ_DIR     = "/home/williams/Downloads/send_Williams (2)/moles"
OUT_CSV     = "/home/williams/Downloads/send_Williams (2)/cm_vectors_molml.csv"
OUT_NPZ     = "/home/williams/Downloads/send_Williams (2)/cm_vectors_molml.npz"
BATCH_SIZE  = 1500      # 0 = all at once; use 1000–3000 for low RAM
FLOAT_FMT   = "{:.8f}"

# Choose ONE representation:
USE_EIGEN_SPECTRUM  = True      # permutation invariant, compact
USE_LOWER_TRIANGLE  = False     # longer, richer vector
# ==========================


# ---------- helpers ----------
def numeric_key(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    return (0, int(stem)) if re.fullmatch(r"\d+", stem) else (1, stem)

def list_xyz_files(xyz_dir):
    files = glob.glob(os.path.join(xyz_dir, "*.xyz"))
    files.sort(key=numeric_key)
    return files

def infer_ids(files):
    ids = []
    for p in files:
        stem = os.path.splitext(os.path.basename(p))[0]
        m = re.search(r"\d+", stem)
        ids.append(m.group(0) if m else stem)
    return np.asarray(ids, dtype=str)

def make_transformer():
    # Do NOT pass max_atoms (older molml rejects it); let it infer internally
    if USE_EIGEN_SPECTRUM:
        return CoulombMatrix(input_type="filename", eigen=True)
    if USE_LOWER_TRIANGLE:
        return CoulombMatrix(input_type="filename", eigen=False, only_lower_triangle=True)
    return CoulombMatrix(input_type="filename", eigen=False, only_lower_triangle=False)

def to_2d_numeric(X):
    """Coerce molml output (list/sparse/object array) to 2-D float32."""
    if hasattr(X, "toarray"):        # scipy sparse
        X = X.toarray()
    X = np.array(X, dtype=object)
    if X.ndim == 2 and X.dtype != object:
        return X.astype(np.float32, copy=False)
    if X.ndim == 1 and X.size > 0 and isinstance(X[0], (list, np.ndarray)):
        X = np.vstack(X)
        return X.astype(np.float32, copy=False)
    # last resort
    return X.astype(np.float32, copy=False)

def count_atoms_fast(xyz_path):
    with open(xyz_path, "r", encoding="utf-8", errors="ignore") as f:
        first = f.readline().strip()
    try:
        return int(first.split()[0])
    except Exception:
        m = re.search(r"\d+", first)
        return int(m.group(0)) if m else 0

def pick_probe_with_max(files, k=64):
    """Probe = first k files + file with max atom count (de-duplicated)."""
    sizes = [(count_atoms_fast(p), p) for p in files]
    max_file = max(sizes, key=lambda t: t[0])[1]
    probe = files[:min(len(files), k)]
    if max_file not in probe:
        probe = probe + [max_file]
    seen, out = set(), []
    for p in probe:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out

def write_csv_header(path, n_feat):
    with open(path, "w", encoding="utf-8") as f:
        header = ["id"] + [f"CM_{i+1}" for i in range(n_feat)]
        f.write(",".join(header) + "\n")

def append_csv_rows(path, ids, mat, fmt=FLOAT_FMT):
    ids = np.asarray(ids, dtype=str)
    mat = np.asarray(mat, dtype=np.float32)
    with open(path, "a", encoding="utf-8") as f:
        for idx, row in zip(ids, mat):
            f.write(idx)
            f.write(",")
            f.write(",".join(fmt.format(x) for x in row))
            f.write("\n")
# ---------------------------


def main():
    if not os.path.isdir(XYZ_DIR):
        raise SystemExit(f"[ERROR] Folder not found: {XYZ_DIR}")

    files = list_xyz_files(XYZ_DIR)
    if not files:
        raise SystemExit(f"[ERROR] No .xyz files found in: {XYZ_DIR}")
    ids = infer_ids(files)

    feat = make_transformer()

    # *** KEY FIX: ensure fit sees the largest molecule ***
    probe = pick_probe_with_max(files, k=64)
    feat.fit(probe)

    # Lock feature width
    probe_X = to_2d_numeric(feat.transform(probe))
    n_feat = probe_X.shape[1]
    write_csv_header(OUT_CSV, n_feat)

    all_rows = []

    def transform_safe(batch_files):
        """Transform with auto-retry: if a larger molecule appears, refit on all files once."""
        try:
            return to_2d_numeric(feat.transform(batch_files))
        except ValueError as e:
            msg = str(e)
            if "fit molecules" in msg and "not as large" in msg:
                # Refit on the entire set once, then retry
                feat.fit(files)
                X2 = to_2d_numeric(feat.transform(batch_files))
                return X2
            raise

    if BATCH_SIZE and BATCH_SIZE > 0:
        for start in range(0, len(files), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(files))
            batch_files = files[start:end]
            batch_ids   = ids[start:end]
            X = transform_safe(batch_files)
            if X.shape[1] != n_feat:
                raise RuntimeError(f"Feature width changed ({X.shape[1]} vs {n_feat}).")
            append_csv_rows(OUT_CSV, batch_ids, X)
            all_rows.append(X.astype(np.float32, copy=False))
            print(f"[OK] Wrote {start+1}–{end} / {len(files)}")
        cm = np.vstack(all_rows)
    else:
        X = transform_safe(files)
        if X.shape[1] != n_feat:
            raise RuntimeError(f"Feature width changed ({X.shape[1]} vs {n_feat}).")
        append_csv_rows(OUT_CSV, ids, X)
        cm = X.astype(np.float32, copy=False)

    mode = "eigen" if USE_EIGEN_SPECTRUM else ("lower_triangle" if USE_LOWER_TRIANGLE else "full")
    np.savez_compressed(
        OUT_NPZ,
        ids=ids,
        cm=cm,
        mode=mode,
        n_feat=n_feat,
        n_mols=len(ids),
        inferred_max=getattr(feat, "n_atoms_max_", None),
    )
    print(f"[DONE] CSV: {OUT_CSV}")
    print(f"[DONE] NPZ: {OUT_NPZ}  (n_mols={len(ids)}, feat_dim={n_feat}, mode={mode}, "
          f"max_atoms={getattr(feat, 'n_atoms_max_', None)})")


if __name__ == "__main__":
    main()
