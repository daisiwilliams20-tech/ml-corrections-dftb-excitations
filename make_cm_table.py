#!/usr/bin/env python3
"""
Compute Coulomb Matrix (sorted & padded) for all .xyz files in moles/
and save to NPZ + CSV (CSV written without pandas to avoid dtype issues).

Run:  python3 make_cm_table_fixed.py
"""

import os
import re
import numpy as np

# ========= CONFIG (tailored to your machine) =========
XYZ_DIR    = "/home/williams/Downloads/send_Williams (2)/moles"
OUT_CSV    = "/home/williams/Downloads/send_Williams (2)/cm_vectors.csv"
OUT_NPZ    = "/home/williams/Downloads/send_Williams (2)/cm_vectors.npz"
UPPER_ONLY = False   # set True to save only upper triangle (smaller files)
FLOAT_FMT  = "{:.8f}"  # CSV numeric format
# =====================================================

# Extend if needed
ZMAP = {
    "H":1, "He":2,
    "Li":3, "Be":4, "B":5, "C":6, "N":7, "O":8, "F":9, "Ne":10,
    "Na":11, "Mg":12, "Al":13, "Si":14, "P":15, "S":16, "Cl":17, "Ar":18,
    # add more if needed, e.g., "Br":35, "I":53
}

def read_xyz(path):
    """Read simple XYZ → (Z, R)."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if len(lines) < 3:
        raise ValueError(f"Bad XYZ: {path}")
    try:
        nat = int(lines[0].split()[0])
    except Exception:
        nat = int(re.findall(r"\d+", lines[0])[0])
    atoms = lines[2:2+nat]
    Z, R = [], []
    for a in atoms:
        parts = a.split()
        sym = parts[0]
        if sym not in ZMAP:
            raise ValueError(f"Element {sym} not in ZMAP (file {path})")
        Z.append(ZMAP[sym])
        R.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.asarray(Z, int), np.asarray(R, float)

def coulomb_matrix(Z, R):
    """Unsorted Coulomb Matrix (n×n)."""
    n = len(Z)
    M = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j:
                M[i, i] = 0.5 * (Z[i] ** 2.4)
            else:
                d = np.linalg.norm(R[i] - R[j])
                M[i, j] = (Z[i]*Z[j] / d) if d > 1e-12 else 0.0
    return M

def sort_rows_by_l2(M):
    """Sort rows/cols by row L2 norm (descending)."""
    if M.size == 0:
        return M
    order = np.argsort(np.linalg.norm(M, axis=1))[::-1]
    return M[order][:, order]

def pad_and_flatten(M, max_atoms, upper_only=False):
    """Pad to (max_atoms×max_atoms), then flatten (or take upper-tri)."""
    n = M.shape[0]
    P = np.zeros((max_atoms, max_atoms), dtype=float)
    P[:n, :n] = M
    if upper_only:
        iu = np.triu_indices(max_atoms)
        return P[iu]
    return P.ravel()

def infer_id_from_filename(fname):
    """e.g., '1234' from '1234.xyz'."""
    stem = os.path.splitext(os.path.basename(fname))[0]
    m = re.search(r"\d+", stem)
    return m.group(0) if m else stem

def detect_max_atoms(xyz_files):
    max_atoms = 0
    for p in xyz_files:
        try:
            Z, _ = read_xyz(p)
            max_atoms = max(max_atoms, len(Z))
        except Exception:
            continue
    return max_atoms

def write_csv(ids, cm, path, float_fmt=FLOAT_FMT):
    """Write CSV without pandas to avoid dtype issues."""
    ids = np.asarray(ids, dtype=str)
    cm  = np.asarray(cm,  dtype=np.float32)
    n, k = cm.shape
    header = ["gdb9_index"] + [f"CM_{i+1}" for i in range(k)]
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for i in range(n):
            row = [ids[i]] + [float_fmt.format(x) for x in cm[i]]
            f.write(",".join(row) + "\n")

def main():
    if not os.path.isdir(XYZ_DIR):
        print(f"[ERROR] Folder not found: {XYZ_DIR}")
        return 1

    files = sorted(
        os.path.join(XYZ_DIR, f)
        for f in os.listdir(XYZ_DIR)
        if f.lower().endswith(".xyz")
    )
    if not files:
        print(f"[ERROR] No .xyz files found in '{XYZ_DIR}'.")
        return 1

    max_atoms = detect_max_atoms(files)
    if max_atoms <= 0:
        print("[ERROR] Could not detect any valid molecules.")
        return 1

    print(f"[INFO] Found {len(files)} xyz files. Using max_atoms={max_atoms}.")

    ids, rows, bad = [], [], 0
    for p in files:
        try:
            mol_id = infer_id_from_filename(p)
            Z, R = read_xyz(p)
            M = coulomb_matrix(Z, R)
            M = sort_rows_by_l2(M)
            vec = pad_and_flatten(M, max_atoms, upper_only=UPPER_ONLY)
            ids.append(mol_id)
            rows.append(vec.astype(np.float32))
        except Exception:
            bad += 1
            continue

    if not rows:
        print("[ERROR] All files failed to parse.")
        return 1

    cm = np.vstack(rows)
    ids = np.array(ids, dtype=str)

    # Save NPZ
    np.savez_compressed(
        OUT_NPZ,
        ids=ids,
        cm=cm,
        max_atoms=max_atoms,
        upper_only=UPPER_ONLY
    )
    print(f"[OK] Saved NPZ: {OUT_NPZ}  (n_mols={len(ids)}, feat_dim={cm.shape[1]})")

    # Save CSV (no pandas)
    write_csv(ids, cm, OUT_CSV)
    print(f"[OK] Saved CSV: {OUT_CSV}")

    if bad:
        print(f"[INFO] Skipped {bad} file(s) due to read/parse issues.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
