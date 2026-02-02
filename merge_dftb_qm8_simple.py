#!/usr/bin/env python3
import csv, os, re, sys

CC2_FILE = "qm8_CC2_only.csv"     # gdb9_index,E1-CC2,E2-CC2,f1-CC2,f2-CC2
OS1_SRC  = "EX_OS1"               # file OR directory → E1-DFTB,f1-DFTB
OS2_SRC  = "EX_OS2"               # file OR directory → E2-DFTB,f2-DFTB
OUT_FILE = "merged_ml_simple.csv"

def extract_index(fname: str):
    m = re.search(r"(\d+)", fname)
    return int(m.group(1)) if m else None

def read_first_two_floats_from_file(path: str):
    """For per-molecule files: return (E,f) from first non-comment line with ≥2 floats."""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#") or s.startswith("!") or s.startswith("//"):
                    continue
                parts = s.replace(",", " ").split()
                vals = []
                for p in parts:
                    try:
                        vals.append(float(p))
                    except:
                        pass
                if len(vals) >= 2:
                    return vals[0], vals[1]
    except:
        pass
    return None, None

def load_source_as_dir(d: str):
    """Return dict idx->(E,f) from a directory of per-molecule files."""
    out = {}
    for fn in os.listdir(d):
        full = os.path.join(d, fn)
        if not os.path.isfile(full):
            continue
        idx = extract_index(fn)
        if idx is None:
            continue
        e, f = read_first_two_floats_from_file(full)
        out[idx] = (e, f)
    return out, []  # dict, order-list empty

def load_source_as_file(path: str):
    """
    Read a single file. Accept lines:
      a) idx energy osc  (>=3 tokens, first is integer)
      b) energy osc      (>=2 tokens, no index) → use row order (1-based)
    Returns (dict_by_idx, list_by_order).
    """
    dict_by_idx = {}
    list_by_order = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#") or s.startswith("!") or s.startswith("//"):
                continue
            parts = s.replace(",", " ").split()

            # try pattern a) idx, e, f
            if len(parts) >= 3 and parts[0].isdigit():
                try:
                    idx = int(parts[0])
                    e = float(parts[1]); ff = float(parts[2])
                    dict_by_idx[idx] = (e, ff)
                    continue
                except:
                    pass

            # try pattern b) e, f (no index)
            vals = []
            for p in parts:
                try:
                    vals.append(float(p))
                except:
                    pass
            if len(vals) >= 2:
                list_by_order.append((vals[0], vals[1]))

    return dict_by_idx, list_by_order

def load_source(path_or_dir: str):
    if os.path.isdir(path_or_dir):
        return load_source_as_dir(path_or_dir)
    if os.path.isfile(path_or_dir):
        return load_source_as_file(path_or_dir)
    print(f"ERROR: '{path_or_dir}' not found.", file=sys.stderr)
    return {}, []

def main():
    print("Loading DFTB OS1/OS2 ...")
    os1_dict, os1_list = load_source(OS1_SRC)   # E1,f1
    os2_dict, os2_list = load_source(OS2_SRC)   # E2,f2

    print("Merging with CC2 ...")
    with open(CC2_FILE, newline="") as fin, open(OUT_FILE, "w", newline="") as fout:
        r = csv.DictReader(fin)
        fieldnames = ["gdb9_index","E1-DFTB","f1-DFTB","E2-DFTB","f2-DFTB","E1-CC2","E2-CC2","f1-CC2","f2-CC2"]
        w = csv.DictWriter(fout, fieldnames=fieldnames)
        w.writeheader()

        # counters for list-by-order fallback
        os1_i = 0
        os2_i = 0
        rows = 0

        for row in r:
            try:
                idx = int(float(row["gdb9_index"]))
            except:
                idx = None

            # Prefer dict (by index). If missing, fall back to list-by-order.
            if idx in os1_dict:
                e1, f1 = os1_dict[idx]
            else:
                e1, f1 = (os1_list[os1_i] if os1_i < len(os1_list) else (None, None))
                if os1_list:
                    os1_i += 1

            if idx in os2_dict:
                e2, f2 = os2_dict[idx]
            else:
                e2, f2 = (os2_list[os2_i] if os2_i < len(os2_list) else (None, None))
                if os2_list:
                    os2_i += 1

            w.writerow({
                "gdb9_index": idx if idx is not None else row.get("gdb9_index"),
                "E1-DFTB": e1, "f1-DFTB": f1,
                "E2-DFTB": e2, "f2-DFTB": f2,
                "E1-CC2": row.get("E1-CC2"),
                "E2-CC2": row.get("E2-CC2"),
                "f1-CC2": row.get("f1-CC2"),
                "f2-CC2": row.get("f2-CC2"),
            })
            rows += 1

    print(f"✅ Wrote {OUT_FILE} (rows: {rows}, columns: 9)")
    print("Columns: gdb9_index,E1-DFTB,f1-DFTB,E2-DFTB,f2-DFTB,E1-CC2,E2-CC2,f1-CC2,f2-CC2")

if __name__ == "__main__":
    main()
