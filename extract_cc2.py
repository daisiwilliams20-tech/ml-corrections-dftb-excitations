#!/usr/bin/env python3
import pandas as pd

# === Input and output paths ===
input_file = "gdb8_22k_elec_spec.txt"
output_file = "gdb8_CC2_only.csv"

# === Read only the CC2 block (Index + 4 CC2 columns) ===
df = pd.read_csv(
    input_file,
    delim_whitespace=True,
    comment="#",
    header=None,
    usecols=[0, 1, 2, 3, 4],   # Index + E1-CC2 + E2-CC2 + f1-CC2 + f2-CC2
    names=["gdb9_index", "E1-CC2", "E2-CC2", "f1-CC2", "f2-CC2"],
    engine="python"
)

# === Save as new CSV file ===
df.to_csv(output_file, index=False)
print(f"✅ Successfully extracted CC2 columns → {output_file}")

