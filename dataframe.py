"""
dataframe.py — Build and save the feature DataFrame.

Uses the updated dataset.py / features.py pipeline.
Relax files are now included as non-stress (label=0).
"""

import os
import pandas as pd
from dataset import load_all_cleaned_with_features
import variables as v

# ── Parameters — must match what classifier.py expects ───────────────────────
WINDOW_SEC = 1      # keep at 1 s for the dataframe; classifier.py uses 2 s
                    # for DL raw windows (those are loaded separately)
OVERLAP    = 0.5

output_path = r"C:\Users\nesri\OneDrive\Desktop\signal\data\Data\dataframe.csv"

# ── Build DataFrame ───────────────────────────────────────────────────────────
print("Building feature DataFrame...")
df = load_all_cleaned_with_features(
    cleaned_dir=v.DIR_CLEANED,
    sfreq=v.SFREQ,
    window_sec=WINDOW_SEC,
    overlap=OVERLAP,
)

# ── Validation report ─────────────────────────────────────────────────────────
print("\n── DataFrame summary ──────────────────────────────────────────────")
print(f"Shape       : {df.shape}")
print(f"Columns     : {len(df.columns)} total")
print(f"Features    : {len(df.columns) - 4}  (excl. label/filename/epoch/window)")
print(f"Label dist  : {df['label'].value_counts().to_dict()}")
print(f"NaN count   : {df.isna().sum().sum()}")
print(f"Inf count   : {(df.replace([float('inf'), float('-inf')], float('nan')).isna().sum().sum())}")

# ── Replace any inf/nan with column medians (safety net) ─────────────────────
import numpy as np
feat_cols = [c for c in df.columns if c not in ("label", "filename", "epoch", "window_in_epoch")]
for col in feat_cols:
    bad = ~np.isfinite(df[col])
    if bad.any():
        median_val = df.loc[~bad, col].median() if (~bad).any() else 0.0
        df.loc[bad, col] = median_val

print(f"NaN/Inf after fix: {df.isna().sum().sum()}")

# ── Save ──────────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df.to_csv(output_path, index=False)
print(f"\nDataFrame saved → {output_path}")
print(f"File size: {os.path.getsize(output_path) / 1e6:.1f} MB")