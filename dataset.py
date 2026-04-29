"""
dataset.py — Relax files treated as score = 5 → label = 0 (non-stress).

Score threshold: label = (score > 5)
Relax score = 5  →  5 > 5 = False  →  label = 0

This keeps all 120 Relax recordings as clean non-stress baseline data
instead of silently dropping them.

ALSO FIXED: column names generated dynamically from actual feature count,
so this file is compatible with any version of features.py.
"""

import os
import re
import numpy as np
import pandas as pd
import scipy.io
from features import extract_all_features
import variables as v


# ─────────────────────────────────────────────────────────────────────────────
# Label loading from scales.xls
# ─────────────────────────────────────────────────────────────────────────────

def load_labels():
    labels = pd.read_excel(v.LABELS_PATH, header=[0, 1])
    labels.set_index(labels.columns[0], inplace=True)
    labels = labels.astype(int) > 5   # True = stress
    return labels


# ─────────────────────────────────────────────────────────────────────────────
# Filename → metadata
# ─────────────────────────────────────────────────────────────────────────────

# Maps lowercase test token from filename → column header in scales.xls
TEST_MAPPING = {
    "arithmetic":   "Maths",
    "mirror_image": "Symmetry",
    "symmetry":     "Symmetry",
    "stroop":       "Stroop",
    # "relax" is handled separately with a fixed score of 5
}

# Relax: we assign score = 5  →  5 > 5 = False  →  label = 0
RELAX_SCORE = 5
RELAX_LABEL = int(RELAX_SCORE > 5)   # = 0


def parse_filename_for_meta(filename):
    """
    Parse a filename like cleaned_Arithmetic_sub_3_trial2.mat.
    Returns (subject, raw_test_type, trial, excel_test_type).
    excel_test_type is None when the test has no column in scales.xls.
    """
    pattern = r"cleaned_(\w+)_sub_(\d+)_trial(\d+)\.mat"
    match   = re.match(pattern, filename, re.IGNORECASE)

    if not match:
        return None, None, None, None

    raw_test_type   = match.group(1).lower()
    subject         = int(match.group(2))
    trial           = int(match.group(3))
    excel_test_type = TEST_MAPPING.get(raw_test_type)   # None for "relax"

    return subject, raw_test_type, trial, excel_test_type


# ─────────────────────────────────────────────────────────────────────────────
# Label lookup from scales.xls
# ─────────────────────────────────────────────────────────────────────────────

def get_label(labels_df, subject, test_type, trial):
    if subject not in labels_df.index:
        raise ValueError(f"Subject {subject} not found in scales.xls")

    col = (f"Trial_{trial}", test_type)
    if col not in labels_df.columns:
        raise ValueError(
            f"Column {col} not found in scales.xls. "
            f"Available: {labels_df.columns.tolist()}"
        )

    return labels_df.loc[subject, col]


# ─────────────────────────────────────────────────────────────────────────────
# Signal splitting into fixed-length epochs
# ─────────────────────────────────────────────────────────────────────────────

def split_data(data, sfreq):
    """
    Split (n_channels, n_samples) into (1, n_epochs, n_channels, sfreq).
    The leading trial axis is always 1 for single-file loading.
    """
    sfreq = int(sfreq)

    if data.ndim == 2:
        data = data[np.newaxis, :, :]   # (1, n_ch, n_samp)

    n_trials, n_channels, n_samples = data.shape
    n_epochs = n_samples // sfreq

    epoched = np.empty((n_trials, n_epochs, n_channels, sfreq), dtype=float)
    for i in range(n_trials):
        for j in range(n_epochs):
            start          = j * sfreq
            epoched[i, j]  = data[i, :, start: start + sfreq]

    return epoched


# ─────────────────────────────────────────────────────────────────────────────
# Dynamic column-name generation
# ─────────────────────────────────────────────────────────────────────────────

def _make_col_names(n_features: int, n_channels: int) -> list:
    """
    Build column names that match the ACTUAL number of features extracted
    by features.py.  Falls back to generic 'f0', 'f1', ... if the
    computed total doesn't match n_features (future-proof).
    """
    FREQ_BANDS = [0.5, 4, 8, 12, 30, 45]
    n_bands    = len(FREQ_BANDS) - 1   # 5

    # Per-channel feature counts (must mirror features.extract_all_features)
    per_ch = {
        "time":    5,
        "absband": n_bands,
        "relband": n_bands,
        "ratios":  4,
        "sef":     2,
        "zcll":    2,
        "hjorth":  3,
        "fractal": 2,
        "entropy": 4,
        "wavelet": 5,
    }

    n_pairs    = min(5, n_channels // 2) if n_channels >= 2 else 0
    asym_total = max(n_pairs * n_bands, 1)   # ≥1 so there's always a column

    per_ch_total   = sum(per_ch.values()) * n_channels
    computed_total = per_ch_total + asym_total

    if computed_total == n_features:
        col_names = []
        for group, count in per_ch.items():
            for ch in range(n_channels):
                for k in range(count):
                    col_names.append(f"ch{ch + 1}_{group}_{k}")
        for pair in range(n_pairs):
            for b in range(n_bands):
                col_names.append(f"asym_pair{pair + 1}_band{b}")
        if n_pairs == 0:
            col_names.append("asym_placeholder")
        return col_names

    # Fallback
    print(
        f"[dataset] WARNING: computed column count ({computed_total}) "
        f"!= actual feature count ({n_features}). "
        f"Using generic column names f0...f{n_features - 1}."
    )
    return [f"f{i}" for i in range(n_features)]


# ─────────────────────────────────────────────────────────────────────────────
# DataFrame builder
# ─────────────────────────────────────────────────────────────────────────────

def create_features_dataframe(
    epoched_data,
    labels_all,
    filenames_all,
    epochs_per_file,
    sfreq=v.SFREQ,
    window_sec=1,
    overlap=0.5,
):
    """
    Extract features and return a labelled DataFrame.

    epoched_data   : (n_files, n_epochs, n_channels, sfreq_samples)
    labels_all     : (total_epochs,)
    filenames_all  : list[str], len = n_files
    epochs_per_file: list[int], len = n_files
    """
    n_trials, n_epochs, n_channels, n_samples = epoched_data.shape

    window_samples    = int(window_sec * sfreq)
    step              = int(window_samples * (1 - overlap))
    windows_per_epoch = max((n_samples - window_samples) // step + 1, 1)

    # ── Feature extraction ────────────────────────────────────────────────────
    features = extract_all_features(
        epoched_data,
        sfreq=sfreq,
        window_sec=window_sec,
        overlap=overlap,
    )

    n_total_windows = features.shape[0]
    n_feature_cols  = features.shape[1]
    total_epochs    = n_trials * n_epochs

    # ── Sanity checks ──────────────────────────────────────────────────────────
    if sum(epochs_per_file) != total_epochs:
        raise ValueError(
            f"epochs_per_file sum ({sum(epochs_per_file)}) "
            f"!= total_epochs ({total_epochs})"
        )

    expected_windows = total_epochs * windows_per_epoch
    if n_total_windows != expected_windows:
        raise ValueError(
            f"Feature row count mismatch: "
            f"got {n_total_windows}, expected {expected_windows} "
            f"({total_epochs} epochs x {windows_per_epoch} windows/epoch)"
        )

    # ── Column names ──────────────────────────────────────────────────────────
    col_names = _make_col_names(n_feature_cols, n_channels)

    # ── Labels: one per window ────────────────────────────────────────────────
    labels_windows = np.repeat(labels_all, windows_per_epoch)
    if len(labels_windows) != n_total_windows:
        raise ValueError(
            f"Label/window mismatch: "
            f"{len(labels_windows)} labels, {n_total_windows} windows"
        )

    # ── Assemble ───────────────────────────────────────────────────────────────
    df = pd.DataFrame(features, columns=col_names)
    df["label"] = labels_windows.astype(int)

    filenames_col = []
    epochs_col    = []
    windows_col   = []

    for filename, n_ep in zip(filenames_all, epochs_per_file):
        for ep in range(n_ep):
            for w in range(windows_per_epoch):
                filenames_col.append(filename)
                epochs_col.append(ep)
                windows_col.append(w)

    df["filename"]         = filenames_col
    df["epoch"]            = epochs_col
    df["window_in_epoch"]  = windows_col

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Top-level loader
# ─────────────────────────────────────────────────────────────────────────────

def load_all_cleaned_with_features(
    cleaned_dir=v.DIR_CLEANED,
    sfreq=v.SFREQ,
    window_sec=1,
    overlap=0.5,
):
    """
    Load every .mat in cleaned_dir, extract features, return a DataFrame.

    Label rules
    -----------
    Arithmetic / Mirror / Stroop  →  look up score in scales.xls
                                     label = (score > 5)
    Relax                         →  score fixed at 5
                                     label = (5 > 5) = 0  (non-stress)
    Anything else                 →  skipped with a warning
    """
    labels_df = load_labels()

    all_epoched     = []
    all_labels      = []
    all_filenames   = []
    epochs_per_file = []

    n_loaded      = 0
    n_relax       = 0
    n_skip_meta   = 0
    n_skip_label  = 0
    n_skip_data   = 0

    for f in sorted(os.listdir(cleaned_dir)):
        if not f.endswith(".mat"):
            continue

        subject, raw_test_type, trial, test_type = parse_filename_for_meta(f)

        # ── Pattern check ─────────────────────────────────────────────────────
        if subject is None:
            print(f"  [skip-pattern] {f}")
            n_skip_meta += 1
            continue

        # ── Determine label ────────────────────────────────────────────────────
        if raw_test_type == "relax":
            # Score = 5  →  5 > 5 = False  →  label = 0
            label   = RELAX_LABEL   # 0
            n_relax += 1

        elif test_type is not None:
            # Normal condition: fetch from scales.xls
            try:
                label = get_label(labels_df, subject, test_type, trial)
            except ValueError as exc:
                print(f"  [skip-label] {f}  ({exc})")
                n_skip_label += 1
                continue

        else:
            print(f"  [skip-unknown] '{raw_test_type}' not mapped → {f}")
            n_skip_meta += 1
            continue

        # ── Load EEG ──────────────────────────────────────────────────────────
        mat_path = os.path.join(cleaned_dir, f)
        mat      = scipy.io.loadmat(mat_path)

        if "data_cleaned" not in mat:
            print(f"  [skip-nokey] 'data_cleaned' missing → {f}")
            n_skip_data += 1
            continue

        data   = mat["data_cleaned"]
        epochs = split_data(data, sfreq)
        n_ep   = epochs.shape[1]

        all_epoched.append(epochs)
        epochs_per_file.append(n_ep)
        all_labels.append(np.repeat(int(label), n_ep))
        all_filenames.append(f)
        n_loaded += 1

    # ── Guard ─────────────────────────────────────────────────────────────────
    if not all_epoched:
        raise ValueError(
            "No valid files loaded. "
            "Check cleaned_dir path and file naming convention."
        )

    # ── Summary ───────────────────────────────────────────────────────────────
    print(
        f"\nFiles loaded   : {n_loaded}  "
        f"(incl. {n_relax} Relax files scored as 5 → label=0)"
    )
    print(
        f"Files skipped  : {n_skip_meta} bad-pattern/unknown, "
        f"{n_skip_label} no-label, "
        f"{n_skip_data} no-data-key"
    )

    epoched_data = np.concatenate(all_epoched, axis=0)
    labels_all   = np.concatenate(all_labels)

    n_stress    = int((labels_all == 1).sum())
    n_nonstress = int((labels_all == 0).sum())
    print(
        f"Total epochs   : {len(labels_all)}  "
        f"(stress={n_stress}, non-stress={n_nonstress})"
    )

    df = create_features_dataframe(
        epoched_data,
        labels_all,
        all_filenames,
        epochs_per_file,
        sfreq=sfreq,
        window_sec=window_sec,
        overlap=overlap,
    )

    return df