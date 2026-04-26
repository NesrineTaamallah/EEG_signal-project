import os
import re
import numpy as np
import pandas as pd
import scipy.io
from features import extract_all_features
import variables as v



def load_labels():
    labels = pd.read_excel(v.LABELS_PATH, header=[0, 1])
    labels.set_index(labels.columns[0], inplace=True)
    labels = labels.astype(int) > 5  
    return labels



TEST_MAPPING = {
    "arithmetic": "Maths",
    "mirror_image": "Symmetry",
    "symmetry": "Symmetry",
    "stroop": "Stroop",
    "relax": "Relax"
}

def parse_filename_for_meta(filename):
    """
    Retourne: subject, raw_test_type, trial, excel_test_type
    """
    pattern = r"cleaned_(\w+)_sub_(\d+)_trial(\d+)\.mat"
    match = re.match(pattern, filename, re.IGNORECASE)

    if not match:
        return None, None, None, None

    raw_test_type = match.group(1).lower()
    subject = int(match.group(2))
    trial = int(match.group(3))
    excel_test_type = TEST_MAPPING.get(raw_test_type)

    return subject, raw_test_type, trial, excel_test_type




def get_label(labels_df, subject, test_type, trial):
    if subject not in labels_df.index:
        raise ValueError(f"Subject {subject} introuvable dans scales.xls")

    col = (f"Trial_{trial}", test_type)
    if col not in labels_df.columns:
        raise ValueError(f"Colonne {col} introuvable dans scales.xls")

    return labels_df.loc[subject, col]




def split_data(data, sfreq):
    sfreq = int(sfreq)

    if data.ndim == 2:
        data = data[np.newaxis, :, :]  

    n_trials, n_channels, n_samples = data.shape
    n_epochs = n_samples // sfreq

    epoched_data = np.empty((n_trials, n_epochs, n_channels, sfreq), dtype=float)

    for i in range(n_trials):
        for j in range(n_epochs):
            start = j * sfreq
            end = start + sfreq
            epoched_data[i, j] = data[i, :, start:end]

    return epoched_data



def create_features_dataframe(
    epoched_data,
    labels_all,
    filenames_all,
    epochs_per_file,
    sfreq=v.SFREQ,
    window_sec=1,
    overlap=0.5
):
    n_trials, n_epochs, n_channels, n_samples = epoched_data.shape

    window_samples = int(window_sec * sfreq)
    step = int(window_samples * (1 - overlap))
    windows_per_epoch = (n_samples - window_samples) // step + 1

    features = extract_all_features(
        epoched_data,
        sfreq=sfreq,
        window_sec=window_sec,
        overlap=overlap
    )

    total_windows = features.shape[0]
    total_epochs = n_trials * n_epochs

    if sum(epochs_per_file) != total_epochs:
        raise ValueError(
            f"Sum of epochs_per_file ({sum(epochs_per_file)}) != total_epochs ({total_epochs})"
        )

    expected_windows = total_epochs * windows_per_epoch
    if total_windows != expected_windows:
        raise ValueError(
            f"Mismatch windows: {total_windows} != {expected_windows}"
        )

    labels_windows = np.repeat(labels_all, windows_per_epoch)

    if len(labels_windows) != total_windows:
        raise ValueError("Labels / windows mismatch")

    time_names = ["variance", "rms", "ptp", "skewness", "kurtosis"]
    freq_names = ["delta_power", "theta_power", "alpha_power", "beta_power", "gamma_power"]
    hjorth_names = ["hj_activity", "hj_mobility", "hj_complexity"]
    fractal_names = ["higuchi_fd", "katz_fd"]
    entropy_names = ["approx_entropy", "sample_entropy", "spectral_entropy", "svd_entropy"]

    per_channel_names = (
        time_names + freq_names + hjorth_names + fractal_names + entropy_names
    )

    col_names = [
        f"ch{ch+1}_{name}"
        for ch in range(n_channels)
        for name in per_channel_names
    ]

    df = pd.DataFrame(features, columns=col_names)
    df["label"] = labels_windows

    filenames_col = []
    epochs_col = []
    windows_col = []

    for filename, n_ep in zip(filenames_all, epochs_per_file):
        for ep in range(n_ep):
            for w in range(windows_per_epoch):
                filenames_col.append(filename)
                epochs_col.append(ep)
                windows_col.append(w)

    df["filename"] = filenames_col
    df["epoch"] = epochs_col
    df["window_in_epoch"] = windows_col

    return df




def load_all_cleaned_with_features(
    cleaned_dir=v.DIR_CLEANED,
    sfreq=v.SFREQ,
    window_sec=1,
    overlap=0.5
):
    labels_df = load_labels()

    all_epoched = []
    all_labels = []
    all_filenames = []
    epochs_per_file = []

    for f in sorted(os.listdir(cleaned_dir)):
        if not f.endswith(".mat"):
            continue

        subject, raw_test_type, trial, test_type = parse_filename_for_meta(f)

        if subject is None or test_type is None:
            print(f"⚠ Ignoré : metadata invalide pour {f}")
            continue

        try:
            label = get_label(labels_df, subject, test_type, trial)
        except ValueError:
            print(f"⚠ Ignoré : test_type {test_type} non présent dans Excel pour {f}")
            continue

        mat = scipy.io.loadmat(os.path.join(cleaned_dir, f))
        if "data_cleaned" not in mat:
            print(f"⚠ Ignoré : data_cleaned absent dans {f}")
            continue

        data = mat["data_cleaned"]
        epochs = split_data(data, sfreq)

        all_epoched.append(epochs)

        n_epochs_file = epochs.shape[1]
        epochs_per_file.append(n_epochs_file)

        all_labels.append(np.repeat(label, n_epochs_file))
        all_filenames.append(f)

    if not all_epoched:
        raise ValueError("Aucun fichier valide chargé")

    epoched_data = np.concatenate(all_epoched, axis=0)
    labels_all = np.concatenate(all_labels)

    df = create_features_dataframe(
        epoched_data,
        labels_all,
        all_filenames,
        epochs_per_file,
        sfreq=sfreq,
        window_sec=window_sec,
        overlap=overlap
    )

    return df
