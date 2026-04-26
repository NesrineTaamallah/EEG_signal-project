import os
import numpy as np
import pandas as pd
import scipy.io
from features import (
    time_series_features, freq_band_features,
    hjorth_features, fractal_features, entropy_features,
)

CLEANED_DIR = r"C:\Users\nesri\OneDrive\Desktop\signal\data\Data\cleaned_data"
SFREQ = 256.0
WINDOW_SEC = 1
OVERLAP = 0.5
FREQ_BANDS = np.array([0.5, 4, 8, 12, 30, 45])


def load_cleaned_mat(file_path):
    mat = scipy.io.loadmat(file_path)
    return mat['data_cleaned']


def frame_signal(signal, sfreq=256.0, window_sec=1, overlap_sec=0.5):
    """Return (n_frames, n_channels, window_len) — already windowed with Hann."""
    n_channels, n_samples = signal.shape
    window_len = int(window_sec * sfreq)
    step = int((window_sec - overlap_sec) * sfreq)
    n_frames = (n_samples - window_len) // step + 1

    frames = np.zeros((n_frames, n_channels, window_len))
    hann_win = np.hanning(window_len)

    for i in range(n_frames):
        start = i * step
        end = start + window_len
        frames[i] = signal[:, start:end] * hann_win
    return frames


def extract_features_from_frames(frames):
    """
    Accept already-windowed frames (n_frames, n_channels, window_len) and
    compute all feature groups without re-windowing.
    Reshape to (n_frames, 1, n_channels, window_len) so the feature
    functions see shape (n_trials=n_frames, n_secs=1, n_channels, sfreq).
    """
    # (n_frames, 1, n_channels, window_len)
    data = frames[:, np.newaxis, :, :]

    t_feats  = time_series_features(data)
    f_feats  = freq_band_features(data, FREQ_BANDS)
    h_feats  = hjorth_features(data)
    fr_feats = fractal_features(data)
    e_feats  = entropy_features(data)

    return np.hstack([t_feats, f_feats, h_feats, fr_feats, e_feats])


all_features_list = []
all_file_info = []

for filename in sorted(os.listdir(CLEANED_DIR)):
    if not filename.endswith('.mat'):
        continue

    file_path = os.path.join(CLEANED_DIR, filename)
    try:
        data = load_cleaned_mat(file_path)
        n_channels, n_samples = data.shape

        frames = frame_signal(data, sfreq=SFREQ, window_sec=WINDOW_SEC,
                              overlap_sec=OVERLAP)

        feats = extract_features_from_frames(frames)

        for i in range(feats.shape[0]):
            all_features_list.append(feats[i])
            all_file_info.append({'filename': filename, 'frame_idx': i})

        print(f"✓ {filename} traité: {feats.shape[0]} frames extraites")

    except Exception as e:
        print(f"Erreur pour {filename}: {e}")

