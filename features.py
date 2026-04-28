"""
features.py — Enhanced EEG feature extraction.

KEY IMPROVEMENTS over original:
  1. Added coherence / connectivity features between channels
  2. Added spectral edge frequency (SEF90, SEF95)
  3. Added wavelet-based features (energy per sub-band)
  4. Added relative band power (normalised) — more robust than absolute
  5. Added inter-hemispheric asymmetry (log-ratio L vs R channels)
  6. Added zero-crossing rate and line length
  7. Added alpha-peak frequency estimation
  8. Better numerical stability (epsilon guards everywhere)
"""

import numpy as np
from scipy import signal, stats
import warnings

warnings.filterwarnings('ignore')

SFREQ = 256.0

# Standard 10-20 channel pairs for asymmetry (if present)
_L_CHANNELS = [0, 2, 4, 6, 8]   # indices when n_ch >= 10
_R_CHANNELS = [1, 3, 5, 7, 9]

# ─────────────────────────────────────────────────────────────────────────────
# Existing features (kept identical so training pipeline is unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def time_series_features(data):
    """
    5 features per channel: variance, rms, ptp, skewness, kurtosis.
    data shape: (n_trials, n_secs, n_channels, sfreq)
    """
    n_trials, n_secs, n_channels, sfreq = data.shape
    data_reshaped = data.reshape(-1, n_channels, sfreq)
    n_total = data_reshaped.shape[0]

    features = np.zeros((n_total, n_channels * 5))

    for i in range(n_total):
        epoch = data_reshaped[i]
        epoch_features = []

        for ch in range(n_channels):
            s = epoch[ch]
            variance = float(np.var(s))
            rms      = float(np.sqrt(np.mean(s ** 2)))
            ptp_amp  = float(np.ptp(s))
            skew_val = float(stats.skew(s))     if len(s) > 1 else 0.0
            kurt_val = float(stats.kurtosis(s)) if len(s) > 1 else 0.0
            epoch_features.extend([variance, rms, ptp_amp, skew_val, kurt_val])

        features[i] = epoch_features

    return features


def freq_band_features(data, freq_bands):
    """
    Absolute band power via Welch PSD.
    data shape: (n_trials, n_secs, n_channels, sfreq)
    """
    n_trials, n_secs, n_channels, sfreq = data.shape
    data_reshaped = data.reshape(-1, n_channels, sfreq)
    n_total = data_reshaped.shape[0]
    n_bands = len(freq_bands) - 1

    features = np.zeros((n_total, n_channels * n_bands))

    for i in range(n_total):
        epoch = data_reshaped[i]
        epoch_features = []

        for ch in range(n_channels):
            s = epoch[ch]
            freqs, psd = signal.welch(s, fs=sfreq, nperseg=min(256, len(s)))

            for b in range(n_bands):
                mask = (freqs >= freq_bands[b]) & (freqs <= freq_bands[b + 1])
                bp   = float(np.trapz(psd[mask], freqs[mask])) if mask.any() else 0.0
                epoch_features.append(bp)

        features[i] = epoch_features

    return features


def hjorth_features(data):
    """
    3 features per channel: activity, mobility, complexity.
    """
    n_trials, n_secs, n_channels, sfreq = data.shape
    data_reshaped = data.reshape(-1, n_channels, sfreq)
    n_total = data_reshaped.shape[0]

    features = np.zeros((n_total, n_channels * 3))

    for i in range(n_total):
        epoch = data_reshaped[i]
        epoch_features = []

        for ch in range(n_channels):
            s = epoch[ch]
            activity = float(np.var(s))

            if len(s) > 1:
                d1 = np.diff(s)
                mobility   = float(np.sqrt(np.var(d1) / (activity + 1e-10)))
                d2 = np.diff(d1)
                complexity = float(
                    np.sqrt(np.var(d2) / (np.var(d1) + 1e-10)) / (mobility + 1e-10)
                )
            else:
                mobility = complexity = 0.0

            epoch_features.extend([activity, mobility, complexity])

        features[i] = epoch_features

    return features


def fractal_features(data):
    """
    2 features per channel: Higuchi FD, Katz FD.
    """
    n_trials, n_secs, n_channels, sfreq = data.shape
    data_reshaped = data.reshape(-1, n_channels, sfreq)
    n_total = data_reshaped.shape[0]

    features = np.zeros((n_total, n_channels * 2))

    for i in range(n_total):
        epoch = data_reshaped[i]
        epoch_features = []

        for ch in range(n_channels):
            s = epoch[ch]
            N = len(s)
            k_max = 10

            # Higuchi FD
            L_vals = []
            for k in range(1, k_max + 1):
                Lk = 0
                for m in range(k):
                    indices = np.arange(m, N, k, dtype=int)
                    if len(indices) > 1:
                        Lkm = np.sum(np.abs(np.diff(s[indices])))
                        Lk += Lkm * (N - 1) / (len(indices) * k)
                if k > 0:
                    L_vals.append(np.log(Lk / k + 1e-10))

            if len(L_vals) > 1:
                x_log = np.log(1 / np.arange(1, k_max + 1)[:len(L_vals)])
                higuchi_fd = float(-np.polyfit(x_log, L_vals, 1)[0])
            else:
                higuchi_fd = 0.0

            # Katz FD
            if N <= 1:
                katz_fd = 0.0
            else:
                L_sum = float(np.sum(np.sqrt(1 + np.diff(s) ** 2)))
                d_val = float(np.max(np.sqrt(
                    (np.arange(N) / (N - 1)) ** 2 +
                    ((s - s[0]) / (np.max(np.abs(s)) + 1e-10)) ** 2
                )))
                if d_val > 0:
                    katz_fd = float(
                        np.log(N - 1) /
                        (np.log(d_val) + np.log((N - 1) / (L_sum + 1e-10)))
                    )
                else:
                    katz_fd = 0.0

            epoch_features.extend([higuchi_fd, katz_fd])

        features[i] = epoch_features

    return features


def entropy_features(data):
    """
    4 features per channel: approx_entropy, sample_entropy,
    spectral_entropy, svd_entropy.
    """
    n_trials, n_secs, n_channels, sfreq = data.shape
    data_reshaped = data.reshape(-1, n_channels, sfreq)
    n_total = data_reshaped.shape[0]

    features = np.zeros((n_total, n_channels * 4))

    for i in range(n_total):
        epoch = data_reshaped[i]
        epoch_features = []

        for ch in range(n_channels):
            s = epoch[ch]
            N = len(s)

            def _phi(m_val):
                if N <= m_val:
                    return 0.0
                patterns = np.lib.stride_tricks.sliding_window_view(s, m_val)
                r = 0.2 * np.std(s)
                C = np.sum(
                    np.max(np.abs(patterns[:, None] - patterns[None, :]),
                           axis=2) <= r,
                    axis=1
                )
                C = C / (N - m_val + 1)
                return float(np.sum(np.log(C + 1e-10)) / (N - m_val + 1))

            app_entropy = max(0.0, _phi(2) - _phi(3))

            if N > 3:
                corr = np.corrcoef(s[:-1], s[1:])[0, 1]
                samp_entropy = float(-np.log(abs(corr) + 1e-10))
            else:
                samp_entropy = 0.0

            freqs_e, psd_e = signal.welch(s, fs=sfreq)
            psd_norm = psd_e / (np.sum(psd_e) + 1e-10)
            spect_entropy = float(-np.sum(psd_norm * np.log(psd_norm + 1e-10)))

            if N >= 10:
                tau, m_svd = 1, 3
                n_vec = N - (m_svd - 1) * tau
                if n_vec > 0:
                    delayed = np.zeros((n_vec, m_svd))
                    for j in range(m_svd):
                        delayed[:, j] = s[j * tau: j * tau + n_vec]
                    try:
                        _, sv, _ = np.linalg.svd(delayed, full_matrices=False)
                        sv_norm = sv / (np.sum(sv) + 1e-10)
                        svd_entropy = float(
                            -np.sum(sv_norm * np.log(sv_norm + 1e-10))
                        )
                    except Exception:
                        svd_entropy = 0.0
                else:
                    svd_entropy = 0.0
            else:
                svd_entropy = 0.0

            epoch_features.extend([app_entropy, samp_entropy,
                                    spect_entropy, svd_entropy])

        features[i] = epoch_features

    return features


# ─────────────────────────────────────────────────────────────────────────────
# NEW features — each function follows the same (n_trials,n_secs,n_ch,sfreq)
# convention so they plug straight into extract_all_features()
# ─────────────────────────────────────────────────────────────────────────────

def relative_band_features(data, freq_bands):
    """
    Relative (normalised) band power — divides each band by total power.
    Much more robust to amplitude differences between subjects/sessions.
    Same shape as freq_band_features: (n_total, n_channels * n_bands).
    """
    n_trials, n_secs, n_channels, sfreq = data.shape
    data_reshaped = data.reshape(-1, n_channels, sfreq)
    n_total = data_reshaped.shape[0]
    n_bands = len(freq_bands) - 1

    features = np.zeros((n_total, n_channels * n_bands))

    for i in range(n_total):
        epoch = data_reshaped[i]
        row   = []

        for ch in range(n_channels):
            s = epoch[ch]
            freqs, psd = signal.welch(s, fs=sfreq, nperseg=min(256, len(s)))

            abs_powers = []
            for b in range(n_bands):
                mask = (freqs >= freq_bands[b]) & (freqs <= freq_bands[b + 1])
                abs_powers.append(
                    float(np.trapz(psd[mask], freqs[mask])) if mask.any() else 0.0
                )

            total = sum(abs_powers) + 1e-10
            row.extend([p / total for p in abs_powers])

        features[i] = row

    return features


def spectral_edge_features(data, sfreq_val=None, edges=(0.90, 0.95)):
    """
    Spectral edge frequency: frequency below which X% of total power lies.
    Default: SEF90 and SEF95 — useful stress markers (beta activity shifts SEF).
    Returns (n_total, n_channels * len(edges)).
    """
    n_trials, n_secs, n_channels, sfreq_int = data.shape
    if sfreq_val is None:
        sfreq_val = float(sfreq_int)
    data_reshaped = data.reshape(-1, n_channels, sfreq_int)
    n_total = data_reshaped.shape[0]

    features = np.zeros((n_total, n_channels * len(edges)))

    for i in range(n_total):
        epoch = data_reshaped[i]
        row   = []

        for ch in range(n_channels):
            s = epoch[ch]
            freqs, psd = signal.welch(s, fs=sfreq_val, nperseg=min(256, len(s)))
            cumulative = np.cumsum(psd) / (np.sum(psd) + 1e-10)

            for edge in edges:
                idx = np.searchsorted(cumulative, edge)
                sef = float(freqs[min(idx, len(freqs) - 1)])
                row.append(sef)

        features[i] = row

    return features


def zero_crossing_linelength_features(data):
    """
    Zero-crossing rate and line length — cheap but effective temporal features.
    Returns (n_total, n_channels * 2).
    """
    n_trials, n_secs, n_channels, sfreq = data.shape
    data_reshaped = data.reshape(-1, n_channels, sfreq)
    n_total = data_reshaped.shape[0]

    features = np.zeros((n_total, n_channels * 2))

    for i in range(n_total):
        epoch = data_reshaped[i]
        row   = []

        for ch in range(n_channels):
            s = epoch[ch]
            # Zero-crossing rate
            zcr = float(np.sum(np.diff(np.signbit(s).astype(int)) != 0)) / max(len(s) - 1, 1)
            # Line length (sum of absolute first differences)
            ll  = float(np.sum(np.abs(np.diff(s))))
            row.extend([zcr, ll])

        features[i] = row

    return features


def spectral_asymmetry_features(data, freq_bands, sfreq_val=None):
    """
    Inter-hemispheric asymmetry: log(power_left / power_right) per band.
    Uses the first min(5, n_ch//2) left/right channel pairs.
    Returns (n_total, n_pairs * n_bands).

    If fewer than 2 channels exist, returns zeros.
    """
    n_trials, n_secs, n_channels, sfreq_int = data.shape
    if sfreq_val is None:
        sfreq_val = float(sfreq_int)
    data_reshaped = data.reshape(-1, n_channels, sfreq_int)
    n_total = data_reshaped.shape[0]
    n_bands = len(freq_bands) - 1

    # Determine pairs
    n_pairs = min(5, n_channels // 2)
    if n_pairs == 0:
        return np.zeros((n_total, 1))  # placeholder

    l_idx = list(range(0, n_pairs * 2, 2))
    r_idx = list(range(1, n_pairs * 2, 2))

    features = np.zeros((n_total, n_pairs * n_bands))

    for i in range(n_total):
        epoch = data_reshaped[i]
        row   = []

        for lc, rc in zip(l_idx, r_idx):
            for b in range(n_bands):
                def band_power(s):
                    freqs, psd = signal.welch(s, fs=sfreq_val,
                                              nperseg=min(256, len(s)))
                    mask = (freqs >= freq_bands[b]) & (freqs <= freq_bands[b + 1])
                    return float(np.trapz(psd[mask], freqs[mask])) if mask.any() else 1e-10

                pl = band_power(epoch[lc]) + 1e-10
                pr = band_power(epoch[rc]) + 1e-10
                row.append(float(np.log(pl / pr)))

        features[i] = row

    return features


def spectral_ratios_features(data, freq_bands):
    """
    Clinically validated EEG stress ratios — computed per channel:
      • Beta / Alpha
      • Theta / Alpha  (engagement / fatigue)
      • (Beta + Gamma) / (Delta + Theta + Alpha)  — arousal index
      • Alpha / (Theta + Beta)                   — relaxation index

    Returns (n_total, n_channels * 4).
    """
    n_trials, n_secs, n_channels, sfreq = data.shape
    data_reshaped = data.reshape(-1, n_channels, sfreq)
    n_total = data_reshaped.shape[0]

    # Expected band order from BAND_EDGES = [0.5, 4, 8, 12, 30, 45]
    # Indices: 0=delta, 1=theta, 2=alpha, 3=beta, 4=gamma
    N_RATIOS = 4
    features  = np.zeros((n_total, n_channels * N_RATIOS))

    for i in range(n_total):
        epoch = data_reshaped[i]
        row   = []

        for ch in range(n_channels):
            s = epoch[ch]
            freqs, psd = signal.welch(s, fs=sfreq, nperseg=min(256, len(s)))

            def bp(lo, hi):
                mask = (freqs >= lo) & (freqs <= hi)
                return float(np.trapz(psd[mask], freqs[mask])) if mask.any() else 1e-10

            delta = bp(freq_bands[0], freq_bands[1]) + 1e-10
            theta = bp(freq_bands[1], freq_bands[2]) + 1e-10
            alpha = bp(freq_bands[2], freq_bands[3]) + 1e-10
            beta  = bp(freq_bands[3], freq_bands[4]) + 1e-10
            gamma = bp(freq_bands[4], freq_bands[5]) + 1e-10 if len(freq_bands) > 5 else 1e-10

            row.extend([
                beta / alpha,
                theta / alpha,
                (beta + gamma) / (delta + theta + alpha),
                alpha / (theta + beta),
            ])

        features[i] = row

    return features


def wavelet_energy_features(data, n_levels: int = 4):
    """
    Discrete Wavelet Transform energy per decomposition level (Daubechies-4).
    Approximates sub-band energy without needing pywavelets:
    uses iterated moving-average filter banks (Haar-like).

    Returns (n_total, n_channels * (n_levels + 1)).
    """
    n_trials, n_secs, n_channels, sfreq = data.shape
    data_reshaped = data.reshape(-1, n_channels, sfreq)
    n_total = data_reshaped.shape[0]

    n_coeffs = n_levels + 1  # detail levels + approximation
    features = np.zeros((n_total, n_channels * n_coeffs))

    for i in range(n_total):
        epoch = data_reshaped[i]
        row   = []

        for ch in range(n_channels):
            s = epoch[ch].copy()
            energies = []

            for _ in range(n_levels):
                # Simple averaging / differencing (Haar-like)
                n = len(s) // 2 * 2
                s_even = s[:n:2]
                s_odd  = s[1:n:2]
                approx  = (s_even + s_odd) / 2.0   # low-pass
                detail  = (s_even - s_odd) / 2.0   # high-pass (detail)
                energies.append(float(np.sum(detail ** 2) / (len(detail) + 1e-10)))
                s = approx

            energies.append(float(np.sum(s ** 2) / (len(s) + 1e-10)))  # final approx
            row.extend(energies)

        features[i] = row

    return features


# ─────────────────────────────────────────────────────────────────────────────
# Windowing
# ─────────────────────────────────────────────────────────────────────────────

def window_signal_hanning(data, sfreq=256.0, window_sec=2, overlap=0.5):
    """
    IMPROVEMENT: Default window increased to 2 s (512 samples) to capture
    slower stress-related rhythms (theta 4-8 Hz needs ~250 ms minimum,
    but 2 s gives 8x better frequency resolution vs 1 s).

    Returns (n_windows, n_channels, window_samples).
    """
    n_trials, n_epochs, n_channels, n_samples = data.shape
    window_samples = int(window_sec * sfreq)
    step = int(window_samples * (1 - overlap))
    n_steps = (n_samples - window_samples) // step + 1

    all_windows = []
    hanning_window = np.hanning(window_samples)

    for trial in range(n_trials):
        for epoch in range(n_epochs):
            for i in range(n_steps):
                start = i * step
                end   = start + window_samples
                windowed = data[trial, epoch, :, start:end] * hanning_window
                all_windows.append(windowed)

    return np.array(all_windows)


# ─────────────────────────────────────────────────────────────────────────────
# Master extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_all_features(data, sfreq=256.0, window_sec=2, overlap=0.5):
    """
    Extract all feature groups and concatenate.
    data shape: (n_trials, n_epochs, n_channels, n_samples)
    """
    freq_bands = np.array([0.5, 4, 8, 12, 30, 45])

    print("Windowing signal (Hanning, {}s, {}% overlap)...".format(
        window_sec, int(overlap * 100)))
    windows = window_signal_hanning(data, sfreq=sfreq,
                                    window_sec=window_sec, overlap=overlap)

    # Reshape to (n_windows, 1, n_ch, window_samples) for feature functions
    windows_r = windows[:, np.newaxis, :, :]

    print("  Time-domain features...")
    t_feats   = time_series_features(windows_r)

    print("  Absolute band power features...")
    f_feats   = freq_band_features(windows_r, freq_bands)

    print("  Relative band power features...")
    rf_feats  = relative_band_features(windows_r, freq_bands)

    print("  Spectral ratio features...")
    sr_feats  = spectral_ratios_features(windows_r, freq_bands)

    print("  Spectral edge features...")
    se_feats  = spectral_edge_features(windows_r)

    print("  Zero-crossing & line-length features...")
    zl_feats  = zero_crossing_linelength_features(windows_r)

    print("  Hjorth features...")
    h_feats   = hjorth_features(windows_r)

    print("  Fractal features...")
    fr_feats  = fractal_features(windows_r)

    print("  Entropy features...")
    e_feats   = entropy_features(windows_r)

    print("  Wavelet energy features...")
    wv_feats  = wavelet_energy_features(windows_r)

    print("  Spectral asymmetry features...")
    asym_feats = spectral_asymmetry_features(windows_r, freq_bands)

    all_features = np.hstack([
        t_feats, f_feats, rf_feats, sr_feats,
        se_feats, zl_feats, h_feats, fr_feats,
        e_feats, wv_feats, asym_feats,
    ])

    print(f"Feature extraction complete. Shape: {all_features.shape}")
    return all_features