"""
features.py — EEG Feature Extraction (PRODUCTION VERSION)

ARCHITECTURE:
  All feature functions accept shape: (n_windows, 1, n_channels, win_samples)
  where n_windows = total windowed segments across all trials/epochs.

  Master function extract_all_features() orchestrates windowing + extraction.

FEATURE GROUPS (per channel unless noted):
  1. time          — 5 features: variance, rms, ptp, skewness, kurtosis
  2. abs_band      — 5 features: absolute PSD per band (Welch)
  3. rel_band      — 5 features: relative (normalised) PSD per band
  4. ratios        — 4 features: beta/alpha, theta/alpha, arousal, relaxation
  5. sef           — 2 features: spectral edge freq 90%, 95%
  6. zcll           — 2 features: zero-crossing rate, line length
  7. hjorth         — 3 features: activity, mobility, complexity
  8. fractal        — 2 features: Higuchi FD, Katz FD
  9. entropy        — 4 features: approx, sample, spectral, SVD
  10. wavelet        — 5 features: Haar-like energy per level
  11. asymmetry      — n_pairs * n_bands features (inter-hemispheric log-ratio)

Total per channel (groups 1-10): 5+5+5+4+2+2+3+2+4+5 = 37
Asymmetry block: min(5, n_ch//2) * 5 bands  (or 1 placeholder)

COLUMN NAMING (matches dataset.py and backend/main.py):
  "ch{i+1}_{group}_{k}"  for groups 1-10
  "asym_pair{p+1}_band{b}" for asymmetry block
"""

import numpy as np
from scipy import signal, stats
import warnings

warnings.filterwarnings('ignore')

SFREQ: float = 256.0
BAND_EDGES = np.array([0.5, 4.0, 8.0, 12.0, 30.0, 45.0])
BAND_NAMES = ["delta", "theta", "alpha", "beta", "gamma"]
N_BANDS = len(BAND_EDGES) - 1  # 5

# Feature counts per channel for each group
FEAT_COUNTS = {
    "time":    5,
    "absband": N_BANDS,      # 5
    "relband": N_BANDS,      # 5
    "ratios":  4,
    "sef":     2,
    "zcll":    2,
    "hjorth":  3,
    "fractal": 2,
    "entropy": 4,
    "wavelet": 5,
}
FEAT_PER_CH = sum(FEAT_COUNTS.values())  # 37


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _welch(s: np.ndarray, fs: float) -> tuple:
    """Welch PSD with safe nperseg."""
    nperseg = min(256, len(s))
    return signal.welch(s, fs=fs, nperseg=nperseg)


def _band_power(psd: np.ndarray, freqs: np.ndarray, lo: float, hi: float) -> float:
    mask = (freqs >= lo) & (freqs <= hi)
    return float(np.trapz(psd[mask], freqs[mask])) if mask.any() else 0.0


def _reshape(data: np.ndarray):
    """(n_trials, n_secs, n_ch, sfreq) → (n_windows, n_ch, sfreq)."""
    n_trials, n_secs, n_ch, sfreq = data.shape
    return data.reshape(-1, n_ch, sfreq)


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction functions  (public API)
# ─────────────────────────────────────────────────────────────────────────────

def time_series_features(data: np.ndarray) -> np.ndarray:
    """5 features / channel: variance, rms, ptp, skewness, kurtosis."""
    windows = _reshape(data)
    n_win, n_ch, _ = windows.shape
    out = np.zeros((n_win, n_ch * 5))
    for i, seg in enumerate(windows):
        row = []
        for ch in range(n_ch):
            s = seg[ch]
            row += [float(np.var(s)),
                    float(np.sqrt(np.mean(s**2))),
                    float(np.ptp(s)),
                    float(stats.skew(s)) if len(s) > 1 else 0.0,
                    float(stats.kurtosis(s)) if len(s) > 1 else 0.0]
        out[i] = row
    return out


def freq_band_features(data: np.ndarray, freq_bands: np.ndarray) -> np.ndarray:
    """Absolute band power (Welch). Shape → (n_win, n_ch * n_bands)."""
    windows = _reshape(data)
    n_win, n_ch, sfreq = windows.shape
    n_bands = len(freq_bands) - 1
    out = np.zeros((n_win, n_ch * n_bands))
    for i, seg in enumerate(windows):
        row = []
        for ch in range(n_ch):
            freqs, psd = _welch(seg[ch], sfreq)
            for b in range(n_bands):
                row.append(_band_power(psd, freqs, freq_bands[b], freq_bands[b+1]))
        out[i] = row
    return out


def relative_band_features(data: np.ndarray, freq_bands: np.ndarray) -> np.ndarray:
    """Relative (normalised) band power. Shape → (n_win, n_ch * n_bands)."""
    windows = _reshape(data)
    n_win, n_ch, sfreq = windows.shape
    n_bands = len(freq_bands) - 1
    out = np.zeros((n_win, n_ch * n_bands))
    for i, seg in enumerate(windows):
        row = []
        for ch in range(n_ch):
            freqs, psd = _welch(seg[ch], sfreq)
            abs_bp = [_band_power(psd, freqs, freq_bands[b], freq_bands[b+1])
                      for b in range(n_bands)]
            total = sum(abs_bp) + 1e-10
            row += [p / total for p in abs_bp]
        out[i] = row
    return out


def spectral_ratios_features(data: np.ndarray, freq_bands: np.ndarray) -> np.ndarray:
    """4 clinical stress ratios per channel."""
    windows = _reshape(data)
    n_win, n_ch, sfreq = windows.shape
    out = np.zeros((n_win, n_ch * 4))
    for i, seg in enumerate(windows):
        row = []
        for ch in range(n_ch):
            freqs, psd = _welch(seg[ch], sfreq)
            delta = _band_power(psd, freqs, freq_bands[0], freq_bands[1]) + 1e-10
            theta = _band_power(psd, freqs, freq_bands[1], freq_bands[2]) + 1e-10
            alpha = _band_power(psd, freqs, freq_bands[2], freq_bands[3]) + 1e-10
            beta  = _band_power(psd, freqs, freq_bands[3], freq_bands[4]) + 1e-10
            gamma = (_band_power(psd, freqs, freq_bands[4], freq_bands[5]) + 1e-10
                     if len(freq_bands) > 5 else 1e-10)
            row += [beta/alpha,
                    theta/alpha,
                    (beta+gamma)/(delta+theta+alpha),
                    alpha/(theta+beta)]
        out[i] = row
    return out


def spectral_edge_features(data: np.ndarray, edges=(0.90, 0.95)) -> np.ndarray:
    """SEF90 & SEF95 per channel. Shape → (n_win, n_ch * 2)."""
    windows = _reshape(data)
    n_win, n_ch, sfreq = windows.shape
    out = np.zeros((n_win, n_ch * len(edges)))
    for i, seg in enumerate(windows):
        row = []
        for ch in range(n_ch):
            freqs, psd = _welch(seg[ch], float(sfreq))
            cumulative = np.cumsum(psd) / (np.sum(psd) + 1e-10)
            for edge in edges:
                idx = np.searchsorted(cumulative, edge)
                row.append(float(freqs[min(idx, len(freqs)-1)]))
        out[i] = row
    return out


def zero_crossing_linelength_features(data: np.ndarray) -> np.ndarray:
    """ZCR & line length per channel. Shape → (n_win, n_ch * 2)."""
    windows = _reshape(data)
    n_win, n_ch, _ = windows.shape
    out = np.zeros((n_win, n_ch * 2))
    for i, seg in enumerate(windows):
        row = []
        for ch in range(n_ch):
            s = seg[ch]
            zcr = float(np.sum(np.diff(np.signbit(s).astype(int)) != 0)) / max(len(s)-1, 1)
            ll  = float(np.sum(np.abs(np.diff(s))))
            row += [zcr, ll]
        out[i] = row
    return out


def hjorth_features(data: np.ndarray) -> np.ndarray:
    """Activity, mobility, complexity per channel."""
    windows = _reshape(data)
    n_win, n_ch, _ = windows.shape
    out = np.zeros((n_win, n_ch * 3))
    for i, seg in enumerate(windows):
        row = []
        for ch in range(n_ch):
            s = seg[ch]
            act = float(np.var(s))
            if len(s) > 1:
                d1  = np.diff(s)
                mob = float(np.sqrt(np.var(d1) / (act + 1e-10)))
                d2  = np.diff(d1)
                cmp = float(np.sqrt(np.var(d2) / (np.var(d1) + 1e-10)) / (mob + 1e-10))
            else:
                mob = cmp = 0.0
            row += [act, mob, cmp]
        out[i] = row
    return out


def fractal_features(data: np.ndarray) -> np.ndarray:
    """Higuchi FD & Katz FD per channel."""
    windows = _reshape(data)
    n_win, n_ch, _ = windows.shape
    out = np.zeros((n_win, n_ch * 2))
    for i, seg in enumerate(windows):
        row = []
        for ch in range(n_ch):
            s = seg[ch]
            N = len(s)
            k_max = 10
            # Higuchi
            L_vals = []
            for k in range(1, k_max+1):
                Lk = 0
                for m in range(k):
                    idx = np.arange(m, N, k, dtype=int)
                    if len(idx) > 1:
                        Lk += np.sum(np.abs(np.diff(s[idx]))) * (N-1) / (len(idx)*k)
                L_vals.append(np.log(Lk/k + 1e-10))
            x_log = np.log(1/np.arange(1, k_max+1))
            higuchi = float(-np.polyfit(x_log, L_vals, 1)[0]) if len(L_vals) > 1 else 0.0
            # Katz
            if N <= 1:
                katz = 0.0
            else:
                L_sum = float(np.sum(np.sqrt(1 + np.diff(s)**2)))
                d_val = float(np.max(np.sqrt(
                    (np.arange(N)/(N-1))**2 +
                    ((s-s[0])/(np.max(np.abs(s))+1e-10))**2)))
                katz = (float(np.log(N-1) / (np.log(d_val) + np.log((N-1)/(L_sum+1e-10))))
                        if d_val > 0 else 0.0)
            row += [higuchi, katz]
        out[i] = row
    return out


def entropy_features(data: np.ndarray) -> np.ndarray:
    """4 entropy measures per channel."""
    windows = _reshape(data)
    n_win, n_ch, sfreq = windows.shape
    out = np.zeros((n_win, n_ch * 4))
    for i, seg in enumerate(windows):
        row = []
        for ch in range(n_ch):
            s = seg[ch]
            N = len(s)
            # Approx entropy
            def _phi(m_val):
                if N <= m_val:
                    return 0.0
                patterns = np.lib.stride_tricks.sliding_window_view(s, m_val)
                r = 0.2 * np.std(s)
                C = np.sum(np.max(np.abs(patterns[:,None] - patterns[None,:]), axis=2) <= r, axis=1)
                C = C / (N - m_val + 1)
                return float(np.sum(np.log(C + 1e-10)) / (N - m_val + 1))
            app_ent = max(0.0, _phi(2) - _phi(3))
            # Sample entropy
            if N > 3:
                corr = np.corrcoef(s[:-1], s[1:])[0, 1]
                samp_ent = float(-np.log(abs(corr) + 1e-10))
            else:
                samp_ent = 0.0
            # Spectral entropy
            freqs_e, psd_e = _welch(s, float(sfreq))
            psd_norm = psd_e / (np.sum(psd_e) + 1e-10)
            spec_ent = float(-np.sum(psd_norm * np.log(psd_norm + 1e-10)))
            # SVD entropy
            if N >= 10:
                tau, m_svd = 1, 3
                n_vec = N - (m_svd-1)*tau
                delayed = np.zeros((n_vec, m_svd))
                for j in range(m_svd):
                    delayed[:, j] = s[j*tau:j*tau+n_vec]
                try:
                    _, sv, _ = np.linalg.svd(delayed, full_matrices=False)
                    sv_norm = sv / (np.sum(sv) + 1e-10)
                    svd_ent = float(-np.sum(sv_norm * np.log(sv_norm + 1e-10)))
                except Exception:
                    svd_ent = 0.0
            else:
                svd_ent = 0.0
            row += [app_ent, samp_ent, spec_ent, svd_ent]
        out[i] = row
    return out


def wavelet_energy_features(data: np.ndarray, n_levels: int = 4) -> np.ndarray:
    """Haar-like wavelet energy per level. Shape → (n_win, n_ch * (n_levels+1))."""
    windows = _reshape(data)
    n_win, n_ch, _ = windows.shape
    n_coeffs = n_levels + 1
    out = np.zeros((n_win, n_ch * n_coeffs))
    for i, seg in enumerate(windows):
        row = []
        for ch in range(n_ch):
            s = seg[ch].copy()
            energies = []
            for _ in range(n_levels):
                n = len(s) // 2 * 2
                approx = (s[:n:2] + s[1:n:2]) / 2.0
                detail = (s[:n:2] - s[1:n:2]) / 2.0
                energies.append(float(np.sum(detail**2) / (len(detail)+1e-10)))
                s = approx
            energies.append(float(np.sum(s**2) / (len(s)+1e-10)))
            row += energies
        out[i] = row
    return out


def spectral_asymmetry_features(data: np.ndarray, freq_bands: np.ndarray) -> np.ndarray:
    """
    Inter-hemispheric log-ratio asymmetry.
    Pairs: (ch0,ch1), (ch2,ch3), ... up to 5 pairs.
    Shape → (n_win, n_pairs * n_bands)  or  (n_win, 1) placeholder.
    """
    windows = _reshape(data)
    n_win, n_ch, sfreq = windows.shape
    n_bands = len(freq_bands) - 1
    n_pairs = min(5, n_ch // 2)

    if n_pairs == 0:
        return np.zeros((n_win, 1))

    out = np.zeros((n_win, n_pairs * n_bands))
    for i, seg in enumerate(windows):
        row = []
        for p in range(n_pairs):
            lc, rc = p*2, p*2+1
            freqs_l, psd_l = _welch(seg[lc], float(sfreq))
            freqs_r, psd_r = _welch(seg[rc], float(sfreq))
            for b in range(n_bands):
                pl = _band_power(psd_l, freqs_l, freq_bands[b], freq_bands[b+1]) + 1e-10
                pr = _band_power(psd_r, freqs_r, freq_bands[b], freq_bands[b+1]) + 1e-10
                row.append(float(np.log(pl/pr)))
        out[i] = row
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Windowing
# ─────────────────────────────────────────────────────────────────────────────

def window_signal_hanning(data: np.ndarray,
                          sfreq: float = 256.0,
                          window_sec: float = 1.0,
                          overlap: float = 0.5) -> np.ndarray:
    """
    Slice (n_trials, n_epochs, n_ch, n_samples) into
    (n_windows, n_ch, win_samples) with Hanning taper.
    """
    n_trials, n_epochs, n_ch, n_samples = data.shape
    win_len = int(window_sec * sfreq)
    step    = int(win_len * (1 - overlap))
    n_steps = (n_samples - win_len) // step + 1
    hann    = np.hanning(win_len)

    all_windows = []
    for trial in range(n_trials):
        for epoch in range(n_epochs):
            for k in range(n_steps):
                s = k * step
                all_windows.append(data[trial, epoch, :, s:s+win_len] * hann)

    return np.array(all_windows)   # (n_windows, n_ch, win_len)


# ─────────────────────────────────────────────────────────────────────────────
# Column name builder  — SINGLE SOURCE OF TRUTH
# ─────────────────────────────────────────────────────────────────────────────

def build_column_names(n_channels: int, freq_bands: np.ndarray = BAND_EDGES) -> list:
    """
    Return the ordered list of column names that extract_all_features() produces.
    Used identically by:
      • dataset.py   (training DataFrame)
      • backend/main.py  (inference)
    """
    n_bands = len(freq_bands) - 1
    group_sizes = {
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

    cols = []
    for group, count in group_sizes.items():
        for ch in range(n_channels):
            for k in range(count):
                cols.append(f"ch{ch+1}_{group}_{k}")

    # Asymmetry block
    n_pairs = min(5, n_channels // 2)
    if n_pairs > 0:
        for p in range(n_pairs):
            for b in range(n_bands):
                cols.append(f"asym_pair{p+1}_band{b}")
    else:
        cols.append("asym_placeholder")

    return cols


# ─────────────────────────────────────────────────────────────────────────────
# Master extraction function
# ─────────────────────────────────────────────────────────────────────────────

def extract_all_features(data: np.ndarray,
                         sfreq: float = 256.0,
                         window_sec: float = 1.0,
                         overlap: float = 0.5) -> np.ndarray:
    """
    Full pipeline: window → extract → hstack.

    data shape: (n_trials, n_epochs, n_channels, n_samples)
    Returns:    (n_windows, n_features)

    Column order matches build_column_names(n_channels).
    """
    freq_bands = BAND_EDGES

    print(f"[features] Windowing ({window_sec}s, {int(overlap*100)}% overlap)…")
    windows = window_signal_hanning(data, sfreq=sfreq,
                                    window_sec=window_sec, overlap=overlap)
    # → (n_windows, n_ch, win_len)
    # Wrap as (n_windows, 1, n_ch, win_len) for all feature functions
    wr = windows[:, np.newaxis, :, :]

    print("[features] time…")
    t  = time_series_features(wr)
    print("[features] abs band power…")
    fb = freq_band_features(wr, freq_bands)
    print("[features] rel band power…")
    rb = relative_band_features(wr, freq_bands)
    print("[features] spectral ratios…")
    sr = spectral_ratios_features(wr, freq_bands)
    print("[features] SEF…")
    se = spectral_edge_features(wr)
    print("[features] ZCR + line-length…")
    zl = zero_crossing_linelength_features(wr)
    print("[features] Hjorth…")
    hj = hjorth_features(wr)
    print("[features] fractal…")
    fr = fractal_features(wr)
    print("[features] entropy…")
    en = entropy_features(wr)
    print("[features] wavelet…")
    wv = wavelet_energy_features(wr)
    print("[features] asymmetry…")
    ay = spectral_asymmetry_features(wr, freq_bands)

    result = np.hstack([t, fb, rb, sr, se, zl, hj, fr, en, wv, ay])
    print(f"[features] Done — shape: {result.shape}")
    return result