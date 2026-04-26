import numpy as np
from scipy import signal, stats
import warnings

warnings.filterwarnings('ignore')

SFREQ = 256.0  
def time_series_features(data):
   
    n_trials, n_secs, n_channels, sfreq = data.shape
    
    data_reshaped = data.reshape(-1, n_channels, sfreq)
    n_total = data_reshaped.shape[0]
    
    features = np.zeros((n_total, n_channels * 5)) 
    
    for i in range(n_total):
        epoch = data_reshaped[i]
        epoch_features = []
        
        for ch in range(n_channels):
            signal_data = epoch[ch]
            
            variance = np.var(signal_data)
            rms = np.sqrt(np.mean(signal_data ** 2))
            ptp_amp = np.ptp(signal_data)
            
            skew_val = stats.skew(signal_data) if len(signal_data) > 1 else 0
            kurt_val = stats.kurtosis(signal_data) if len(signal_data) > 1 else 0
            
            epoch_features.extend([variance, rms, ptp_amp, skew_val, kurt_val])
        
        features[i] = epoch_features
    
    return features


def freq_band_features(data, freq_bands):
    
    n_trials, n_secs, n_channels, sfreq = data.shape
    data_reshaped = data.reshape(-1, n_channels, sfreq)
    n_total = data_reshaped.shape[0]
    
    n_bands = len(freq_bands) - 1
    features = np.zeros((n_total, n_channels * n_bands))
    
    for i in range(n_total):
        epoch = data_reshaped[i]
        epoch_features = []
        
        for ch in range(n_channels):
            signal_data = epoch[ch]
            
            freqs, psd = signal.welch(signal_data, fs=sfreq, nperseg=min(256, len(signal_data)))
            
            band_powers = []
            for band_idx in range(n_bands):
                low_freq = freq_bands[band_idx]
                high_freq = freq_bands[band_idx + 1]
                
                idx = np.logical_and(freqs >= low_freq, freqs <= high_freq)
                if np.any(idx):
                    band_power = np.trapz(psd[idx], freqs[idx])
                    band_powers.append(band_power)
                else:
                    band_powers.append(0.0)
            
            epoch_features.extend(band_powers)
        
        features[i] = epoch_features
    
    return features


def hjorth_features(data):
    
    n_trials, n_secs, n_channels, sfreq = data.shape
    data_reshaped = data.reshape(-1, n_channels, sfreq)
    n_total = data_reshaped.shape[0]
    
    features = np.zeros((n_total, n_channels * 3))  
    
    for i in range(n_total):
        epoch = data_reshaped[i]
        epoch_features = []
        
        for ch in range(n_channels):
            signal_data = epoch[ch]
            
            activity = np.var(signal_data)
            
            if len(signal_data) > 1:
                mobility = np.sqrt(np.var(np.diff(signal_data)) / (activity + 1e-10))
                complexity = np.sqrt(np.var(np.diff(np.diff(signal_data))) / 
                                   (np.var(np.diff(signal_data)) + 1e-10)) / (mobility + 1e-10)
            else:
                mobility = complexity = 0
            
            epoch_features.extend([activity, mobility, complexity])
        
        features[i] = epoch_features
    
    return features


def fractal_features(data):
    
    n_trials, n_secs, n_channels, sfreq = data.shape
    data_reshaped = data.reshape(-1, n_channels, sfreq)
    n_total = data_reshaped.shape[0]
    
    features = np.zeros((n_total, n_channels * 2)) 
    
    for i in range(n_total):
        epoch = data_reshaped[i]
        epoch_features = []
        
        for ch in range(n_channels):
            signal_data = epoch[ch]
            
            N = len(signal_data)
            k_max = 10
            L = []
            
            for k in range(1, k_max + 1):
                Lk = 0
                for m in range(k):
                    indices = np.arange(m, N, k, dtype=int)
                    if len(indices) > 1:
                        Lkm = np.sum(np.abs(np.diff(signal_data[indices])))
                        Lk += Lkm * (N - 1) / (len(indices) * k)
                if k > 0:
                    L.append(np.log(Lk / k))
            
            if len(L) > 1:
                x = np.log(1 / np.arange(1, k_max + 1)[:len(L)])
                higuchi_fd = -np.polyfit(x, L, 1)[0]
            else:
                higuchi_fd = 0
            
            if N <= 1:
                katz_fd = 0
            else:
                L = np.sum(np.sqrt(1 + np.diff(signal_data) ** 2))
                d = np.max(np.sqrt((np.arange(N) / (N - 1)) ** 2 + 
                                  ((signal_data - signal_data[0]) / (np.max(np.abs(signal_data)) + 1e-10)) ** 2))
                if d > 0:
                    katz_fd = np.log(N - 1) / (np.log(d) + np.log((N - 1) / (L + 1e-10)))
                else:
                    katz_fd = 0
            
            epoch_features.extend([higuchi_fd, katz_fd])
        
        features[i] = epoch_features
    
    return features


def entropy_features(data):
    
    n_trials, n_secs, n_channels, sfreq = data.shape
    data_reshaped = data.reshape(-1, n_channels, sfreq)
    n_total = data_reshaped.shape[0]
    
    features = np.zeros((n_total, n_channels * 4))  
    
    for i in range(n_total):
        epoch = data_reshaped[i]
        epoch_features = []
        
        for ch in range(n_channels):
            signal_data = epoch[ch]
            N = len(signal_data)
            
            # Entropie approchée
            def _phi(m):
                if N <= m:
                    return 0
                
                patterns = np.lib.stride_tricks.sliding_window_view(signal_data, m)
                C = np.sum(np.max(np.abs(patterns[:, None] - patterns[None, :]), axis=2) <= 0.2*np.std(signal_data), axis=1)
                C = C / (N - m + 1)
                return np.sum(np.log(C + 1e-10)) / (N - m + 1)
            
            app_entropy = max(0, _phi(2) - _phi(3))
            
            if N > 3:
                samp_entropy = -np.log(np.corrcoef(signal_data[:-1], signal_data[1:])[0, 1] + 1e-10)
            else:
                samp_entropy = 0
            
            freqs, psd = signal.welch(signal_data, fs=sfreq)
            psd_norm = psd / (np.sum(psd) + 1e-10)
            spect_entropy = -np.sum(psd_norm * np.log(psd_norm + 1e-10))
            
            if N >= 10:
                tau = 1
                m = 3
                n_vectors = N - (m - 1) * tau
                if n_vectors > 0:
                    delayed_matrix = np.zeros((n_vectors, m))
                    for j in range(m):
                        delayed_matrix[:, j] = signal_data[j * tau:j * tau + n_vectors]
                    
                    try:
                        _, s, _ = np.linalg.svd(delayed_matrix, full_matrices=False)
                        s_norm = s / (np.sum(s) + 1e-10)
                        svd_entropy = -np.sum(s_norm * np.log(s_norm + 1e-10))
                    except:
                        svd_entropy = 0
                else:
                    svd_entropy = 0
            else:
                svd_entropy = 0
            
            epoch_features.extend([app_entropy, samp_entropy, spect_entropy, svd_entropy])
        
        features[i] = epoch_features
    
    return features


def extract_all_features(data, sfreq=256.0, window_sec=1, overlap=0.5):
    
    print("Découpage en fenêtres Hanning...")
    windows = window_signal_hanning(data, sfreq=sfreq, window_sec=window_sec, overlap=overlap)
    
    windows_reshaped = windows[:, np.newaxis, :, :]  
    
    print("Extraction des features temporelles...")
    time_feats = time_series_features(windows_reshaped)
    
    print("Extraction des features fréquentielles...")
    freq_bands = np.array([0.5, 4, 8, 12, 30, 45])
    freq_feats = freq_band_features(windows_reshaped, freq_bands)
    
    print("Extraction des features de Hjorth...")
    hjorth_feats = hjorth_features(windows_reshaped)
    
    print("Extraction des features fractales...")
    fractal_feats = fractal_features(windows_reshaped)
    
    print("Extraction des features d'entropie...")
    entropy_feats = entropy_features(windows_reshaped)
    
    all_features = np.hstack([time_feats, freq_feats, hjorth_feats, fractal_feats, entropy_feats])
    print(f"✓ Extraction terminée. Shape: {all_features.shape}")
    
    return all_features



def window_signal_hanning(data, sfreq=256.0, window_sec=1, overlap=0.5):
    
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
                end = start + window_samples
                windowed = data[trial, epoch, :, start:end] * hanning_window
                all_windows.append(windowed)
    
    return np.array(all_windows)  