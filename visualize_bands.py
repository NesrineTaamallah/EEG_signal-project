

import os
import numpy as np
import pandas as pd
import scipy.io
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize
import matplotlib.gridspec as gridspec
from scipy import stats
from features import freq_band_features, window_signal_hanning

CLEANED_DIR = r"C:\Users\nesri\Downloads\Data\Data\my_cleaned_data"
OUTPUT_DIR = r"C:\Users\nesri\Downloads\Data\Data\visualise_band"
SFREQ = 256.0
WINDOW_SEC = 1.0
OVERLAP = 0.5
MAX_CHANNELS_SPECTRO = 6
MAX_CHANNELS_TOPOPLOT = 32  
FIG_DPI = 300  
SAVE_FORMATS = ['png', 'pdf']  

BAND_EDGES = np.array([0.5, 4, 8, 12, 30, 45])
BAND_NAMES = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]
BAND_COLORS = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd', '#8c564b']  
BAND_COLORS_RGB = [(31/255, 119/255, 180/255),  
                   (44/255, 160/255, 44/255),   
                   (214/255, 39/255, 40/255),   
                   (148/255, 103/255, 189/255), 
                   (140/255, 86/255, 75/255)]   

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'figure.dpi': FIG_DPI,
    'savefig.dpi': FIG_DPI,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.5,
    'patch.linewidth': 0.8
})

os.makedirs(OUTPUT_DIR, exist_ok=True)

def robust_load_mat(path):
    """Chargement robuste des fichiers MAT"""
    mat = scipy.io.loadmat(path, squeeze_me=False, mat_dtype=True)
    return mat

def ensure_channel_axis(data):
    """Assure que les données sont en format (canaux, échantillons)"""
    if data.ndim != 2:
        raise ValueError("data_cleaned doit être 2D (n_canaux, n_échantillons)")
    n0, n1 = data.shape
    if n0 > 500 and n1 < 500:  
        return data.T
    return data

def infer_channel_names(mat, n_channels):
    """Infère les noms des canaux"""
    keys = ['cleaned_ch_names', 'ch_names', 'channel_names', 'channels', 'EEG_chans', 'labels']
    for k in keys:
        if k in mat:
            arr = mat[k]
            try:
                if isinstance(arr, np.ndarray):
                    if arr.dtype.names:
                        names = [str(arr[dtype][0]) for dtype in arr.dtype.names[:n_channels]]
                    elif arr.dtype == object:
                        names = [str(arr[i][0]) if isinstance(arr[i], np.ndarray) else str(arr[i]) 
                                for i in range(min(len(arr), n_channels))]
                    else:
                        names = [str(arr[i]) for i in range(min(len(arr), n_channels))]
                    
                    if len(names) >= n_channels:
                        return names[:n_channels]
            except Exception:
                pass
    return [f"Ch{i+1:02d}" for i in range(n_channels)]

def create_professional_cmaps():
    
    band_cmap = LinearSegmentedColormap.from_list(
        'band_power',
        ['#f7fbff', '#6baed6', '#08519c', '#08306b'],
        N=256
    )
    
    diff_cmap = LinearSegmentedColormap.from_list(
        'diff_power',
        ['#b2182b', '#f7f7f7', '#2166ac'],
        N=256
    )
    
    return band_cmap, diff_cmap

def calculate_band_statistics(bandpower_data):
    stats_dict = {}
    for i, band in enumerate(BAND_NAMES):
        band_data = bandpower_data[:, i]
        stats_dict[band] = {
            'mean': np.mean(band_data),
            'std': np.std(band_data),
            'median': np.median(band_data),
            'q1': np.percentile(band_data, 25),
            'q3': np.percentile(band_data, 75),
            'min': np.min(band_data),
            'max': np.max(band_data),
            'cv': np.std(band_data) / np.mean(band_data) if np.mean(band_data) > 0 else 0
        }
    return pd.DataFrame(stats_dict).T

def plot_bandpower_heatmap_pro(bandpower_mean, ch_names, out_base_path):
    
    fig = plt.figure(figsize=(12, max(6, len(ch_names) * 0.25)))
    gs = gridspec.GridSpec(1, 2, width_ratios=[0.9, 0.1], wspace=0.05)
    
    ax = plt.subplot(gs[0])
    band_cmap, _ = create_professional_cmaps()
    
    norm = LogNorm(vmin=np.min(bandpower_mean[bandpower_mean > 0]), 
                   vmax=np.max(bandpower_mean))
    
    im = ax.imshow(bandpower_mean, aspect='auto', cmap=band_cmap, norm=norm,
                   interpolation='nearest')
    
    ax.set_xticks(np.arange(len(BAND_NAMES)))
    ax.set_xticklabels(BAND_NAMES, rotation=45, ha='right')
    ax.set_yticks(np.arange(len(ch_names)))
    ax.set_yticklabels(ch_names)
    
    ax.set_xlabel('Frequency Bands (Hz)', fontweight='bold')
    ax.set_ylabel('Channels', fontweight='bold')
    ax.set_title(f'Band Power Distribution\n(Mean across windows)', 
                fontweight='bold', pad=20)
    
    if len(ch_names) <= 20:  
        for i in range(len(ch_names)):
            for j in range(len(BAND_NAMES)):
                text = ax.text(j, i, f'{bandpower_mean[i, j]:.2f}',
                             ha="center", va="center",
                             color="white" if bandpower_mean[i, j] > np.median(bandpower_mean) else "black",
                             fontsize=7)
    
    ax_cb = plt.subplot(gs[1])
    cb = plt.colorbar(im, cax=ax_cb)
    cb.set_label('Power (μV²/Hz)', fontweight='bold')
    
    plt.tight_layout()
    
    for fmt in SAVE_FORMATS:
        out_path = f"{out_base_path}_heatmap_pro.{fmt}"
        plt.savefig(out_path, dpi=FIG_DPI)
    plt.close()

def plot_bandpower_boxplot(bandpower_all_windows, ch_names, out_base_path, 
                          top_n_channels=10):
   
    n_windows, n_channels, n_bands = bandpower_all_windows.shape
    
    mean_power_per_channel = np.mean(bandpower_all_windows, axis=(0, 2))
    top_indices = np.argsort(mean_power_per_channel)[-top_n_channels:][::-1]
    top_ch_names = [ch_names[i] for i in top_indices]
    
    fig, axes = plt.subplots(1, top_n_channels, figsize=(max(12, top_n_channels*2), 6),
                           sharey=True)
    
    if top_n_channels == 1:
        axes = [axes]
    
    for idx, (ch_idx, ax) in enumerate(zip(top_indices, axes)):
        data_to_plot = []
        positions = []
        
        for band_idx in range(n_bands):
            band_data = bandpower_all_windows[:, ch_idx, band_idx]
            data_to_plot.append(band_data)
            positions.append(band_idx + 1)
        
        
        bp = ax.boxplot(data_to_plot, positions=positions, widths=0.6,
                       patch_artist=True,
                       medianprops=dict(color='black', linewidth=1.5),
                       whiskerprops=dict(color='gray', linewidth=1),
                       capprops=dict(color='gray', linewidth=1),
                       flierprops=dict(marker='o', markersize=3, alpha=0.5))
        
        for patch, color in zip(bp['boxes'], BAND_COLORS_RGB):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_title(top_ch_names[idx], fontsize=10, fontweight='bold')
        ax.set_xticks(positions)
        ax.set_xticklabels(BAND_NAMES, rotation=45, ha='right')
        
        if idx == 0:
            ax.set_ylabel('Power (μV²/Hz)', fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.suptitle(f'Band Power Distribution (Top {top_n_channels} Channels)', 
                fontweight='bold', y=1.02)
    plt.tight_layout()
    
    for fmt in SAVE_FORMATS:
        out_path = f"{out_base_path}_boxplot.{fmt}"
        plt.savefig(out_path, dpi=FIG_DPI)
    plt.close()

def plot_spectral_profile(bandpower_mean, ch_names, out_base_path, 
                         highlight_channels=None):
    
    fig, ax = plt.figure(figsize=(10, 6)), plt.gca()
    
    bandpower_norm = bandpower_mean / np.sum(bandpower_mean, axis=1, keepdims=True)
    
    x_pos = np.arange(len(BAND_NAMES))
    width = 0.8 / len(ch_names) if len(ch_names) <= 10 else 0.4
    
    for i, ch in enumerate(ch_names[:15]):  
        offset = (i - len(ch_names[:15])/2) * width
        bars = ax.bar(x_pos + offset, bandpower_norm[i], width,
                     label=ch, alpha=0.7,
                     color=BAND_COLORS_RGB,
                     edgecolor='black', linewidth=0.5)
        
        if highlight_channels and ch in highlight_channels:
            for bar in bars:
                bar.set_alpha(1.0)
                bar.set_linewidth(1.5)
                bar.set_edgecolor('red')
    
    ax.set_xlabel('Frequency Band', fontweight='bold')
    ax.set_ylabel('Normalized Power (%)', fontweight='bold')
    ax.set_title('Spectral Profile by Channel', fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'{BAND_NAMES[i]}\n({BAND_EDGES[i]}-{BAND_EDGES[i+1]} Hz)' 
                       for i in range(len(BAND_NAMES))])
    ax.legend(title='Channels', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    for fmt in SAVE_FORMATS:
        out_path = f"{out_base_path}_spectral_profile.{fmt}"
        plt.savefig(out_path, dpi=FIG_DPI)
    plt.close()

def plot_time_frequency_analysis(data, sfreq, ch_idx, ch_name, out_base_path):
    """
    Analyse temps-fréquence professionnelle pour un canal
    """
    from scipy.signal import spectrogram
    
    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(3, 2, height_ratios=[2, 1, 1], hspace=0.4, wspace=0.3)
    
    ax1 = plt.subplot(gs[0, :])
    nperseg = int(sfreq * 2)  
    f, t, Sxx = spectrogram(data[ch_idx], fs=sfreq, nperseg=nperseg, 
                           noverlap=nperseg//2, scaling='density')
    
    freq_mask = f <= 50
    f = f[freq_mask]
    Sxx = Sxx[freq_mask, :]
    
    pcm = ax1.pcolormesh(t, f, 10*np.log10(Sxx + 1e-10), 
                        shading='gouraud', cmap='viridis', 
                        rasterized=True)
    ax1.set_ylabel('Frequency (Hz)', fontweight='bold')
    ax1.set_xlabel('Time (s)', fontweight='bold')
    ax1.set_title(f'Time-Frequency Representation - {ch_name}', 
                 fontweight='bold', pad=10)
    cb1 = plt.colorbar(pcm, ax=ax1, pad=0.01)
    cb1.set_label('Power (dB)', fontweight='bold')
    
    ax2 = plt.subplot(gs[1, :])
    window_samples = int(sfreq)
    n_windows = len(data[ch_idx]) // window_samples
    
    band_power_time = np.zeros((n_windows, len(BAND_NAMES)))
    
    for i in range(n_windows):
        start = i * window_samples
        end = start + window_samples
        segment = data[ch_idx, start:end]
        
        from scipy.signal import welch
        freqs, psd = welch(segment, fs=sfreq, nperseg=window_samples)
        
        for band_idx in range(len(BAND_NAMES)):
            band_mask = (freqs >= BAND_EDGES[band_idx]) & (freqs < BAND_EDGES[band_idx + 1])
            band_power_time[i, band_idx] = np.trapz(psd[band_mask], freqs[band_mask])
    
    time_axis = np.arange(n_windows) * (window_samples/sfreq)
    for band_idx, band in enumerate(BAND_NAMES):
        ax2.plot(time_axis, band_power_time[:, band_idx], 
                label=band, color=BAND_COLORS_RGB[band_idx], linewidth=1.5)
    
    ax2.set_xlabel('Time (s)', fontweight='bold')
    ax2.set_ylabel('Band Power', fontweight='bold')
    ax2.set_title('Temporal Evolution of Band Power', fontweight='bold')
    ax2.legend(loc='upper right', ncol=3)
    ax2.grid(True, alpha=0.3)
    
    ax3 = plt.subplot(gs[2, 0])
    violin_data = [band_power_time[:, i] for i in range(len(BAND_NAMES))]
    vp = ax3.violinplot(violin_data, showmeans=True, showmedians=True)
    
    for i, pc in enumerate(vp['bodies']):
        pc.set_facecolor(BAND_COLORS_RGB[i])
        pc.set_alpha(0.7)
    
    ax3.set_xticks(np.arange(1, len(BAND_NAMES) + 1))
    ax3.set_xticklabels(BAND_NAMES, rotation=45)
    ax3.set_ylabel('Power Distribution', fontweight='bold')
    ax3.set_title('Statistical Distribution', fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    ax4 = plt.subplot(gs[2, 1])
    ratios = []
    ratio_labels = ['Theta/Alpha', 'Beta/Alpha', '(Beta+Gamma)/(Theta+Alpha)']
    
    theta_idx, alpha_idx, beta_idx, gamma_idx = 1, 2, 3, 4
    ratios.append(np.mean(band_power_time[:, theta_idx] / (band_power_time[:, alpha_idx] + 1e-10)))
    ratios.append(np.mean(band_power_time[:, beta_idx] / (band_power_time[:, alpha_idx] + 1e-10)))
    ratios.append(np.mean((band_power_time[:, beta_idx] + band_power_time[:, gamma_idx]) / 
                         (band_power_time[:, theta_idx] + band_power_time[:, alpha_idx] + 1e-10)))
    
    bars = ax4.bar(ratio_labels, ratios, color=['#2ca02c', '#9467bd', '#ff7f0e'])
    ax4.set_ylabel('Ratio Value', fontweight='bold')
    ax4.set_title('Spectral Ratios', fontweight='bold')
    ax4.tick_params(axis='x', rotation=45)
    
    for bar, val in zip(bars, ratios):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'Comprehensive Spectral Analysis - {ch_name}', 
                fontweight='bold', y=1.02)
    plt.tight_layout()
    
    for fmt in SAVE_FORMATS:
        out_path = f"{out_base_path}_{ch_name}_tfa.{fmt}"
        plt.savefig(out_path, dpi=FIG_DPI, bbox_inches='tight')
    plt.close()

def create_summary_report(bandpower_mean, bandpower_all_windows, ch_names, 
                         out_base_path, subject_id=""):
    
    import json
    
    
    band_stats = calculate_band_statistics(bandpower_mean)
    
    
    report_data = {
        'subject_id': subject_id,
        'n_channels': len(ch_names),
        'n_bands': len(BAND_NAMES),
        'band_edges': BAND_EDGES.tolist(),
        'band_names': BAND_NAMES,
        'statistics': band_stats.to_dict(),
        'top_channels': {}
    }
    
    for band_idx, band in enumerate(BAND_NAMES):
        top_idx = np.argsort(bandpower_mean[:, band_idx])[-3:][::-1]
        report_data['top_channels'][band] = {
            'channels': [ch_names[i] for i in top_idx],
            'values': [float(bandpower_mean[i, band_idx]) for i in top_idx]
        }
    
    report_path = f"{out_base_path}_report.json"
    with open(report_path, 'w') as f:
        json.dump(report_data, f, indent=2)
    
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 3, hspace=0.3, wspace=0.3)
    
    ax1 = plt.subplot(gs[0, 0])
    im = ax1.imshow(bandpower_mean, aspect='auto', cmap='viridis')
    ax1.set_title('Band Power Matrix', fontweight='bold')
    ax1.set_xticks(np.arange(len(BAND_NAMES)))
    ax1.set_xticklabels(BAND_NAMES, rotation=45)
    ax1.set_yticks(np.arange(len(ch_names)))
    ax1.set_yticklabels(ch_names)
    plt.colorbar(im, ax=ax1, shrink=0.8)
    
    ax2 = plt.subplot(gs[0, 1])
    global_profile = np.mean(bandpower_mean, axis=0)
    ax2.bar(BAND_NAMES, global_profile, color=BAND_COLORS)
    ax2.set_title('Global Spectral Profile', fontweight='bold')
    ax2.set_ylabel('Mean Power')
    ax2.tick_params(axis='x', rotation=45)
    
    ax3 = plt.subplot(gs[0, 2])
    if 10 <= len(ch_names) <= 32:
        
        total_power = np.sum(bandpower_mean, axis=1)
        norm_power = (total_power - np.min(total_power)) / (np.max(total_power) - np.min(total_power))
        
        
        angles = np.linspace(0, 2*np.pi, len(ch_names), endpoint=False)
        x = np.cos(angles)
        y = np.sin(angles)
        
        scatter = ax3.scatter(x, y, s=norm_power*500 + 50, c=total_power, 
                            cmap='hot', alpha=0.7, edgecolors='black')
        
        for i, (xi, yi, name) in enumerate(zip(x, y, ch_names)):
            ax3.text(xi*1.1, yi*1.1, name, ha='center', va='center', fontsize=8)
        
        ax3.set_aspect('equal')
        ax3.set_title('Channel Power Topography', fontweight='bold')
        ax3.axis('off')
        plt.colorbar(scatter, ax=ax3, shrink=0.8, label='Total Power')
    else:
        ax3.text(0.5, 0.5, 'Topography requires 10-32 channels', 
                ha='center', va='center', transform=ax3.transAxes)
        ax3.axis('off')
    
    ax4 = plt.subplot(gs[1, 0])
    stats_to_plot = ['mean', 'std', 'median']
    x = np.arange(len(BAND_NAMES))
    width = 0.25
    
    for i, stat in enumerate(stats_to_plot):
        values = [band_stats.loc[band, stat] for band in BAND_NAMES]
        ax4.bar(x + i*width - width, values, width, label=stat)
    
    ax4.set_xticks(x)
    ax4.set_xticklabels(BAND_NAMES, rotation=45)
    ax4.set_title('Band Statistics', fontweight='bold')
    ax4.legend()
    
    ax5 = plt.subplot(gs[1, 1])
    ratios = {
        'Beta/Alpha': np.mean(bandpower_mean[:, 3] / bandpower_mean[:, 2]),
        'Theta/Beta': np.mean(bandpower_mean[:, 1] / bandpower_mean[:, 3]),
        '(Alpha+Theta)/Total': np.mean((bandpower_mean[:, 2] + bandpower_mean[:, 1]) / 
                                      np.sum(bandpower_mean, axis=1))
    }
    
    ax5.bar(ratios.keys(), ratios.values(), color=['#9467bd', '#2ca02c', '#ff7f0e'])
    ax5.set_title('Spectral Ratios', fontweight='bold')
    ax5.tick_params(axis='x', rotation=45)
    
    ax6 = plt.subplot(gs[1, 2])
    for band_idx, band in enumerate(BAND_NAMES):
        sorted_data = np.sort(bandpower_mean[:, band_idx])
        cumsum = np.cumsum(sorted_data) / np.sum(sorted_data)
        ax6.plot(sorted_data, cumsum, label=band, color=BAND_COLORS_RGB[band_idx])
    
    ax6.set_xlabel('Power')
    ax6.set_ylabel('Cumulative Probability')
    ax6.set_title('Cumulative Distribution', fontweight='bold')
    ax6.legend(loc='lower right')
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle(f'EEG Spectral Analysis Summary - {subject_id}', 
                fontweight='bold', fontsize=16, y=1.02)
    plt.tight_layout()
    
    for fmt in SAVE_FORMATS:
        out_path = f"{out_base_path}_summary.{fmt}"
        plt.savefig(out_path, dpi=FIG_DPI, bbox_inches='tight')
    plt.close()
    
    print(f"   ✓ Rapport de synthèse généré: {out_base_path}_summary.png")
    print(f"   ✓ Données statistiques: {report_path}")

def main():
    print("=" * 70)
    print("ANALYSE SPECTRALE EEG PROFESSIONNELLE")
    print("=" * 70)
    
    processed_files = 0
    
    for fname in sorted(os.listdir(CLEANED_DIR)):
        if not fname.endswith(".mat"):
            continue
        
        file_path = os.path.join(CLEANED_DIR, fname)
        base_name = os.path.splitext(fname)[0]
        subject_id = base_name.split('_')[0] if '_' in base_name else base_name
        
        print(f"\n📊 Traitement: {fname}")
        print(f"   Subject ID: {subject_id}")
        print("-" * 50)
        
        try:
            mat = robust_load_mat(file_path)
            
            # Extraction des données
            if 'data_cleaned' in mat:
                data = np.array(mat['data_cleaned'], dtype=float)
                print(f"   ✓ Données trouvées: data_cleaned")
            else:
                data = None
                for k, v in mat.items():
                    if k.startswith('__'):
                        continue
                    if isinstance(v, np.ndarray) and v.ndim == 2:
                        if 1 < v.shape[0] < 256 and v.shape[1] > 1000:
                            data = v.astype(float)
                            print(f"   ✓ Données trouvées: {k} ({v.shape})")
                            break
                
                if data is None:
                    print(f"   ⚠ Aucune donnée 2D trouvée, fichier ignoré")
                    continue
            
            data = ensure_channel_axis(data)
            n_channels, n_samples = data.shape
            print(f"   • Canaux: {n_channels}")
            print(f"   • Échantillons: {n_samples}")
            print(f"   • Durée: {n_samples/SFREQ:.1f}s")
            
            ch_names = infer_channel_names(mat, n_channels)
            print(f"   • Canaux identifiés: {', '.join(ch_names[:5])}" + 
                 (f" ... ({n_channels} total)" if n_channels > 5 else ""))
            
            print(f"   • Découpage en fenêtres...")
            windows = window_signal_hanning(data[np.newaxis, np.newaxis, :, :], 
                                          sfreq=SFREQ,
                                          window_sec=WINDOW_SEC, 
                                          overlap=OVERLAP)
            n_windows = windows.shape[0]
            print(f"   • Fenêtres générées: {n_windows}")
            
            print(f"   • Calcul des bandes de fréquence...")
            windows_reshaped = windows[:, np.newaxis, :, :]
            freq_feats = freq_band_features(windows_reshaped, BAND_EDGES)
            
            freq_block = freq_feats.reshape(freq_feats.shape[0], n_channels, len(BAND_NAMES))
            bandpower_mean = np.mean(freq_block, axis=0)
            bandpower_std = np.std(freq_block, axis=0)
            
            subject_output_dir = os.path.join(OUTPUT_DIR, subject_id)
            os.makedirs(subject_output_dir, exist_ok=True)
            
            base_out_path = os.path.join(subject_output_dir, base_name)
            
            csv_path = f"{base_out_path}_bandpower_detailed.csv"
            detailed_data = []
            for ch_idx, ch_name in enumerate(ch_names):
                for band_idx, band_name in enumerate(BAND_NAMES):
                    detailed_data.append({
                        'channel': ch_name,
                        'band': band_name,
                        'mean_power': float(bandpower_mean[ch_idx, band_idx]),
                        'std_power': float(bandpower_std[ch_idx, band_idx]),
                        'freq_range': f"{BAND_EDGES[band_idx]}-{BAND_EDGES[band_idx+1]} Hz"
                    })
            
            pd.DataFrame(detailed_data).to_csv(csv_path, index=False)
            print(f"   ✓ CSV détaillé: {csv_path}")
            
            print(f"   • Génération des visualisations...")
            
            plot_bandpower_heatmap_pro(bandpower_mean, ch_names, base_out_path)
            print(f"   ✓ Heatmap professionnelle générée")
            
            if n_channels > 1:
                plot_bandpower_boxplot(freq_block, ch_names, base_out_path, 
                                     top_n_channels=min(8, n_channels))
                print(f"   ✓ Boxplot généré")
            
            plot_spectral_profile(bandpower_mean, ch_names, base_out_path)
            print(f"   ✓ Profil spectral généré")
            
            n_tfa_channels = min(3, n_channels)
            for ch_idx in range(n_tfa_channels):
                plot_time_frequency_analysis(data, SFREQ, ch_idx, ch_names[ch_idx], base_out_path)
            print(f"   ✓ Analyses temps-fréquence ({n_tfa_channels} canaux)")
            
            create_summary_report(bandpower_mean, freq_block, ch_names, base_out_path, subject_id)
            
            np.savez_compressed(
                f"{base_out_path}_bandpower_raw.npz",
                bandpower_mean=bandpower_mean,
                bandpower_all=freq_block,
                bandpower_std=bandpower_std,
                channel_names=np.array(ch_names),
                band_names=np.array(BAND_NAMES),
                band_edges=BAND_EDGES,
                subject_id=subject_id
            )
            
            processed_files += 1
            print(f"   ✅ Analyse terminée pour {subject_id}")
            
        except Exception as e:
            print(f"   ❌ Erreur: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("RAPPORT FINAL")
    print("=" * 70)
    print(f"Fichiers traités avec succès: {processed_files}")
    print(f"Répertoire de sortie: {OUTPUT_DIR}")
    print(f"Formats générés: {', '.join(SAVE_FORMATS)}")
    print("\nFichiers générés par sujet:")
    print("  • bandpower_detailed.csv - Données détaillées")
    print("  • *_heatmap_pro.* - Heatmap professionnelle")
    print("  • *_boxplot.* - Distribution statistique")
    print("  • *_spectral_profile.* - Profil spectral")
    print("  • *_tfa.* - Analyses temps-fréquence")
    print("  • *_summary.* - Rapport de synthèse")
    print("  • *_report.json - Données statistiques")
    print("  • *_bandpower_raw.npz - Données brutes compressées")
    print("\n✅ Analyse spectrale EEG complétée!")

if __name__ == "__main__":
    main()