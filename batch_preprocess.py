import os
from scipy.io import loadmat, savemat
from preprocessor import (
    load_mat_eeg,
    preprocess_pipeline,
    detect_and_interpolate_bad_channels,
    plot_all_channels,
    plot_single_channel_detail
)
import numpy as np
import matplotlib.pyplot as plt

RAW_DIR = r'C:\Users\nesri\OneDrive\Desktop\signal\data\Data\raw_data'
CLEANED_DIR = r'C:\Users\nesri\OneDrive\Desktop\signal\data\Data\cleaned_data'

SFREQ = 256.0
SCALE = 1e-6   
USE_ASR = True

FIGURES_ROOT = r'C:\Users\nesri\OneDrive\Desktop\signal\data\Data\figures_preprocessing'
MAX_CHANNELS_TO_PLOT = 8     
MULTI_CHANNEL_DISPLAY = 6    
OFFSET = 100e-6              

os.makedirs(CLEANED_DIR, exist_ok=True)
os.makedirs(FIGURES_ROOT, exist_ok=True)

def find_2d_array_in_mat(mat):
    """Renvoie (name, array) pour la première variable 2D plausible (heuristique)."""
    for k, v in mat.items():
        if isinstance(v, (list, tuple)):
            continue
        try:
            import numpy as _np
            if isinstance(v, _np.ndarray) and v.ndim == 2:
                cand = v.astype(float)
                if cand.shape[0] > cand.shape[1] and cand.shape[0] > 256:
                    cand = cand.T
                if cand.shape[0] < 256 and cand.shape[1] > 10:
                    return k, cand
        except Exception:
            continue
    return None, None

for filename in os.listdir(RAW_DIR):
    if not filename.endswith(".mat"):
        continue

    raw_path = os.path.join(RAW_DIR, filename)
    out_path = os.path.join(CLEANED_DIR, f"cleaned_{filename}")

    print(f"\nTraitement: {filename}")

    try:
        raw_mat = loadmat(raw_path)
        var_name, data_raw_orig = find_2d_array_in_mat(raw_mat)
        if data_raw_orig is None:
            raise ValueError("Impossible de détecter les données brutes dans le .mat")

        print(f"   ✓ Variable détectée (raw): {var_name}")
        print(f"   ✓ Dimensions (raw): {data_raw_orig.shape[0]} canaux × {data_raw_orig.shape[1]} échantillons")

        if SCALE != 1.0:
            data_for_processing = data_raw_orig * SCALE
            scale_unit = 'V'
        else:
            data_for_processing = data_raw_orig.copy()
            scale_unit = 'µV'

        raw_processed = preprocess_pipeline(data_for_processing, sfreq=SFREQ, use_asr=USE_ASR)

        raw_processed = detect_and_interpolate_bad_channels(raw_processed)

        data_processed = raw_processed.get_data()  

        metadata = {
            'sfreq': SFREQ,
            'ch_names': raw_processed.ch_names,
            'input_file': filename,
            'processing_date': np.datetime64('now').astype(str),
            'bad_channels': raw_processed.info.get('bads', [])
        }

        mat_dict = {
            'data_raw_original': data_raw_orig,     
            'data_cleaned': data_processed,         
            'scale_factor': SCALE,
            'scale_unit': scale_unit,
            'cleaned_sfreq': SFREQ,
            'cleaned_ch_names': raw_processed.ch_names,
            'cleaned_input_file': filename,
            'cleaned_processing_date': metadata['processing_date'],
            'cleaned_bad_channels': metadata['bad_channels']
        }

        savemat(out_path, mat_dict, do_compression=True)
        print(f"   ✓ Fichier sauvegardé: {out_path}")
        print(f"   • Shape cleaned: {data_processed.shape[0]} × {data_processed.shape[1]} (type: {data_processed.dtype})")

        
        try:
            filename_no_ext = os.path.splitext(filename)[0]
            figures_dir = os.path.join(FIGURES_ROOT, filename_no_ext)
            os.makedirs(figures_dir, exist_ok=True)

            print(f"   ▶ Génération des visualisations dans: {figures_dir}")

            
            try:
                fig_multi = plot_all_channels(
                    data_for_processing, data_processed,
                    sfreq=SFREQ,
                    n_channels=min(MULTI_CHANNEL_DISPLAY, data_for_processing.shape[0]),
                    offset=OFFSET
                )
                multi_path = os.path.join(figures_dir, f"multi_channels_comparison.png")
                fig_multi.savefig(multi_path, bbox_inches='tight', dpi=150)
                plt.close(fig_multi)
                print(f"      ✓ Multi-canaux sauvegardé: {multi_path}")
            except Exception as e:
                print(f"      ⚠ Échec génération multi-canaux: {e}")

            n_ch_total = data_for_processing.shape[0]
            n_to_plot = min(n_ch_total, MAX_CHANNELS_TO_PLOT)
            for ch_idx in range(n_to_plot):
                try:
                    fig_ch = plot_single_channel_detail(
                        data_for_processing, data_processed,
                        sfreq=SFREQ,
                        channel_idx=ch_idx,
                        time_range=None  
                    )
                    ch_path = os.path.join(figures_dir, f"channel_{ch_idx+1:02d}_detail.png")
                    fig_ch.savefig(ch_path, bbox_inches='tight', dpi=150)
                    plt.close(fig_ch)
                    print(f"      ✓ Canal {ch_idx+1} sauvegardé: {ch_path}")
                except Exception as e:
                    print(f"      Échec génération canal {ch_idx+1}: {e}")

            
            if n_ch_total > MAX_CHANNELS_TO_PLOT:
                print(f"      • {n_ch_total} canaux présents — seuls les {MAX_CHANNELS_TO_PLOT} premiers ont été sauvegardés individuellement.")
        except Exception as e:
            print(f"   Erreur lors de la génération des visualisations: {e}")

    except Exception as e:
        print(f"Erreur lors du traitement de {filename}: {e}")

print("\nTraitement terminé pour tous les fichiers.")
