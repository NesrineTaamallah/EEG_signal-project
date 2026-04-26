import os
import argparse
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
import mne

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


try:
    from asrpy import ASR
    ASR_AVAILABLE = True
except ImportError:
    ASR_AVAILABLE = False
    print("⚠ ASRpy non disponible. Utilisation d'ICA à la place.")
    print("   Installation: pip install asrpy")

def load_mat_eeg(filepath, scale_factor=1.0):
    """Charge un fichier .mat et détecte automatiquement les données EEG."""
    mat = loadmat(filepath)
    for k, v in mat.items():
        if isinstance(v, np.ndarray) and v.ndim == 2:
            if v.shape[0] < 256 and v.shape[1] > 10:
                data_scaled = v.copy() * scale_factor
                return mat, k, data_scaled
    raise ValueError("Aucune variable 2D EEG trouvée dans le .mat.")

def check_data_range(data, sfreq=256.0):
    """Vérifie la plage des données et donne des recommandations."""
    max_val = np.max(np.abs(data))
    mean_val = np.mean(np.abs(data))
    
    print(f"   • Amplitude max: {max_val:.6f} V ({max_val*1e6:.1f} µV)")
    print(f"   • Amplitude moyenne: {mean_val:.6f} V ({mean_val*1e6:.1f} µV)")
    
    if max_val > 1.0: 
        print("   ATTENTION: Valeurs anormalement élevées!")
        print("    Essayez avec --scale 1e-6 (si données en volts)")
        return False
    elif max_val < 1e-9:  
        print("     ATTENTION: Valeurs anormalement faibles!")
        print("    Essayez avec --scale 1e6 (si données en microvolts)")
        return False
    elif max_val > 0.01:  
        print("    Valeurs élevées détectées")
        print("   Vérifiez l'unité de vos données")
        return True  
    else:
        print("   ✓ Plage d'amplitude OK")
        return True

def create_raw_from_array(data, sfreq=256.0, ch_names=None):
    """Crée un objet RawArray MNE."""
    n_ch, n_samp = data.shape
    
    if ch_names is None:
        ch_names = [f"EEG{c+1}" for c in range(n_ch)]
    
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=['eeg']*n_ch)
    raw = mne.io.RawArray(data, info)
    
    return raw

def preprocess_pipeline(data, sfreq=256.0, use_asr=True):
    """Pipeline de prétraitement principal sans ICA."""
    raw = create_raw_from_array(data, sfreq=sfreq)
    
    print("1. Filtre notch (50Hz, 100Hz)...")
    raw.notch_filter([50, 100], picks='eeg', verbose=False)
    
    print("2. Filtre passe-bande (1-40Hz)...")
    raw.filter(1, 40, picks='eeg', fir_design='firwin', verbose=False)
    
    print("3. Re-référencement à la moyenne...")
    raw.set_eeg_reference('average', projection=False)
    
    if use_asr and ASR_AVAILABLE:
        print("4. ASR - Suppression des artefacts...")
        print("   [INFO] Fitting ASR sur les données...")
        asr = ASR(sfreq=raw.info["sfreq"], cutoff=15)  
        asr.fit(raw)
        raw = asr.transform(raw)
        print("   ✓ ASR appliqué avec succès")
    else:
        print("4. Suppression des artefacts: ASR désactivé ou non disponible, aucune autre méthode appliquée")
    
    return raw




def detect_and_interpolate_bad_channels(raw, threshold=5.0):
    """Détecte et interpole les canaux mauvais."""
    data = raw.get_data()
    n_channels = data.shape[0]
    
    stds = np.std(data, axis=1)
    mean_std = np.mean(stds)
    std_std = np.std(stds)
    
    bad_channels = []
    
    z_scores = (stds - mean_std) / (std_std + 1e-10)
    for i in range(n_channels):
        if abs(z_scores[i]) > threshold:
            bad_channels.append(raw.ch_names[i])
    
    flat_threshold = mean_std * 0.1  
    for i in range(n_channels):
        if stds[i] < flat_threshold:
            if raw.ch_names[i] not in bad_channels:
                bad_channels.append(raw.ch_names[i])
    
    if bad_channels:
        print(f"   • Canaux mauvais détectés: {bad_channels}")
        raw.info['bads'] = bad_channels
        
        try:
            raw.interpolate_bads(reset_bads=True)
            print("   ✓ Canaux mauvais interpolés")
        except Exception as e:
            print(f"     Impossible d'interpoler: {e}")
    else:
        print("   • Aucun canal mauvais détecté")
    
    return raw

def save_cleaned_mat(data_processed, output_path, metadata=None, scale_factor=1.0, scale_unit='V'):
    """
    Sauvegarde les données nettoyées dans un fichier .mat.
    - data_processed: array (canaux x échantillons)
    - metadata: dict (optionnel)
    - scale_factor: float, facteur appliqué aux données d'origine pour arriver à data_processed.
      Par exemple: si vous avez converti µV -> V avec *1e-6, passez scale_factor=1e-6.
    - scale_unit: 'V' ou 'µV' : unité dans laquelle les données sont stockées dans data_processed.
    """
    mat_dict = {
        'data_cleaned': data_processed,
        'cleaned_signal': data_processed,
        'eeg_cleaned': data_processed,
        'scale_factor': scale_factor,
        'scale_unit': scale_unit
    }

    if metadata:
        for key, value in metadata.items():
            mat_dict[f'cleaned_{key}'] = value

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    savemat(output_path, mat_dict, do_compression=True)

    print(f"   ✓ Fichier cleaned.mat sauvegardé: {output_path}")
    print(f"   • Shape: {data_processed.shape[0]} canaux × {data_processed.shape[1]} échantillons")
    print(f"   • Type de données: {data_processed.dtype}")
    print(f"   • Taille: {os.path.getsize(output_path) / 1024:.1f} KB")

    return output_path


def plot_all_channels(data_raw, data_processed, sfreq=256.0, 
                     n_channels=6, offset=100e-6):
    """Affiche plusieurs canaux côte à côte."""
    
    n_channels = min(n_channels, data_raw.shape[0])
    n_samples = min(data_raw.shape[1], data_processed.shape[1])
    
    display_samples = min(int(5 * sfreq), n_samples)
    times = np.arange(display_samples) / sfreq
    
    fig, axes = plt.subplots(2, 1, figsize=(18, 12), sharex=True)
    
    for i in range(n_channels):
        signal = data_raw[i, :display_samples]
        axes[0].plot(times, signal + i*offset, 
                    color='red', linewidth=1.0, alpha=0.8)
        axes[0].text(times[-1] + 0.15, i*offset, f'EEG{i+1}', 
                    verticalalignment='center', fontsize=10, 
                    color='darkred', fontweight='bold')
    
    axes[0].set_title(f'{n_channels} CANAUX - SIGNAL BRUT', 
                     fontsize=16, fontweight='bold', color='darkred')
    axes[0].set_ylabel('Amplitude (V)', fontsize=12)
    axes[0].grid(True, alpha=0.2, linestyle=':')
    axes[0].set_xlim([0, times[-1]])
    
    for i in range(n_channels):
        signal = data_processed[i, :display_samples]
        axes[1].plot(times, signal + i*offset, 
                    color='blue', linewidth=1.0, alpha=0.8)
        axes[1].text(times[-1] + 0.15, i*offset, f'EEG{i+1}', 
                    verticalalignment='center', fontsize=10,
                    color='darkblue', fontweight='bold')
    
    axes[1].set_title(f'{n_channels} CANAUX - SIGNAL PRÉTRAITÉ', 
                     fontsize=16, fontweight='bold', color='darkblue')
    axes[1].set_xlabel('Temps (s)', fontsize=12)
    axes[1].set_ylabel('Amplitude (V)', fontsize=12)
    axes[1].grid(True, alpha=0.2, linestyle=':')
    axes[1].set_xlim([0, times[-1]])
    
    plt.suptitle('COMPARAISON MULTI-CANAUX: BRUT vs PRÉTRAITÉ', 
                fontsize=18, fontweight='bold', y=1.02)
    
    info_text = (
        f"Configuration:\n"
        f"• Canaux affichés: {n_channels}/{data_raw.shape[0]}\n"
        f"• Durée affichée: {display_samples/sfreq:.1f}s\n"
        f"• Offset vertical: {offset*1e6:.0f} µV\n"
        f"• Fréquence: {sfreq} Hz"
    )
    fig.text(0.02, 0.98, info_text, fontsize=10, 
            verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
    
    plt.tight_layout()
    return fig

def plot_single_channel_detail(data_raw, data_processed, sfreq=256.0, 
                              channel_idx=0, time_range=None):
    """Affiche un seul canal en détail avec superposition."""
    
    n_samples = min(data_raw.shape[1], data_processed.shape[1])
    
    if time_range:
        start_idx = int(time_range[0] * sfreq)
        end_idx = int(time_range[1] * sfreq)
        start_idx = max(0, min(start_idx, n_samples-1))
        end_idx = max(start_idx+1, min(end_idx, n_samples-1))
        times = np.arange(start_idx, end_idx) / sfreq
        data_raw_sel = data_raw[channel_idx, start_idx:end_idx]
        data_proc_sel = data_processed[channel_idx, start_idx:end_idx]
    else:
        display_samples = min(int(5 * sfreq), n_samples)
        times = np.arange(display_samples) / sfreq
        data_raw_sel = data_raw[channel_idx, :display_samples]
        data_proc_sel = data_processed[channel_idx, :display_samples]
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    
    ax1.plot(times, data_raw_sel, 'r-', linewidth=1.5, alpha=0.7, label='Signal Brut')
    ax1.set_title(f'CANAL EEG{channel_idx+1} - SIGNAL BRUT', 
                 fontsize=14, fontweight='bold', color='darkred')
    ax1.set_ylabel('Amplitude (µV)', fontsize=12)
    ax1.legend(loc='upper right', fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(times, data_proc_sel, 'b-', linewidth=1.5, alpha=0.7, label='Signal Prétraité')
    ax2.set_title(f'CANAL EEG{channel_idx+1} - SIGNAL PRÉTRAITÉ', 
                 fontsize=14, fontweight='bold', color='darkblue')
    ax2.set_ylabel('Amplitude (µV)', fontsize=12)
    ax2.legend(loc='upper right', fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    ax3.plot(times, data_raw_sel, 'r-', linewidth=1.0, alpha=0.5, label='Brut')
    ax3.plot(times, data_proc_sel, 'b-', linewidth=1.0, alpha=0.5, label='Prétraité')
    ax3.set_title('SUPERPOSITION DES DEUX SIGNAUX', 
                 fontsize=14, fontweight='bold', color='purple')
    ax3.set_xlabel('Temps (s)', fontsize=12)
    ax3.set_ylabel('Amplitude (µV)', fontsize=12)
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    std_raw = np.std(data_raw_sel) * 1e6
    std_proc = np.std(data_proc_sel) * 1e6
    reduction = 100 * (1 - std_proc/std_raw) if std_raw > 0 else 0
    
    stats_text = (
        f"STATISTIQUES:\n"
        f"Signal Brut:\n"
        f"  • Écart-type: {std_raw:.1f} µV\n"
        f"  • Max: {np.max(np.abs(data_raw_sel))*1e6:.1f} µV\n\n"
        f"Signal Prétraité:\n"
        f"  • Écart-type: {std_proc:.1f} µV\n"
        f"  • Max: {np.max(np.abs(data_proc_sel))*1e6:.1f} µV\n\n"
        f"Réduction de bruit: {reduction:.1f}%"
    )
    
    fig.text(0.02, 0.02, stats_text, fontsize=11,
            verticalalignment='bottom',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
    
    plt.suptitle(f'ANALYSE DÉTAILLÉE - CANAL EEG{channel_idx+1}', 
                fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    return fig

def main():
    parser = argparse.ArgumentParser(
        description="Pipeline complet de prétraitement EEG - Retourne cleaned.mat",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  python preprocessor.py fichier.mat                    # Prétraiter et sauvegarder cleaned.mat
  python preprocessor.py fichier.mat --all-channels    # Tous les canaux
  python preprocessor.py fichier.mat --channel 5       # Canal spécifique
  python preprocessor.py fichier.mat --output cleaned_output/  # Spécifier le dossier de sortie
  python preprocessor.py fichier.mat --scale 1e-6      # Scaling pour données en volts
  python preprocessor.py fichier.mat --no-save-plots   # Ne pas sauvegarder les plots
        """
    )
    
    parser.add_argument("input_file", 
                       help="Fichier .mat contenant les données EEG")
    
    vis_group = parser.add_mutually_exclusive_group()
    vis_group.add_argument("--all-channels", action="store_true",
                          help="Afficher plusieurs canaux (défaut: 6)")
    vis_group.add_argument("--channel", type=int, default=0,
                          help="Canal spécifique à analyser (0-based, défaut: 0)")
    
    parser.add_argument("--n-channels", type=int, default=6,
                       help="Nombre de canaux à afficher avec --all-channels")
    
    parser.add_argument("--sfreq", type=float, default=256.0,
                       help="Fréquence d'échantillonnage (défaut: 256 Hz)")
    parser.add_argument("--offset", type=float, default=100e-6,
                       help="Offset vertical entre canaux (défaut: 100e-6)")
    parser.add_argument("--no-asr", action="store_true",
                       help="Désactiver ASR (utiliser ICA)")
    
    parser.add_argument("--scale", type=float, default=1.0,
                       help="Facteur de scaling pour les données (défaut: 1.0)")
    
    parser.add_argument("--output", type=str, default=None,
                       help="Dossier de sortie pour cleaned.mat (défaut: dossier du fichier d'entrée)")
    parser.add_argument("--no-save-plots", action="store_true",
                       help="Ne pas sauvegarder les graphiques")
    parser.add_argument("--show-plots", action="store_true",
                       help="Afficher les graphiques (défaut: sauvegarder seulement)")
    
    args = parser.parse_args()
    
    print("="*70)
    print("PIPELINE EEG COMPLET - PRÉTRAITEMENT ET SAUVEGARDE DE CLEANED.MAT")
    print("="*70)
    
    if args.scale != 1.0:
        print(f"\n Facteur de scaling appliqué: {args.scale}")
    
    if args.output is None:
        output_dir = os.path.dirname(os.path.abspath(args.input_file))
        if output_dir == "":
            output_dir = "."
    else:
        output_dir = args.output
    
    os.makedirs(output_dir, exist_ok=True)
    
    cleaned_mat_path = os.path.join(output_dir, "cleaned.mat")
    
    print(f"\n Chargement: {os.path.basename(args.input_file)}")
    try:
        mat_data, var_name, data_raw = load_mat_eeg(args.input_file, scale_factor=args.scale)
        print(f"   ✓ Variable: {var_name}")
        print(f"   ✓ Dimensions: {data_raw.shape[0]} canaux × {data_raw.shape[1]} échantillons")
        print(f"   ✓ Durée: {data_raw.shape[1]/args.sfreq:.2f} secondes")
        
        print(f"   ✓ Facteur de scaling: {args.scale}")
        check_data_range(data_raw, args.sfreq)
        
    except Exception as e:
        print(f"   ✗ Erreur de chargement: {e}")
        return
    
    print(f"\n Prétraitement en cours...")
    use_asr = not args.no_asr and ASR_AVAILABLE
    
    try:
        raw_processed = preprocess_pipeline(data_raw, sfreq=args.sfreq, use_asr=use_asr)
        
        print("5. Détection des canaux mauvais...")
        raw_processed = detect_and_interpolate_bad_channels(raw_processed)
        
        data_processed = raw_processed.get_data()
        print(f"   ✓ Prétraitement terminé")
    except Exception as e:
        print(f"   ✗ Erreur lors du prétraitement: {e}")
        return
    
    print(f"\n Sauvegarde des données nettoyées...")
    
    metadata = {
        'sfreq': args.sfreq,
        'ch_names': raw_processed.ch_names,
        'scale_factor': args.scale,
        'processing_date': np.datetime64('now').astype(str),
        'input_file': os.path.basename(args.input_file),
        'bad_channels': raw_processed.info['bads'] if raw_processed.info['bads'] else []
    }
    
    
    scale_factor_to_save = metadata.get('scale_factor', 1.0)
    
    save_cleaned_mat(data_processed, cleaned_mat_path, metadata,
                    scale_factor=scale_factor_to_save, scale_unit='V')
    
    if not args.no_save_plots or args.show_plots:
        print(f"\n Génération des visualisations...")
        
        figures_dir = os.path.join(output_dir, "figures")
        os.makedirs(figures_dir, exist_ok=True)
        
        if args.all_channels:
            print(f"   Mode: Multi-canaux ({args.n_channels} canaux)")
            fig = plot_all_channels(
                data_raw, data_processed,
                sfreq=args.sfreq,
                n_channels=args.n_channels,
                offset=args.offset
            )
            
            if not args.no_save_plots:
                save_path = os.path.join(figures_dir, "multi_channels_comparison.png")
                fig.savefig(save_path, bbox_inches='tight', dpi=150)
                print(f"   ✓ Figure sauvegardée: {save_path}")
        else:
            print(f"   Mode: Canal unique (EEG{args.channel+1})")
            fig = plot_single_channel_detail(
                data_raw, data_processed,
                sfreq=args.sfreq,
                channel_idx=args.channel
            )
            
            if not args.no_save_plots:
                save_path = os.path.join(figures_dir, f"channel_{args.channel+1}_detail.png")
                fig.savefig(save_path, bbox_inches='tight', dpi=150)
                print(f"   ✓ Figure sauvegardée: {save_path}")
        
        if args.show_plots:
            plt.show()
        else:
            plt.close('all')
    
    print(f"\n Sauvegarde des formats supplémentaires...")
    
    try:
        fif_path = os.path.join(output_dir, "cleaned_eeg.fif")
        raw_processed.save(fif_path, overwrite=True)
        print(f"   ✓ Fichier .fif: {fif_path}")
    except Exception as e:
        print(f"   Erreur .fif: {e}")
    
    try:
        npz_path = os.path.join(output_dir, "cleaned_eeg.npz")
        np.savez_compressed(
            npz_path,
            data_cleaned=data_processed,
            sfreq=args.sfreq,
            ch_names=raw_processed.ch_names,
            scale_factor=args.scale,
            bad_channels=raw_processed.info['bads'] if raw_processed.info['bads'] else []
        )
        print(f"   ✓ Fichier .npz: {npz_path}")
    except Exception as e:
        print(f"    Erreur .npz: {e}")
    
    print("\n" + "="*70)
    print(" RAPPORT D'ANALYSE")
    print("="*70)
    
    print(f"\nANALYSE DES {min(5, data_raw.shape[0])} PREMIERS CANAUX:")
    print("-"*50)
    
    for i in range(min(5, data_raw.shape[0])):
        std_raw = np.std(data_raw[i]) * 1e6  
        std_proc = np.std(data_processed[i]) * 1e6
        reduction = 100 * (1 - std_proc/std_raw) if std_raw > 0 else 0
        
        if reduction > 30:
            rating = "Excellent"
        elif reduction > 20:
            rating = "Très bon"
        elif reduction > 10:
            rating = "Bon"
        elif reduction > 5:
            rating = "Modéré"
        elif reduction > 0:
            rating = "Léger"
        elif reduction == 0:
            rating = "Nul"
        else:
            rating = " Négatif"
        
        print(f"EEG{i+1:2d}: {std_raw:6.1f} µV → {std_proc:6.1f} µV "
              f"| Réduction: {reduction:5.1f}% | {rating}")
    
    print("\n" + "="*70)
    print(" TRAITEMENT TERMINÉ AVEC SUCCÈS!")
    print("="*70)
    
    print(f"\n FICHIER CLEANED.MAT:")
    print(f"   Chemin: {cleaned_mat_path}")
    print(f"   Taille: {os.path.getsize(cleaned_mat_path) / 1024:.1f} KB")
    print(f"   Canaux: {data_processed.shape[0]}")
    print(f"   Échantillons: {data_processed.shape[1]}")
    print(f"   Durée: {data_processed.shape[1]/args.sfreq:.2f} secondes")
    
    print("\n Contenu du fichier cleaned.mat:")
    print("   • data_cleaned: Données EEG nettoyées (canaux × échantillons)")
    print("   • cleaned_signal: Alias pour data_cleaned")
    print("   • eeg_cleaned: Alias pour data_cleaned")
    print("   • cleaned_sfreq: Fréquence d'échantillonnage")
    print("   • cleaned_ch_names: Noms des canaux")
    
    print("\n Pour utiliser avec features_visualize.py:")
    print(f"   python features_visualize.py --input {cleaned_mat_path}")
    print(f"   python features_visualize.py --input {cleaned_mat_path} --image-size 128 --max-images 12")
    
    if not args.show_plots:
        plt.close('all')

if __name__ == "__main__":
    main()