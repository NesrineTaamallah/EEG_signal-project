import os
import re
import numpy as np
import pandas as pd
import scipy.io

def load_labels_excel(labels_path):
    
    df = pd.read_excel(labels_path, header=[0,1])
    first_col = df.columns[0]
    df = df.set_index(first_col)
    try:
        df.index = df.index.astype(int)
    except Exception:
        pass
    return df

def parse_filename_for_meta(filename):
    
    name = filename.lower()
    patterns = [
        r"sub(?:ject)?[_\-\. ]*0*([0-9]{1,3}).*?(maths|symmetry|stroop).*?trial[_\-\. ]*([123])",
        r"^0*([0-9]{1,3})[_\-\. ].*?(maths|symmetry|stroop).*?trial[_\-\. ]*([123])",
        r"(maths|symmetry|stroop).*?[_\-\. ]*trial[_\-\. ]*([123]).*?0*([0-9]{1,3})",
        r"([0-9]{1,3})[_\-]?trial[_\-]?([123])[_\-]?(maths|symmetry|stroop)"
    ]
    for p in patterns:
        m = re.search(p, name, flags=re.I)
        if m:
            groups = m.groups()
            nums = [g for g in groups if g and g.isdigit()]
            texts = [g for g in groups if g and not g.isdigit()]
            try:
                subject = int(nums[0]) if nums else None
                if len(groups) == 3:
                    if groups[0].isdigit():
                        if groups[1].isalpha():
                            subject = int(groups[0]); test = groups[1].title(); trial = int(groups[2])
                        else:
                            subject = int(groups[0]); test = groups[2].title(); trial = int(groups[1])
                    else:
                        
                        test = [g for g in groups if g and not g.isdigit()][0].title()
                        trial = int([g for g in groups if g and g.isdigit()][-1])
                else:
                    test = texts[0].title() if texts else None
                    trial = int(nums[-1]) if nums else None
                return subject, test, trial
            except Exception:
                continue
    return None, None, None

def get_meta_from_mat(mat_path):
    try:
        mat = scipy.io.loadmat(mat_path)
        for k in mat.keys():
            if isinstance(k, str) and 'cleaned_subject' in k:
                return int(mat[k].squeeze())
            if k == 'cleaned_input_file':
                s = mat[k].squeeze()
                if isinstance(s, str):
                    return s
    except Exception:
        return None

def get_label_for_file(labels_df, subject, test_type, trial):
    
    if subject is None or test_type is None or trial is None:
        raise ValueError("subject/test_type/trial requis pour récupérer le label.")
    col = (f"Trial_{trial}", test_type)
    if col not in labels_df.columns:
        raise KeyError(f"Colonne {col} introuvable dans scales.xls. Colonnes disponibles: {labels_df.columns.tolist()}")
    val = labels_df.loc[subject, col]
    return bool(val > 5)

def build_dataset_with_labels(cleaned_dir, labels_path, sfreq=256.0):
    
    labels_df = load_labels_excel(labels_path)
    file_list = sorted([f for f in os.listdir(cleaned_dir) if f.endswith('.mat')])
    all_epoched = []
    labels_per_epoch = []
    filenames = []

    import dataset as ds  
    for f in file_list:
        path = os.path.join(cleaned_dir, f)
        try:
            mat = scipy.io.loadmat(path)
            data = mat.get('data_cleaned', None)
            if data is None:
                for alt in ['eeg_cleaned', 'cleaned_signal']:
                    if alt in mat:
                        data = mat[alt]
                        break
            if data is None:
                print(f"⚠ {f}: data_cleaned introuvable, fichier ignoré.")
                continue

            if data.shape[0] < data.shape[1] and data.shape[0] <= 128:
                n_channels, n_samples = data.shape
            else:
                if data.shape[1] <= 128:
                    data = data.T
                    n_channels, n_samples = data.shape
                else:
                    n_channels, n_samples = data.shape

            n_epochs = n_samples // sfreq
            if n_epochs == 0:
                print(f"⚠ {f}: durée < 1s (sfreq={sfreq}), ignoré.")
                continue

            subject, test_type, trial = parse_filename_for_meta(f)
            if subject is None:
                meta = mat.get('cleaned_subject', None) or mat.get('cleaned_input_file', None)
                if meta is not None:
                    if isinstance(meta, np.ndarray):
                        try:
                            meta = meta.squeeze()
                            if isinstance(meta, bytes):
                                meta = meta.decode('utf-8')
                        except Exception:
                            meta = None
                    if isinstance(meta, str):
                        s,t,tr = parse_filename_for_meta(meta)
                        if s is not None:
                            subject, test_type, trial = s,t,tr

            if subject is None or test_type is None or trial is None:
                raise ValueError(f"Impossible d'extraire metadata pour {f}. Normalise noms ou ajoute metadata au .mat")

            label_bool = get_label_for_file(labels_df, subject, test_type, trial)

            epochs = np.zeros((n_epochs, n_channels, sfreq))
            for e in range(n_epochs):
                start = e * sfreq
                end = start + sfreq
                epochs[e] = data[:, start:end]

            all_epoched.append(epochs[np.newaxis, ...]) 

            labels_per_epoch.append(np.full((n_epochs,), label_bool, dtype=bool))

            filenames.append(f)
            print(f"✓ {f}: subject={subject}, test={test_type}, trial={trial}, epochs={n_epochs}, label={label_bool}")

        except Exception as e:
            print(f"Erreur {f}: {e}")
            continue

    if len(all_epoched) == 0:
        raise RuntimeError("Aucun fichier valide chargé.")

    epoched_array = np.concatenate(all_epoched, axis=0)  
    labels_all = np.concatenate(labels_per_epoch, axis=0)  

    return epoched_array, labels_all, filenames
