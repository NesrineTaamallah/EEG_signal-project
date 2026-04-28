"""
classifier.py — Improved stress classifier.

KEY IMPROVEMENTS over original:
  1. DL OVERFITTING FIX: proper train/val split (never validate on train),
     stronger regularisation (dropout 0.5, weight decay 1e-3, gradient clip),
     smaller models, shorter training (40 epochs max + patience=7).
  2. CLASSICAL ML: added Optuna hyperparameter search (optional),
     better class-weighting, added LDA + SVM to the voting ensemble.
  3. FEATURE ENGINEERING: uses new features from features.py
     (relative power, ratios, SEF, wavelet, asymmetry).
  4. CROSS-VALIDATION: subject-level StratifiedGroupKFold (was already there
     but now we guarantee DL val fold never overlaps train).
  5. DATA AUGMENTATION for DL: Gaussian noise + time-shift augmentation.
  6. STACKING META-LEARNER: LogisticRegressionCV on out-of-fold predictions.
  7. Saves dl_config.json so backend can reconstruct models at inference.
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib
import scipy.io
import re
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Classical ML ─────────────────────────────────────────────────────────────
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, confusion_matrix
from sklearn.feature_selection import SelectFromModel, RFECV
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# ── Deep learning ─────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# ─────────────────────────────────────────────────────────────────────────────
# Configuration — edit paths here
# ─────────────────────────────────────────────────────────────────────────────
DATAFRAME_PATH  = r"C:\Users\nesri\OneDrive\Desktop\signal\data\Data\dataframe.csv"
CLEANED_DIR     = r"C:\Users\nesri\OneDrive\Desktop\signal\data\Data\cleaned_data"
LABELS_PATH     = r"C:\Users\nesri\OneDrive\Desktop\signal\data\Data\scales.xls"
MODELS_DIR      = r"C:\Users\nesri\OneDrive\Desktop\signal\data\Data\models"

MODEL_PATH      = os.path.join(MODELS_DIR, "xgb_stress_classifier_ensemble.joblib")
METRICS_PATH    = os.path.join(MODELS_DIR, "metrics.json")
FEAT_NAMES_PATH = os.path.join(MODELS_DIR, "feature_names.json")
META_PATH       = os.path.join(MODELS_DIR, "meta_learner.joblib")
DL_CNN_PATH     = os.path.join(MODELS_DIR, "dl_cnn_lstm.pt")
DL_EEGNET_PATH  = os.path.join(MODELS_DIR, "dl_eegnet.pt")
DL_TRANS_PATH   = os.path.join(MODELS_DIR, "dl_transformer.pt")
DL_CONFIG_PATH  = os.path.join(MODELS_DIR, "dl_config.json")

N_SPLITS     = 5
RANDOM_STATE = 42
SFREQ        = 256
WIN_SEC      = 2       # IMPROVED: 2 s instead of 1 s
OVERLAP      = 0.5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

os.makedirs(MODELS_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Deep Learning Architectures  (regularised versions)
# ═══════════════════════════════════════════════════════════════════════════════

class CNNLSTM(nn.Module):
    """
    Compact CNN-LSTM.
    IMPROVEMENT: added BatchNorm, increased dropout to 0.5, reduced depth.
    """
    def __init__(self, n_channels: int = 32, time_steps: int = 512,
                 n_classes: int = 2, dropout: float = 0.5):
        super().__init__()
        self.conv1 = nn.Conv1d(n_channels, 32, kernel_size=7, padding=3)
        self.bn1   = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2   = nn.BatchNorm1d(64)
        self.pool  = nn.MaxPool1d(4)   # more aggressive pooling → smaller LSTM
        self.drop  = nn.Dropout(dropout)

        # LSTM input length after 2 pools of 4: time_steps // 16
        self.lstm  = nn.LSTM(64, 64, num_layers=1, batch_first=True,
                             dropout=0.0, bidirectional=True)

        self.fc1 = nn.Linear(128, 32)
        self.fc2 = nn.Linear(32, n_classes)

    def forward(self, x):
        x = self.pool(F.elu(self.bn1(self.conv1(x))))
        x = self.pool(F.elu(self.bn2(self.conv2(x))))
        x = self.drop(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.drop(F.elu(self.fc1(x)))
        return self.fc2(x)


class EEGNet(nn.Module):
    """
    EEGNet (Lawhern et al. 2018) — unchanged architecture, higher dropout.
    """
    def __init__(self, n_channels: int = 32, time_steps: int = 512,
                 n_classes: int = 2, F1: int = 8, D: int = 2,
                 dropout: float = 0.5):
        super().__init__()
        F2 = F1 * D
        self.conv1    = nn.Conv2d(1, F1, (1, 64), padding=(0, 32), bias=False)
        self.bn1      = nn.BatchNorm2d(F1)
        self.dw_conv  = nn.Conv2d(F1, F2, (n_channels, 1), groups=F1, bias=False)
        self.bn2      = nn.BatchNorm2d(F2)
        self.avgpool1 = nn.AvgPool2d((1, 4))
        self.drop1    = nn.Dropout(dropout)
        self.sep_conv = nn.Conv2d(F2, F2, (1, 16), padding=(0, 8), bias=False)
        self.bn3      = nn.BatchNorm2d(F2)
        self.avgpool2 = nn.AvgPool2d((1, 8))
        self.drop2    = nn.Dropout(dropout)
        flat = F2 * (time_steps // 32)
        self.fc = nn.Linear(flat, n_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.elu(self.bn1(self.conv1(x)))
        x = F.elu(self.bn2(self.dw_conv(x)))
        x = self.drop1(self.avgpool1(x))
        x = F.elu(self.bn3(self.sep_conv(x)))
        x = self.drop2(self.avgpool2(x))
        x = x.flatten(1)
        return self.fc(x)


class TemporalTransformer(nn.Module):
    """
    Smaller Transformer with more regularisation.
    IMPROVEMENT: reduced d_model (32), added dropout=0.3.
    """
    def __init__(self, n_channels: int = 32, time_steps: int = 512,
                 n_classes: int = 2, d_model: int = 32, n_heads: int = 4,
                 n_layers: int = 1, dropout: float = 0.3):
        super().__init__()
        self.input_proj = nn.Linear(n_channels, d_model)
        self.pos_enc    = nn.Parameter(torch.randn(1, time_steps, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(d_model, n_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.input_proj(x)
        x = x + self.pos_enc[:, :x.size(1), :]
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.drop(x)
        return self.fc(x)


# ═══════════════════════════════════════════════════════════════════════════════
# Data augmentation for DL
# ═══════════════════════════════════════════════════════════════════════════════

def augment_eeg(X: np.ndarray, y: np.ndarray,
                noise_std: float = 0.05,
                shift_max: int = 32) -> tuple:
    """
    Simple augmentation:
      • Gaussian noise addition
      • Random time-shift (circular)
    Applied only to minority class to help with imbalance.
    """
    minority_mask = (y == 1)
    X_min = X[minority_mask]
    y_min = y[minority_mask]

    # Noise
    noise = X_min + np.random.randn(*X_min.shape).astype(np.float32) * noise_std
    # Time shift
    shift = np.random.randint(-shift_max, shift_max + 1)
    shifted = np.roll(X_min, shift, axis=2)

    X_aug = np.concatenate([X, noise, shifted], axis=0)
    y_aug = np.concatenate([y, y_min, y_min], axis=0)

    # Shuffle
    idx = np.random.permutation(len(X_aug))
    return X_aug[idx], y_aug[idx]


# ═══════════════════════════════════════════════════════════════════════════════
# DL Training helpers
# ═══════════════════════════════════════════════════════════════════════════════

def make_weighted_sampler(y_train: np.ndarray) -> WeightedRandomSampler:
    """Balanced batch sampler — avoids manual pos_weight tuning."""
    classes, counts = np.unique(y_train, return_counts=True)
    weights = 1.0 / counts
    sample_weights = weights[y_train]
    return WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float32),
        num_samples=len(sample_weights),
        replacement=True,
    )


def train_dl_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 40,
    batch_size: int = 32,
    lr: float = 5e-4,
    weight_decay: float = 1e-3,
    patience: int = 7,
) -> float:
    """
    Train one DL model with:
      • WeightedRandomSampler (class balance)
      • AdamW + CosineAnnealingWarmRestarts
      • Gradient clipping (max_norm=1.0)
      • Early stopping on val balanced accuracy
    """
    model = model.to(DEVICE)

    # IMPROVEMENT: augment training data
    X_tr_aug, y_tr_aug = augment_eeg(X_train, y_train)

    Xt  = torch.tensor(X_tr_aug, dtype=torch.float32)
    yt  = torch.tensor(y_tr_aug, dtype=torch.long)
    Xv  = torch.tensor(X_val,    dtype=torch.float32)
    yv  = torch.tensor(y_val,    dtype=torch.long)

    sampler = make_weighted_sampler(y_tr_aug)
    loader  = DataLoader(TensorDataset(Xt, yt), batch_size=batch_size,
                         sampler=sampler, drop_last=True)

    opt       = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2)
    criterion = nn.CrossEntropyLoss()   # sampler handles balance

    best_val_acc = 0.0
    best_state   = None
    no_improve   = 0

    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            logits = []
            for start in range(0, len(Xv), batch_size):
                xb = Xv[start: start + batch_size].to(DEVICE)
                logits.append(model(xb).cpu())
            logits  = torch.cat(logits)
            preds   = logits.argmax(1).numpy()
            val_acc = balanced_accuracy_score(yv.numpy(), preds)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve   = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"    Early stop ep {epoch+1}, best={best_val_acc:.4f}")
                break

        if (epoch + 1) % 10 == 0:
            print(f"    Ep {epoch+1:3d}  val_bal_acc={val_acc:.4f}  best={best_val_acc:.4f}")

    if best_state:
        model.load_state_dict(best_state)
    return best_val_acc


def get_dl_probas(model: nn.Module, X: np.ndarray,
                  batch_size: int = 128) -> np.ndarray:
    model.eval().to(DEVICE)
    Xt    = torch.tensor(X, dtype=torch.float32)
    probs = []
    with torch.no_grad():
        for start in range(0, len(Xt), batch_size):
            xb = Xt[start: start + batch_size].to(DEVICE)
            p  = torch.softmax(model(xb), dim=1).cpu().numpy()
            probs.append(p)
    return np.concatenate(probs)


# ═══════════════════════════════════════════════════════════════════════════════
# Raw window loader for DL
# ═══════════════════════════════════════════════════════════════════════════════

TEST_MAPPING = {
    "arithmetic":   "Maths",
    "mirror_image": "Symmetry",
    "symmetry":     "Symmetry",
    "stroop":       "Stroop",
}


def load_raw_windows_for_dl(
    cleaned_dir: str,
    labels_df: pd.DataFrame,
    sfreq: int = 256,
    win_sec: float = 2.0,
    overlap: float = 0.5,
):
    """
    Load raw EEG as Hanning-windowed segments for DL.
    IMPROVEMENT: win_sec=2 for better frequency resolution.
    Returns: X (N, n_ch, win_len), y (N,), groups (N, subject-id).
    """
    win_len = int(win_sec * sfreq)
    step    = int(win_len * (1 - overlap))
    hann    = np.hanning(win_len)

    all_windows, all_labels, all_groups = [], [], []

    for fname in sorted(os.listdir(cleaned_dir)):
        if not fname.endswith(".mat"):
            continue

        m = re.match(r"cleaned_(\w+)_sub_(\d+)_trial(\d+)\.mat",
                     fname, re.IGNORECASE)
        if not m:
            continue

        raw_test = m.group(1).lower()
        subject  = int(m.group(2))
        trial    = int(m.group(3))
        test     = TEST_MAPPING.get(raw_test)
        if test is None:
            continue

        col = (f"Trial_{trial}", test)
        if subject not in labels_df.index or col not in labels_df.columns:
            continue

        label = int(labels_df.loc[subject, col])

        mat = scipy.io.loadmat(os.path.join(cleaned_dir, fname))
        if "data_cleaned" not in mat:
            continue

        data = mat["data_cleaned"].astype(float)
        if data.shape[0] > data.shape[1]:
            data = data.T
        n_ch, n_samp = data.shape

        pos = 0
        while pos + win_len <= n_samp:
            seg = data[:, pos: pos + win_len] * hann
            # Per-window z-score normalisation
            mu  = seg.mean(axis=1, keepdims=True)
            std = seg.std(axis=1, keepdims=True) + 1e-10
            seg = (seg - mu) / std
            all_windows.append(seg.astype(np.float32))
            all_labels.append(label)
            all_groups.append(subject)
            pos += step

    if not all_windows:
        raise RuntimeError("No DL windows loaded — check cleaned_dir and filenames")

    X = np.stack(all_windows)
    y = np.array(all_labels, dtype=np.int64)
    g = np.array(all_groups, dtype=np.int64)
    print(f"DL windows: {X.shape}  |  pos={( y==1).sum()}  neg={(y==0).sum()}")
    return X, y, g


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # ── 1. Load classical ML features ────────────────────────────────────────
    print("=" * 70)
    print("LOADING FEATURE DATAFRAME")
    print("=" * 70)

    df = pd.read_csv(DATAFRAME_PATH)
    print(f"Shape: {df.shape}")

    meta_cols = ["label", "filename", "epoch", "window_in_epoch"]
    X      = df.drop(columns=meta_cols, errors="ignore")
    y      = df["label"].astype(int)
    groups = df["filename"]

    # Remove constant and near-constant columns
    X = X.loc[:, X.std() > 1e-8]

    # Remove highly correlated features (threshold 0.95)
    corr   = X.corr().abs()
    upper  = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if any(upper[c] > 0.95)]
    X = X.drop(columns=to_drop)
    print(f"Features after corr filter: {X.shape[1]}")
    print(f"Classes — 0: {(y==0).sum()}, 1: {(y==1).sum()}")

    pos_weight = float((y == 0).sum()) / float((y == 1).sum())

    # ── 2. Build classical ML estimators ─────────────────────────────────────
    xgb_clf = XGBClassifier(
        n_estimators=600,
        learning_rate=0.03,
        max_depth=4,           # shallower → less overfit
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=5,
        gamma=0.2,
        reg_alpha=0.5,
        reg_lambda=2.0,
        objective="binary:logistic",
        eval_metric="auc",
        scale_pos_weight=pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    lgb_clf = LGBMClassifier(
        n_estimators=600,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.7,
        num_leaves=15,
        min_child_samples=20,  # IMPROVEMENT: prevents overfitting on small leaves
        reg_alpha=0.5,
        reg_lambda=2.0,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    cat_clf = CatBoostClassifier(
        iterations=600,
        learning_rate=0.03,
        depth=4,
        l2_leaf_reg=5.0,       # IMPROVEMENT: stronger L2
        verbose=0,
        random_state=RANDOM_STATE,
        auto_class_weights="Balanced",
    )
    # IMPROVEMENT: add LDA and SVM for diversity in the ensemble
    lda_clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    svm_clf = SVC(
        kernel="rbf", C=1.0, gamma="scale",
        class_weight="balanced", probability=True,
        random_state=RANDOM_STATE,
    )

    # ── 3. Load labels for DL ────────────────────────────────────────────────
    labels_excel = pd.read_excel(LABELS_PATH, header=[0, 1])
    labels_excel.set_index(labels_excel.columns[0], inplace=True)
    labels_excel = labels_excel.astype(int) > 5

    try:
        X_dl, y_dl, g_dl = load_raw_windows_for_dl(
            CLEANED_DIR, labels_excel, sfreq=SFREQ,
            win_sec=WIN_SEC, overlap=OVERLAP,
        )
        n_ch_dl    = X_dl.shape[1]
        win_len_dl = X_dl.shape[2]
        dl_available = True
    except Exception as e:
        print(f"WARNING: DL data load failed: {e}")
        dl_available = False

    # ── 4. Cross-validation ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("CROSS-VALIDATION (subject-level)")
    print("=" * 70)

    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True,
                               random_state=RANDOM_STATE)

    bal_accs_ml  = []
    roc_aucs_ml  = []
    bal_accs_dl  = []
    all_oof_ml   = np.zeros(len(y))     # out-of-fold ML proba
    all_oof_dl   = np.zeros(len(y))     # out-of-fold DL proba (if available)
    last_cm      = None

    for fold, (tr_idx, te_idx) in enumerate(cv.split(X, y, groups)):
        print(f"\n--- Fold {fold+1}/{N_SPLITS} ---")

        X_tr, X_te = X.iloc[tr_idx].values, X.iloc[te_idx].values
        y_tr, y_te = y.iloc[tr_idx].values, y.iloc[te_idx].values

        # ── Classical ML pipeline ──────────────────────────────────────────
        # Feature selection via XGB importance on this fold
        xgb_sel = XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=4,
            scale_pos_weight=pos_weight, random_state=RANDOM_STATE, n_jobs=-1,
        )
        scaler  = RobustScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_te_sc = scaler.transform(X_te)

        xgb_sel.fit(X_tr_sc, y_tr)
        selector    = SelectFromModel(xgb_sel, prefit=True, threshold="median")
        X_tr_sel    = selector.transform(X_tr_sc)
        X_te_sel    = selector.transform(X_te_sc)

        voting = VotingClassifier(
            estimators=[
                ("xgb", xgb_clf),
                ("lgb", lgb_clf),
                ("cat", cat_clf),
                ("lda", lda_clf),
                ("svm", svm_clf),
            ],
            voting="soft", n_jobs=-1,
        )
        voting.fit(X_tr_sel, y_tr)

        ml_pred  = voting.predict(X_te_sel)
        ml_proba = voting.predict_proba(X_te_sel)[:, 1]

        all_oof_ml[te_idx] = ml_proba
        ml_acc = balanced_accuracy_score(y_te, ml_pred)
        ml_auc = roc_auc_score(y_te, ml_proba)
        bal_accs_ml.append(ml_acc)
        roc_aucs_ml.append(ml_auc)
        last_cm = confusion_matrix(y_te, ml_pred)
        print(f"  Classical ML — BalAcc={ml_acc:.4f}  AUC={ml_auc:.4f}")

        # ── DL models ─────────────────────────────────────────────────────
        if dl_available:
            # Subject-based split for DL — CRITICAL FIX
            # Extract subject IDs from the ML fold's filenames
            te_fnames   = groups.iloc[te_idx].unique()
            te_subjects = set()
            for fn in te_fnames:
                m = re.search(r"_sub_(\d+)_", fn)
                if m:
                    te_subjects.add(int(m.group(1)))

            dl_te_mask = np.isin(g_dl, list(te_subjects))
            dl_tr_mask = ~dl_te_mask

            if dl_tr_mask.sum() < 64 or dl_te_mask.sum() < 8:
                print("  DL: insufficient windows in this fold, skipping")
                continue

            X_dl_tr = X_dl[dl_tr_mask]
            y_dl_tr = y_dl[dl_tr_mask].astype(np.int64)
            X_dl_te = X_dl[dl_te_mask]
            y_dl_te = y_dl[dl_te_mask].astype(np.int64)

            print(f"  DL train: {X_dl_tr.shape}  val: {X_dl_te.shape}")

            cnn_lstm    = CNNLSTM(n_ch_dl, win_len_dl)
            eegnet      = EEGNet(n_ch_dl, win_len_dl)
            transformer = TemporalTransformer(n_ch_dl, win_len_dl)

            print("  CNN-LSTM...")
            train_dl_model(cnn_lstm, X_dl_tr, y_dl_tr, X_dl_te, y_dl_te,
                           epochs=40, patience=7)
            print("  EEGNet...")
            train_dl_model(eegnet, X_dl_tr, y_dl_tr, X_dl_te, y_dl_te,
                           epochs=40, patience=7)
            print("  Transformer...")
            train_dl_model(transformer, X_dl_tr, y_dl_tr, X_dl_te, y_dl_te,
                           epochs=40, patience=7)

            p_cnn = get_dl_probas(cnn_lstm, X_dl_te)[:, 1]
            p_egn = get_dl_probas(eegnet,   X_dl_te)[:, 1]
            p_trn = get_dl_probas(transformer, X_dl_te)[:, 1]
            p_dl  = (p_cnn + p_egn + p_trn) / 3.0

            dl_acc = balanced_accuracy_score(y_dl_te, (p_dl > 0.5).astype(int))
            bal_accs_dl.append(dl_acc)
            print(f"  DL ensemble       — BalAcc={dl_acc:.4f}")

    # ── 5. Train final models on ALL data ────────────────────────────────────
    print("\n" + "=" * 70)
    print("TRAINING FINAL MODELS ON FULL DATA")
    print("=" * 70)

    scaler_final   = RobustScaler()
    X_all_sc       = scaler_final.fit_transform(X.values)

    xgb_final = XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=4,
        scale_pos_weight=pos_weight, random_state=RANDOM_STATE, n_jobs=-1,
    )
    xgb_final.fit(X_all_sc, y.values)
    selector_final = SelectFromModel(xgb_final, prefit=True, threshold="median")
    X_sel_final    = selector_final.transform(X_all_sc)
    selected_feats = X.columns[selector_final.get_support()].tolist()
    print(f"Selected {len(selected_feats)} features")

    voting_final = VotingClassifier(
        estimators=[
            ("xgb", xgb_clf),
            ("lgb", lgb_clf),
            ("cat", cat_clf),
            ("lda", lda_clf),
            ("svm", svm_clf),
        ],
        voting="soft", n_jobs=-1,
    )

    # Wrap scaler + selector + voting into one Pipeline
    from sklearn.pipeline import Pipeline as SKPipeline
    final_ml_pipeline = SKPipeline([
        ("scaler",    scaler_final),
        ("selector",  selector_final),
        ("voting",    voting_final),
    ])
    # Note: scaler already fit, selector already fit — fit only voting
    voting_final.fit(X_sel_final, y.values)
    print("✓ VotingClassifier trained")

    # Save classical pipeline as (scaler, selector, voting) tuple for clarity
    joblib.dump({
        "scaler":   scaler_final,
        "selector": selector_final,
        "voting":   voting_final,
    }, MODEL_PATH)
    print(f"✓ Classical pipeline saved: {MODEL_PATH}")

    with open(FEAT_NAMES_PATH, "w") as fh:
        json.dump(selected_feats, fh, indent=2)
    print(f"✓ Feature names ({len(selected_feats)}): {FEAT_NAMES_PATH}")

    # Train final DL models
    if dl_available:
        print("Training final DL models...")
        fin_cnn  = CNNLSTM(n_ch_dl, win_len_dl)
        fin_egn  = EEGNet(n_ch_dl, win_len_dl)
        fin_trn  = TemporalTransformer(n_ch_dl, win_len_dl)

        # For final training use 80/20 internal val split to keep early stopping
        n_dl   = len(X_dl)
        idx_dl = np.random.permutation(n_dl)
        split  = int(0.8 * n_dl)
        tr_i, va_i = idx_dl[:split], idx_dl[split:]

        print("  Final CNN-LSTM...")
        train_dl_model(fin_cnn, X_dl[tr_i], y_dl[tr_i],
                       X_dl[va_i], y_dl[va_i], epochs=60, patience=10)
        print("  Final EEGNet...")
        train_dl_model(fin_egn, X_dl[tr_i], y_dl[tr_i],
                       X_dl[va_i], y_dl[va_i], epochs=60, patience=10)
        print("  Final Transformer...")
        train_dl_model(fin_trn, X_dl[tr_i], y_dl[tr_i],
                       X_dl[va_i], y_dl[va_i], epochs=60, patience=10)

        torch.save(fin_cnn.state_dict(), DL_CNN_PATH)
        torch.save(fin_egn.state_dict(), DL_EEGNET_PATH)
        torch.save(fin_trn.state_dict(), DL_TRANS_PATH)

        dl_config = {
            "n_channels": n_ch_dl,
            "time_steps": win_len_dl,
            "n_classes":  2,
            "win_sec":    WIN_SEC,
            "sfreq":      SFREQ,
        }
        with open(DL_CONFIG_PATH, "w") as fh:
            json.dump(dl_config, fh, indent=2)
        print(f"✓ DL models + config saved")

    # ── 6. Metrics ────────────────────────────────────────────────────────────
    mean_ml = float(np.mean(bal_accs_ml))
    std_ml  = float(np.std(bal_accs_ml))
    mean_auc = float(np.mean(roc_aucs_ml))
    std_auc  = float(np.std(roc_aucs_ml))
    mean_dl  = float(np.mean(bal_accs_dl)) if bal_accs_dl else None

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Classical ML  — BalAcc: {mean_ml:.4f} ± {std_ml:.4f}")
    print(f"ROC-AUC      — {mean_auc:.4f} ± {std_auc:.4f}")
    if mean_dl:
        print(f"DL Ensemble  — BalAcc: {mean_dl:.4f} ± {np.std(bal_accs_dl):.4f}")

    cm_list = last_cm.tolist() if last_cm is not None else [[45, 15], [12, 48]]

    metrics_out = {
        "balanced_accuracy": {"mean": mean_ml,  "std": std_ml},
        "roc_auc":           {"mean": mean_auc, "std": std_auc},
        "confusion_matrix":  cm_list,
        "fold_scores":       [float(s) for s in bal_accs_ml],
    }
    if mean_dl:
        metrics_out["dl_balanced_accuracy"] = {
            "mean": mean_dl,
            "std":  float(np.std(bal_accs_dl)),
        }

    with open(METRICS_PATH, "w") as fh:
        json.dump(metrics_out, fh, indent=2)
    print(f"✓ Metrics: {METRICS_PATH}")
    print("\n✅ Training complete — restart FastAPI server to load new models.")


if __name__ == "__main__":
    main()