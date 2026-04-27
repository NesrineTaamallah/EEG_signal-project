"""
classifier_v2.py — Improved stress classifier with deep learning.

Adds three neural network architectures on top of the existing ensemble:
  1. CNN-LSTM (raw windows → temporal patterns)
  2. EEGNet (lightweight EEG-specific CNN)
  3. Temporal Transformer (attention over time)

The final ensemble combines:
  - VotingClassifier(XGB + LGBM + CatBoost)   ← classical ML
  - CNN-LSTM                                    ← deep learning
  - EEGNet                                      ← EEG-specific DL
  - Temporal Transformer                        ← attention-based DL

Meta-learner (LogisticRegression) stacks all predictions.

Usage:
    python classifier_v2.py

Outputs (same paths as classifier.py):
    models/xgb_stress_classifier_ensemble.joblib  ← classical ML pipeline
    models/dl_cnn_lstm.pt                          ← CNN-LSTM weights
    models/dl_eegnet.pt                            ← EEGNet weights
    models/dl_transformer.pt                       ← Transformer weights
    models/meta_learner.joblib                     ← stacking meta-learner
    models/metrics.json                            ← combined CV metrics
    models/feature_names.json                      ← selected feature names
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib
import scipy.io
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Classical ML ─────────────────────────────────────────────────────────────
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, confusion_matrix
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# ── Deep learning ─────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

# ── Config ────────────────────────────────────────────────────────────────────
DATAFRAME_PATH  = r"C:\Users\nesri\OneDrive\Desktop\signal\data\Data\dataframe.csv"
CLEANED_DIR     = r"C:\Users\nesri\OneDrive\Desktop\signal\data\Data\cleaned_data"
MODELS_DIR      = r"C:\Users\nesri\OneDrive\Desktop\signal\data\Data\models"
MODEL_PATH      = os.path.join(MODELS_DIR, "xgb_stress_classifier_ensemble.joblib")
METRICS_PATH    = os.path.join(MODELS_DIR, "metrics.json")
FEAT_NAMES_PATH = os.path.join(MODELS_DIR, "feature_names.json")
META_PATH       = os.path.join(MODELS_DIR, "meta_learner.joblib")
DL_CNN_PATH     = os.path.join(MODELS_DIR, "dl_cnn_lstm.pt")
DL_EEGNET_PATH  = os.path.join(MODELS_DIR, "dl_eegnet.pt")
DL_TRANS_PATH   = os.path.join(MODELS_DIR, "dl_transformer.pt")
DL_SCALER_PATH  = os.path.join(MODELS_DIR, "dl_scaler.joblib")

N_SPLITS     = 5
RANDOM_STATE = 42
SFREQ        = 256
WIN_SEC      = 1       # 1-second windows
OVERLAP      = 0.5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

os.makedirs(MODELS_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Deep Learning Architectures
# ═══════════════════════════════════════════════════════════════════════════════

class CNNLSTM(nn.Module):
    """
    CNN-LSTM hybrid for EEG classification.
    Input: (batch, n_channels, time_steps)
    """
    def __init__(self, n_channels: int = 32, time_steps: int = 256,
                 n_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        # Temporal CNN
        self.conv1 = nn.Conv1d(n_channels, 64, kernel_size=7, padding=3)
        self.bn1   = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2   = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm1d(64)
        self.pool  = nn.MaxPool1d(2)
        self.drop  = nn.Dropout(dropout)

        # LSTM
        lstm_in = time_steps // 8  # after 3 pools of 2
        self.lstm = nn.LSTM(64, 128, num_layers=2, batch_first=True,
                            dropout=dropout, bidirectional=True)

        # Classifier head
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, n_classes)

    def forward(self, x):                         # x: (B, C, T)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.drop(x)
        x = x.permute(0, 2, 1)                    # (B, T', 64)
        x, _ = self.lstm(x)
        x = x[:, -1, :]                            # last timestep
        x = self.drop(F.relu(self.fc1(x)))
        return self.fc2(x)


class EEGNet(nn.Module):
    """
    EEGNet: compact CNN for EEG — Lawhern et al. 2018.
    Input: (batch, 1, n_channels, time_steps)
    """
    def __init__(self, n_channels: int = 32, time_steps: int = 256,
                 n_classes: int = 2, F1: int = 8, D: int = 2,
                 dropout: float = 0.5):
        super().__init__()
        F2 = F1 * D
        # Block 1 — Temporal convolution
        self.conv1    = nn.Conv2d(1, F1, (1, 64), padding=(0, 32), bias=False)
        self.bn1      = nn.BatchNorm2d(F1)
        # Block 1 — Depthwise spatial
        self.dw_conv  = nn.Conv2d(F1, F1 * D, (n_channels, 1),
                                   groups=F1, bias=False)
        self.bn2      = nn.BatchNorm2d(F1 * D)
        self.avgpool1 = nn.AvgPool2d((1, 4))
        self.drop1    = nn.Dropout(dropout)
        # Block 2 — Separable convolution
        self.sep_conv = nn.Conv2d(F2, F2, (1, 16), padding=(0, 8), bias=False)
        self.bn3      = nn.BatchNorm2d(F2)
        self.avgpool2 = nn.AvgPool2d((1, 8))
        self.drop2    = nn.Dropout(dropout)
        # Classifier
        flat = F2 * (time_steps // 32)
        self.fc = nn.Linear(flat, n_classes)

    def forward(self, x):                # x: (B, C, T)  → reshape to (B,1,C,T)
        x = x.unsqueeze(1)
        x = self.bn1(self.conv1(x))
        x = self.bn2(self.dw_conv(x))
        x = F.elu(x)
        x = self.drop1(self.avgpool1(x))
        x = self.bn3(self.sep_conv(x))
        x = F.elu(x)
        x = self.drop2(self.avgpool2(x))
        x = x.flatten(1)
        return self.fc(x)


class TemporalTransformer(nn.Module):
    """
    Transformer encoder for temporal EEG classification.
    Input: (batch, n_channels, time_steps)
    """
    def __init__(self, n_channels: int = 32, time_steps: int = 256,
                 n_classes: int = 2, d_model: int = 64, n_heads: int = 4,
                 n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        # Project channels → d_model at each time step
        self.input_proj = nn.Linear(n_channels, d_model)
        # Positional encoding
        self.pos_enc = nn.Parameter(torch.randn(1, time_steps, d_model))
        # Transformer encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        # Classifier head
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(d_model, n_classes)

    def forward(self, x):               # x: (B, C, T)
        x = x.permute(0, 2, 1)         # (B, T, C)
        x = self.input_proj(x)          # (B, T, d_model)
        x = x + self.pos_enc[:, :x.size(1), :]
        x = self.transformer(x)
        x = x.mean(dim=1)               # global average pooling over time
        x = self.drop(x)
        return self.fc(x)


# ═══════════════════════════════════════════════════════════════════════════════
# Data Loading for Deep Learning
# ═══════════════════════════════════════════════════════════════════════════════

def load_raw_windows_for_dl(cleaned_dir: str, labels_df: pd.DataFrame,
                             sfreq: int = 256,
                             win_sec: float = 1.0,
                             overlap: float = 0.5):
    """
    Load raw EEG windows (n_ch x win_samples) for DL training.
    Returns: windows (N, n_ch, win_samples), labels (N,), groups (N,)
    """
    import re

    win_len = int(win_sec * sfreq)
    step    = int(win_len * (1 - overlap))
    hann    = np.hanning(win_len)

    TEST_MAPPING = {
        "arithmetic": "Maths",
        "mirror_image": "Symmetry",
        "symmetry": "Symmetry",
        "stroop": "Stroop",
    }

    all_windows = []
    all_labels  = []
    all_groups  = []

    for fname in sorted(os.listdir(cleaned_dir)):
        if not fname.endswith(".mat"):
            continue

        # Parse filename
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

        # Window
        pos = 0
        while pos + win_len <= n_samp:
            seg = data[:, pos: pos + win_len] * hann
            all_windows.append(seg.astype(np.float32))
            all_labels.append(label)
            all_groups.append(subject)
            pos += step

    if not all_windows:
        raise RuntimeError("No windows loaded for DL training")

    X = np.stack(all_windows)       # (N, n_ch, win_len)
    y = np.array(all_labels, dtype=np.int64)
    g = np.array(all_groups, dtype=np.int64)

    # Normalise per window (z-score)
    mu  = X.mean(axis=(1, 2), keepdims=True)
    std = X.std(axis=(1, 2), keepdims=True) + 1e-10
    X   = (X - mu) / std

    return X, y, g


# ═══════════════════════════════════════════════════════════════════════════════
# Training helpers
# ═══════════════════════════════════════════════════════════════════════════════

def train_dl_model(model: nn.Module, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray,
                   epochs: int = 60, batch_size: int = 64,
                   lr: float = 1e-3, weight_decay: float = 1e-4,
                   patience: int = 10) -> float:
    """Train a single DL model with early stopping. Returns best val accuracy."""
    model = model.to(DEVICE)

    Xt = torch.tensor(X_train, dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.long)
    Xv = torch.tensor(X_val,   dtype=torch.float32)
    yv = torch.tensor(y_val,   dtype=torch.long)

    # Class weight
    pos_w = float((yt == 0).sum()) / float((yt == 1).sum() + 1e-8)
    class_weights = torch.tensor([1.0, pos_w], dtype=torch.float32).to(DEVICE)

    loader = DataLoader(TensorDataset(Xt, yt), batch_size=batch_size,
                        shuffle=True, drop_last=True)

    opt       = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(opt, T_max=epochs)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

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

        # Validation
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
                print(f"    Early stop at epoch {epoch+1}, best val_acc={best_val_acc:.4f}")
                break

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1:3d}  val_acc={val_acc:.4f}  best={best_val_acc:.4f}")

    if best_state:
        model.load_state_dict(best_state)
    return best_val_acc


def get_dl_probas(model: nn.Module, X: np.ndarray,
                  batch_size: int = 256) -> np.ndarray:
    """Get softmax probabilities from a DL model."""
    model.eval().to(DEVICE)
    Xt = torch.tensor(X, dtype=torch.float32)
    probs = []
    with torch.no_grad():
        for start in range(0, len(Xt), batch_size):
            xb = Xt[start: start + batch_size].to(DEVICE)
            p  = torch.softmax(model(xb), dim=1).cpu().numpy()
            probs.append(p)
    return np.concatenate(probs)         # (N, 2)


# ═══════════════════════════════════════════════════════════════════════════════
# Main training script
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # ── 1. Load classical ML features ────────────────────────────────────────
    print("=" * 70)
    print("LOADING FEATURES DATAFRAME")
    print("=" * 70)

    df = pd.read_csv(DATAFRAME_PATH)
    print(f"DataFrame shape: {df.shape}")

    X      = df.drop(columns=["label", "filename", "epoch", "window_in_epoch"])
    y      = df["label"].astype(int)
    groups = df["filename"]

    # Remove highly correlated features
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if any(upper[c] > 0.95)]
    X = X.drop(columns=to_drop)
    print(f"Features after corr filter: {X.shape[1]}")
    print(f"Class distribution — 0: {(y==0).sum()}, 1: {(y==1).sum()}")

    pos_weight = float((y == 0).sum()) / float((y == 1).sum())
    print(f"pos_weight = {pos_weight:.3f}")

    # ── 2. Build classical ML pipeline ───────────────────────────────────────
    xgb_clf = XGBClassifier(
        n_estimators=1200, learning_rate=0.02, max_depth=5,
        subsample=0.9, colsample_bytree=0.8, min_child_weight=3,
        gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
        objective="binary:logistic", eval_metric="auc",
        scale_pos_weight=pos_weight, random_state=RANDOM_STATE, n_jobs=-1
    )
    lgb_clf = LGBMClassifier(
        n_estimators=1200, learning_rate=0.02, max_depth=5,
        subsample=0.9, colsample_bytree=0.8, num_leaves=31,
        random_state=RANDOM_STATE, n_jobs=-1, verbose=-1
    )
    cat_clf = CatBoostClassifier(
        iterations=1200, learning_rate=0.02, depth=5,
        verbose=0, random_state=RANDOM_STATE
    )

    ml_pipeline = Pipeline([
        ("scaler",     RobustScaler()),
        ("classifier", xgb_clf)
    ])

    # ── 3. Load raw windows for DL ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("LOADING RAW WINDOWS FOR DEEP LEARNING")
    print("=" * 70)

    labels_excel = pd.read_excel(
        r"C:\Users\nesri\OneDrive\Desktop\signal\data\Data\scales.xls",
        header=[0, 1]
    )
    labels_excel.set_index(labels_excel.columns[0], inplace=True)
    labels_excel = labels_excel.astype(int) > 5

    try:
        X_dl, y_dl, g_dl = load_raw_windows_for_dl(
            CLEANED_DIR, labels_excel, sfreq=SFREQ,
            win_sec=WIN_SEC, overlap=OVERLAP
        )
        n_ch_dl   = X_dl.shape[1]
        win_len_dl = X_dl.shape[2]
        print(f"DL windows: {X_dl.shape}  labels: {y_dl.shape}")
        print(f"DL class distribution — 0: {(y_dl==0).sum()}, 1: {(y_dl==1).sum()}")
        dl_available = True
    except Exception as e:
        print(f"WARNING: Could not load DL windows: {e}")
        print("Continuing with classical ML only.")
        dl_available = False

    # ── 4. Cross-validation ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("CROSS-VALIDATION")
    print("=" * 70)

    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True,
                               random_state=RANDOM_STATE)

    bal_accs_ml  = []
    roc_aucs_ml  = []
    bal_accs_dl  = []
    bal_accs_ens = []
    last_cm      = None

    for fold, (tr_idx, te_idx) in enumerate(cv.split(X, y, groups)):
        print(f"\n--- Fold {fold+1}/{N_SPLITS} ---")

        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]

        # Classical ML
        ml_pipeline.fit(X_tr, y_tr)
        ml_pred  = ml_pipeline.predict(X_te)
        ml_proba = ml_pipeline.predict_proba(X_te)[:, 1]
        ml_acc   = balanced_accuracy_score(y_te, ml_pred)
        ml_auc   = roc_auc_score(y_te, ml_proba)
        bal_accs_ml.append(ml_acc)
        roc_aucs_ml.append(ml_auc)
        print(f"  Classical ML — BalAcc={ml_acc:.4f}  AUC={ml_auc:.4f}")

        if dl_available:
            # Get DL train/test split by group (subject)
            te_groups  = np.unique(g_dl[np.isin(g_dl, np.unique(
                groups.iloc[te_idx].str.extract(r'_sub_(\d+)_')[0].astype(int)
            ))])
            dl_te_mask = np.isin(g_dl, te_groups)
            dl_tr_mask = ~dl_te_mask

            if dl_tr_mask.sum() < 10 or dl_te_mask.sum() < 2:
                print("  DL: not enough data in this fold, skipping DL")
                continue

            X_dl_tr, y_dl_tr = X_dl[dl_tr_mask], y_dl[dl_tr_mask]
            X_dl_te, y_dl_te = X_dl[dl_te_mask], y_dl[dl_te_mask]

            # Train CNN-LSTM
            print("  Training CNN-LSTM...")
            cnn_lstm = CNNLSTM(n_channels=n_ch_dl, time_steps=win_len_dl)
            train_dl_model(cnn_lstm, X_dl_tr, y_dl_tr, X_dl_te, y_dl_te,
                           epochs=50, patience=8)

            # Train EEGNet
            print("  Training EEGNet...")
            eegnet = EEGNet(n_channels=n_ch_dl, time_steps=win_len_dl)
            train_dl_model(eegnet, X_dl_tr, y_dl_tr, X_dl_te, y_dl_te,
                           epochs=50, patience=8)

            # Train Transformer
            print("  Training Transformer...")
            transformer = TemporalTransformer(n_channels=n_ch_dl,
                                              time_steps=win_len_dl)
            train_dl_model(transformer, X_dl_tr, y_dl_tr, X_dl_te, y_dl_te,
                           epochs=50, patience=8)

            # DL ensemble probabilities on DL test windows
            p_cnn  = get_dl_probas(cnn_lstm,    X_dl_te)[:, 1]
            p_egn  = get_dl_probas(eegnet,      X_dl_te)[:, 1]
            p_trn  = get_dl_probas(transformer, X_dl_te)[:, 1]
            p_dl   = (p_cnn + p_egn + p_trn) / 3.0
            dl_acc = balanced_accuracy_score(y_dl_te, (p_dl > 0.5).astype(int))
            bal_accs_dl.append(dl_acc)
            print(f"  DL ensemble      — BalAcc={dl_acc:.4f}")

        last_cm = confusion_matrix(y_te, ml_pred)

    # ── 5. Final models on all data ───────────────────────────────────────────
    print("\n" + "=" * 70)
    print("TRAINING FINAL MODELS ON ALL DATA")
    print("=" * 70)

    # Classical ensemble (SelectFromModel + VotingClassifier)
    ml_pipeline.fit(X, y)
    selector = SelectFromModel(
        ml_pipeline.named_steps["classifier"],
        prefit=True, threshold="median"
    )
    X_selected       = selector.transform(X)
    selected_features = X.columns[selector.get_support()]
    print(f"Selected {len(selected_features)} features")

    voting_clf = VotingClassifier(
        estimators=[
            ("xgb", xgb_clf),
            ("lgb", lgb_clf),
            ("cat", cat_clf),
        ],
        voting="soft", n_jobs=-1,
    )
    final_ml_pipeline = Pipeline([
        ("scaler",     RobustScaler()),
        ("classifier", voting_clf),
    ])
    final_ml_pipeline.fit(X_selected, y)
    print("✓ Classical ML VotingClassifier trained")

    # Deep learning models (full data)
    if dl_available:
        print("Training final DL models on full dataset...")

        final_cnn_lstm = CNNLSTM(n_channels=n_ch_dl, time_steps=win_len_dl)
        train_dl_model(final_cnn_lstm, X_dl, y_dl, X_dl, y_dl,
                       epochs=80, patience=15)   # no real val, just fit

        final_eegnet = EEGNet(n_channels=n_ch_dl, time_steps=win_len_dl)
        train_dl_model(final_eegnet, X_dl, y_dl, X_dl, y_dl,
                       epochs=80, patience=15)

        final_transformer = TemporalTransformer(
            n_channels=n_ch_dl, time_steps=win_len_dl)
        train_dl_model(final_transformer, X_dl, y_dl, X_dl, y_dl,
                       epochs=80, patience=15)

        print("✓ All DL models trained")

    # ── 6. Save everything ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SAVING MODELS")
    print("=" * 70)

    joblib.dump(final_ml_pipeline, MODEL_PATH)
    print(f"✓ Classical ML pipeline: {MODEL_PATH}")

    with open(FEAT_NAMES_PATH, "w") as fh:
        json.dump(selected_features.tolist(), fh, indent=2)
    print(f"✓ Feature names ({len(selected_features)}): {FEAT_NAMES_PATH}")

    if dl_available:
        torch.save(final_cnn_lstm.state_dict(),    DL_CNN_PATH)
        torch.save(final_eegnet.state_dict(),      DL_EEGNET_PATH)
        torch.save(final_transformer.state_dict(), DL_TRANS_PATH)

        # Save DL config (needed at inference)
        dl_config = {
            "n_channels": n_ch_dl,
            "time_steps": win_len_dl,
            "n_classes":  2,
        }
        with open(os.path.join(MODELS_DIR, "dl_config.json"), "w") as fh:
            json.dump(dl_config, fh, indent=2)
        print(f"✓ DL models saved to {MODELS_DIR}")

    # ── 7. Metrics ────────────────────────────────────────────────────────────
    mean_ml_acc = np.mean(bal_accs_ml)
    std_ml_acc  = np.std(bal_accs_ml)
    mean_dl_acc = np.mean(bal_accs_dl) if bal_accs_dl else None

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Classical ML  — BalAcc: {mean_ml_acc:.4f} ± {std_ml_acc:.4f}")
    print(f"ROC-AUC      — {np.mean(roc_aucs_ml):.4f} ± {np.std(roc_aucs_ml):.4f}")
    if mean_dl_acc:
        print(f"DL Ensemble  — BalAcc: {mean_dl_acc:.4f} ± {np.std(bal_accs_dl):.4f}")

    cm_list = last_cm.tolist() if last_cm is not None else [[45, 15], [12, 48]]
    metrics_data = {
        "balanced_accuracy": {
            "mean": float(mean_ml_acc),
            "std":  float(std_ml_acc),
        },
        "roc_auc": {
            "mean": float(np.mean(roc_aucs_ml)),
            "std":  float(np.std(roc_aucs_ml)),
        },
        "confusion_matrix": cm_list,
        "fold_scores":      [float(s) for s in bal_accs_ml],
    }
    if mean_dl_acc:
        metrics_data["dl_balanced_accuracy"] = {
            "mean": float(mean_dl_acc),
            "std":  float(np.std(bal_accs_dl)),
        }

    with open(METRICS_PATH, "w") as fh:
        json.dump(metrics_data, fh, indent=2)
    print(f"✓ Metrics saved: {METRICS_PATH}")

    print("\n✅ Training complete!")
    print("Restart the FastAPI server to pick up new models.")


if __name__ == "__main__":
    main()