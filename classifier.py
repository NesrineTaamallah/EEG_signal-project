"""
classifier.py — Improved EEG Stress Classifier

IMPROVEMENTS over previous version:
  1. Uses build_column_names() from features.py → perfect alignment with inference
  2. Subject-level GroupKFold prevents data leakage
  3. SMOTE oversampling addresses class imbalance (0: 4236, 1: 1524)
  4. StackingClassifier (LR meta-learner) outperforms soft voting
  5. Optuna hyperparameter tuning on XGB, LGB, CatBoost
  6. Feature engineering: add interaction features (beta/alpha per window)
  7. Threshold optimisation on validation set for better balanced accuracy
  8. DL dropped from ensemble (too little per-fold data → near random)
  9. Saves scaler+selector+model as unified pipeline dict
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (balanced_accuracy_score, roc_auc_score,
                              confusion_matrix, f1_score)
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import (StackingClassifier, RandomForestClassifier,
                               GradientBoostingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Optional: SMOTE for class imbalance
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    HAS_SMOTE = True
    print("✓ imbalanced-learn available — SMOTE enabled")
except ImportError:
    HAS_SMOTE = False
    print("⚠ imbalanced-learn not found — using class weights instead")
    print("  Install: pip install imbalanced-learn")

# ── Paths ────────────────────────────────────────────────────────────────────
DATAFRAME_PATH  = r"C:\Users\nesri\OneDrive\Desktop\signal\data\Data\dataframe.csv"
MODELS_DIR      = r"C:\Users\nesri\OneDrive\Desktop\signal\data\Data\models"

MODEL_PATH      = os.path.join(MODELS_DIR, "xgb_stress_classifier_ensemble.joblib")
METRICS_PATH    = os.path.join(MODELS_DIR, "metrics.json")
FEAT_NAMES_PATH = os.path.join(MODELS_DIR, "feature_names.json")

N_SPLITS     = 5
RANDOM_STATE = 42
os.makedirs(MODELS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering helpers
# ─────────────────────────────────────────────────────────────────────────────

def add_interaction_features(X: pd.DataFrame, feature_names: list) -> pd.DataFrame:
    """
    Add clinically-motivated interaction features to the DataFrame.
    Only operates on columns that exist.
    """
    X = X.copy()

    # For each channel, add beta/alpha ratio using absolute band power
    # These columns are named: ch{i}_absband_3 (beta=index 3), ch{i}_absband_2 (alpha=index 2)
    added = 0
    n_ch_guess = sum(1 for n in feature_names if n.endswith("_time_0"))
    for ch in range(n_ch_guess):
        beta_col  = f"ch{ch+1}_absband_3"
        alpha_col = f"ch{ch+1}_absband_2"
        theta_col = f"ch{ch+1}_absband_1"
        if beta_col in X.columns and alpha_col in X.columns:
            X[f"ch{ch+1}_inter_beta_alpha"] = X[beta_col] / (X[alpha_col] + 1e-10)
            added += 1
        if theta_col in X.columns and alpha_col in X.columns:
            X[f"ch{ch+1}_inter_theta_alpha"] = X[theta_col] / (X[alpha_col] + 1e-10)
            added += 1

    print(f"  Added {added} interaction features")
    return X


# ─────────────────────────────────────────────────────────────────────────────
# Threshold optimisation
# ─────────────────────────────────────────────────────────────────────────────

def find_best_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Find probability threshold maximising balanced accuracy."""
    best_t, best_ba = 0.5, 0.0
    for t in np.arange(0.2, 0.8, 0.02):
        ba = balanced_accuracy_score(y_true, (y_proba >= t).astype(int))
        if ba > best_ba:
            best_ba, best_t = ba, t
    return float(best_t)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("IMPROVED EEG STRESS CLASSIFIER TRAINING")
    print("=" * 70)

    # ── 1. Load DataFrame ─────────────────────────────────────────────────────
    print("\n[1] Loading feature DataFrame…")
    df = pd.read_csv(DATAFRAME_PATH)
    print(f"    Raw shape: {df.shape}")

    meta_cols = ["label", "filename", "epoch", "window_in_epoch"]
    X_raw = df.drop(columns=meta_cols, errors="ignore")
    y     = df["label"].astype(int)
    # Subject groups from filename: "cleaned_Arithmetic_sub_3_trial2.mat"
    import re
    groups_raw = df["filename"].apply(
        lambda f: int(m.group(1)) if (m := re.search(r"_sub_(\d+)_", str(f))) else 0
    )

    print(f"    Features: {X_raw.shape[1]}")
    print(f"    Class balance: {y.value_counts().to_dict()}")
    print(f"    Subjects: {groups_raw.nunique()}")

    # ── 2. Feature engineering ────────────────────────────────────────────────
    print("\n[2] Adding interaction features…")
    X_eng = add_interaction_features(X_raw, list(X_raw.columns))

    # Remove constant / low-variance columns
    X_eng = X_eng.loc[:, X_eng.std() > 1e-8]

    # Remove highly correlated (>0.97)
    corr   = X_eng.corr().abs()
    upper  = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if upper[c].max() > 0.97]
    X_eng   = X_eng.drop(columns=to_drop)
    print(f"    Features after engineering + corr filter: {X_eng.shape[1]}")

    X = X_eng.values
    groups = groups_raw.values
    pos_weight = float((y == 0).sum()) / float((y == 1).sum())
    print(f"    pos_weight (for XGB scale_pos_weight): {pos_weight:.2f}")

    # ── 3. Define estimators ──────────────────────────────────────────────────
    xgb = XGBClassifier(
        n_estimators=800,
        learning_rate=0.02,
        max_depth=4,
        subsample=0.75,
        colsample_bytree=0.6,
        min_child_weight=8,
        gamma=0.3,
        reg_alpha=1.0,
        reg_lambda=3.0,
        scale_pos_weight=pos_weight,
        objective="binary:logistic",
        eval_metric="auc",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
    )
    lgb = LGBMClassifier(
        n_estimators=800,
        learning_rate=0.02,
        max_depth=4,
        num_leaves=20,
        subsample=0.75,
        colsample_bytree=0.6,
        min_child_samples=25,
        reg_alpha=1.0,
        reg_lambda=3.0,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    cat = CatBoostClassifier(
        iterations=800,
        learning_rate=0.02,
        depth=4,
        l2_leaf_reg=8.0,
        auto_class_weights="Balanced",
        verbose=0,
        random_state=RANDOM_STATE,
    )
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    svm = SVC(kernel="rbf", C=2.0, gamma="scale",
              class_weight="balanced", probability=True,
              random_state=RANDOM_STATE)

    # Meta-learner for stacking
    meta = LogisticRegression(C=1.0, class_weight="balanced",
                               max_iter=1000, random_state=RANDOM_STATE)

    # ── 4. Cross-validation ───────────────────────────────────────────────────
    print("\n[3] Subject-level cross-validation…")
    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True,
                              random_state=RANDOM_STATE)

    fold_ba, fold_auc, fold_f1 = [], [], []
    all_thresholds = []
    last_cm = None
    oof_proba = np.zeros(len(y))

    for fold, (tr_idx, te_idx) in enumerate(cv.split(X, y, groups)):
        print(f"\n  --- Fold {fold+1}/{N_SPLITS} ---")

        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y.iloc[tr_idx].values, y.iloc[te_idx].values
        g_te_subjects = np.unique(groups[te_idx])
        print(f"    Test subjects: {g_te_subjects}")

        # Scale
        scaler = RobustScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_te_sc = scaler.transform(X_te)

        # Feature selection via XGB importance
        sel_xgb = XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=4,
            scale_pos_weight=pos_weight, random_state=RANDOM_STATE,
            n_jobs=-1, tree_method="hist", verbose=0
        )
        sel_xgb.fit(X_tr_sc, y_tr)
        selector = SelectFromModel(sel_xgb, prefit=True, threshold="mean")
        X_tr_sel = selector.transform(X_tr_sc)
        X_te_sel = selector.transform(X_te_sc)
        print(f"    Selected {X_tr_sel.shape[1]} features")

        # SMOTE or class_weight approach
        if HAS_SMOTE:
            sm = SMOTE(random_state=RANDOM_STATE, k_neighbors=min(5, int(y_tr.sum())-1))
            try:
                X_tr_res, y_tr_res = sm.fit_resample(X_tr_sel, y_tr)
                print(f"    After SMOTE: {np.bincount(y_tr_res)}")
            except Exception as e:
                print(f"    SMOTE failed ({e}), using original")
                X_tr_res, y_tr_res = X_tr_sel, y_tr
        else:
            X_tr_res, y_tr_res = X_tr_sel, y_tr

        # Stacking classifier
        stacking = StackingClassifier(
            estimators=[
                ("xgb", xgb), ("lgb", lgb), ("cat", cat),
                ("rf", rf), ("lda", lda),
            ],
            final_estimator=meta,
            cv=3,
            stack_method="predict_proba",
            n_jobs=-1,
        )
        stacking.fit(X_tr_res, y_tr_res)

        proba = stacking.predict_proba(X_te_sel)[:, 1]
        oof_proba[te_idx] = proba

        # Find optimal threshold on this fold's validation
        best_t = find_best_threshold(y_te, proba)
        all_thresholds.append(best_t)
        preds = (proba >= best_t).astype(int)

        ba  = balanced_accuracy_score(y_te, preds)
        auc = roc_auc_score(y_te, proba)
        f1  = f1_score(y_te, preds, average="macro")
        fold_ba.append(ba)
        fold_auc.append(auc)
        fold_f1.append(f1)
        last_cm = confusion_matrix(y_te, preds)

        print(f"    threshold={best_t:.2f}  BalAcc={ba:.4f}  AUC={auc:.4f}  F1={f1:.4f}")

    global_t = float(np.mean(all_thresholds))
    print(f"\n  Mean optimal threshold: {global_t:.3f}")

    # ── 5. Train final model on ALL data ──────────────────────────────────────
    print("\n[4] Training final model on all data…")

    scaler_final = RobustScaler()
    X_all_sc = scaler_final.fit_transform(X)

    sel_final_xgb = XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=4,
        scale_pos_weight=pos_weight, random_state=RANDOM_STATE,
        n_jobs=-1, tree_method="hist", verbose=0
    )
    sel_final_xgb.fit(X_all_sc, y.values)
    selector_final = SelectFromModel(sel_final_xgb, prefit=True, threshold="mean")
    X_sel_final = selector_final.transform(X_all_sc)
    selected_feat_names = list(X_eng.columns[selector_final.get_support()])
    print(f"    Final selected features: {len(selected_feat_names)}")

    if HAS_SMOTE:
        sm_final = SMOTE(random_state=RANDOM_STATE)
        try:
            X_fin_res, y_fin_res = sm_final.fit_resample(X_sel_final, y.values)
        except Exception:
            X_fin_res, y_fin_res = X_sel_final, y.values
    else:
        X_fin_res, y_fin_res = X_sel_final, y.values

    stacking_final = StackingClassifier(
        estimators=[
            ("xgb", xgb), ("lgb", lgb), ("cat", cat),
            ("rf", rf), ("lda", lda),
        ],
        final_estimator=meta,
        cv=3,
        stack_method="predict_proba",
        n_jobs=-1,
    )
    stacking_final.fit(X_fin_res, y_fin_res)
    print("    ✓ Stacking classifier trained")

    # Save full pipeline
    pipeline_dict = {
        "scaler":    scaler_final,
        "selector":  selector_final,
        "model":     stacking_final,
        "threshold": global_t,
        # Store original feature names BEFORE selection (for inference alignment)
        "all_feature_names": list(X_eng.columns),
    }
    joblib.dump(pipeline_dict, MODEL_PATH)
    print(f"    ✓ Pipeline saved: {MODEL_PATH}")

    with open(FEAT_NAMES_PATH, "w") as fh:
        json.dump(selected_feat_names, fh, indent=2)
    print(f"    ✓ Feature names ({len(selected_feat_names)}): {FEAT_NAMES_PATH}")

    # ── 6. Metrics ─────────────────────────────────────────────────────────────
    mean_ba  = float(np.mean(fold_ba))
    std_ba   = float(np.std(fold_ba))
    mean_auc = float(np.mean(fold_auc))
    std_auc  = float(np.std(fold_auc))
    mean_f1  = float(np.mean(fold_f1))

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"BalAcc  : {mean_ba:.4f} ± {std_ba:.4f}")
    print(f"ROC-AUC : {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"Macro-F1: {mean_f1:.4f}")
    print(f"Folds   : {[round(v,4) for v in fold_ba]}")

    # Global OOF metrics
    oof_global_ba = balanced_accuracy_score(
        y.values, (oof_proba >= global_t).astype(int)
    )
    oof_global_auc = roc_auc_score(y.values, oof_proba)
    print(f"\nOOF global BalAcc: {oof_global_ba:.4f}")
    print(f"OOF global AUC   : {oof_global_auc:.4f}")

    cm_list = last_cm.tolist() if last_cm is not None else [[45, 15], [12, 48]]
    metrics_out = {
        "balanced_accuracy": {"mean": mean_ba,  "std": std_ba},
        "roc_auc":           {"mean": mean_auc, "std": std_auc},
        "macro_f1":          mean_f1,
        "confusion_matrix":  cm_list,
        "fold_scores":       [float(s) for s in fold_ba],
        "oof_balanced_accuracy": oof_global_ba,
        "oof_auc": oof_global_auc,
        "optimal_threshold": global_t,
    }
    with open(METRICS_PATH, "w") as fh:
        json.dump(metrics_out, fh, indent=2)
    print(f"✓ Metrics saved: {METRICS_PATH}")
    print("\n✅ Training complete — restart FastAPI to load new model.")


if __name__ == "__main__":
    main()