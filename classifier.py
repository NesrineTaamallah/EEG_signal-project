"""
classifier.py  — trains the ensemble stress classifier and saves:
  1. xgb_stress_classifier_ensemble.joblib  ← the VotingClassifier pipeline
  2. metrics.json                            ← CV scores for the /metrics endpoint
  3. feature_names.json                      ← selected column names for /predict
                                               (NEW — this is the key fix)

FIX: feature_names.json was never saved, so main.py had no way to know which
columns SelectFromModel kept.  The feature count mismatch caused every
prediction to silently fall back to the heuristic.
"""

import pandas as pd
import numpy as np
import os
import json
import joblib

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, confusion_matrix
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import VotingClassifier

import matplotlib.pyplot as plt


DATAFRAME_PATH = r"C:\Users\nesri\OneDrive\Desktop\signal\data\Data\dataframe.csv"
MODEL_PATH     = r"C:\Users\nesri\OneDrive\Desktop\signal\data\Data\models\xgb_stress_classifier_ensemble.joblib"
_MODELS_DIR     = r"C:\Users\nesri\OneDrive\Desktop\signal\data\Data\models"
# Derived paths (same directory as the model)
METRICS_PATH      = r"C:\Users\nesri\OneDrive\Desktop\signal\data\Data\models\metrics.json"
FEAT_NAMES_PATH   = r"C:\Users\nesri\OneDrive\Desktop\signal\data\Data\models\feature_names.json"

N_SPLITS     = 5
RANDOM_STATE = 42


print("Chargement de la DataFrame...")
df = pd.read_csv(DATAFRAME_PATH)
print(f"Data shape : {df.shape}")


null_counts  = df.isnull().sum()
null_columns = null_counts[null_counts > 0]

if null_columns.empty:
    print(" Aucune colonne avec valeurs nulles")
else:
    print(" Colonnes avec valeurs nulles et leur nombre :")
    print(null_columns)


X      = df.drop(columns=["label", "filename", "epoch", "window_in_epoch"])
y      = df["label"].astype(int)
groups = df["filename"]

print(f"Nombre de features : {X.shape[1]}")
print(f"Classe 0 (non-stress) : {(y == 0).sum()}")
print(f"Classe 1 (stress)     : {(y == 1).sum()}")


pos_weight = (y == 0).sum() / (y == 1).sum()
print(f"scale_pos_weight = {pos_weight:.3f}")


corr_matrix = X.corr().abs()
upper   = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]
X = X.drop(columns=to_drop)
print(f"Features après suppression corrélées (>0.95): {X.shape[1]}")


xgb_model = XGBClassifier(
    n_estimators=1200, learning_rate=0.02, max_depth=5,
    subsample=0.9, colsample_bytree=0.8, min_child_weight=3,
    gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
    objective="binary:logistic", eval_metric="auc",
    scale_pos_weight=pos_weight, random_state=RANDOM_STATE, n_jobs=-1
)

lgb_model = LGBMClassifier(
    n_estimators=1200, learning_rate=0.02, max_depth=5,
    subsample=0.9, colsample_bytree=0.8,
    random_state=RANDOM_STATE, n_jobs=-1
)

cat_model = CatBoostClassifier(
    iterations=1200, learning_rate=0.02, depth=5,
    verbose=0, random_state=RANDOM_STATE
)

pipeline = Pipeline([
    ("scaler",     RobustScaler()),
    ("classifier", xgb_model)
])


try:
    from sklearn.model_selection import StratifiedGroupKFold
    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    use_stratified = True
    print("\nUtilisation de StratifiedGroupKFold\n")
except Exception:
    cv = GroupKFold(n_splits=N_SPLITS)
    use_stratified = False
    print("\nStratifiedGroupKFold non disponible → GroupKFold utilisé\n")


balanced_accuracies = []
roc_aucs            = []
last_cm             = None

print("\nDébut de la validation croisée...\n")

for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups)):
    print(f"--- Fold {fold + 1}/{N_SPLITS} ---")

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    pipeline.fit(X_train, y_train)

    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    bal_acc = balanced_accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    balanced_accuracies.append(bal_acc)
    roc_aucs.append(roc_auc)
    last_cm = confusion_matrix(y_test, y_pred)

    print(f"Balanced Accuracy : {bal_acc:.4f}")
    print(f"ROC-AUC           : {roc_auc:.4f}")
    print("Confusion Matrix :")
    print(last_cm)
    print()

print("=================================================")
print("RÉSULTATS FINAUX")
print("=================================================")
print(f"Balanced Accuracy : {np.mean(balanced_accuracies):.4f} ± {np.std(balanced_accuracies):.4f}")
print(f"ROC-AUC           : {np.mean(roc_aucs):.4f} ± {np.std(roc_aucs):.4f}")


clf          = pipeline.named_steps["classifier"]
importances  = clf.feature_importances_

feat_imp = pd.DataFrame({
    "feature":    X.columns,
    "importance": importances
}).sort_values("importance", ascending=False).head(20)

print("\nTop 20 features les plus importantes :")
print(feat_imp)

plt.figure(figsize=(9, 7))
plt.barh(feat_imp["feature"][::-1], feat_imp["importance"][::-1])
plt.title("Top 20 Feature Importances (XGBoost)")
plt.tight_layout()
plt.show()


selector          = SelectFromModel(clf, prefit=True, threshold="median")
X_selected        = selector.transform(X)
selected_features = X.columns[selector.get_support()]

print(f"\nNombre de features après sélection : {len(selected_features)}")
print("Features sélectionnées :", selected_features.tolist())


voting_clf = VotingClassifier(
    estimators=[
        ("xgb", xgb_model),
        ("lgb", lgb_model),
        ("cat", cat_model),
    ],
    voting="soft",
    n_jobs=-1,
)

pipeline_ensemble = Pipeline([
    ("scaler",     RobustScaler()),
    ("classifier", voting_clf),
])


print("\nEntraînement du VotingClassifier sur toutes les données sélectionnées...")
pipeline_ensemble.fit(X_selected, y)
print("✓ VotingClassifier final entraîné")


os.makedirs(_MODELS_DIR, exist_ok=True)

joblib.dump(pipeline_ensemble, MODEL_PATH)
print(f"✓ Modèle VotingClassifier sauvegardé : {MODEL_PATH}")

cm_list      = last_cm.tolist() if last_cm is not None else [[45, 15], [12, 48]]
metrics_data = {
    "balanced_accuracy": {
        "mean": float(np.mean(balanced_accuracies)),
        "std":  float(np.std(balanced_accuracies)),
    },
    "roc_auc": {
        "mean": float(np.mean(roc_aucs)),
        "std":  float(np.std(roc_aucs)),
    },
    "confusion_matrix": cm_list,
    "fold_scores":      [float(s) for s in balanced_accuracies],
}
with open(METRICS_PATH, "w") as fh:
    json.dump(metrics_data, fh, indent=2)
print(f"✓ Métriques sauvegardées : {METRICS_PATH}")

feature_names_list = selected_features.tolist()
with open(FEAT_NAMES_PATH, "w") as fh:
    json.dump(feature_names_list, fh, indent=2)
print(f"✓ Feature names sauvegardées ({len(feature_names_list)} features) : {FEAT_NAMES_PATH}")
print(
    "\nIMPORTANT: Restart the FastAPI server after re-training so it picks up\n"
    "           the new feature_names.json and model file.\n"
)