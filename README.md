# NeuroStress Control Room

EEG-based stress classification dashboard with real-time signal processing,
3D brain visualisation, and an AI-powered chat assistant.

---

## Folder Structure

```
neuroctrl/
│
├── backend/                      ← FastAPI Python server
│   ├── __init__.py
│   └── main.py                   ← /preprocess  /predict  /metrics  /health
│
├── src/
│   ├── App.tsx                   ← Root React component + router
│   ├── main.tsx                  ← React DOM entry point
│   ├── index.css                 ← Tailwind + theme tokens
│   │
│   ├── components/
│   │   ├── Layout.tsx            ← Sidebar + main shell
│   │   ├── Dashboard.tsx         ← CV metrics, SAM dataset overview
│   │   ├── SignalAnalyzer.tsx    ← Upload .mat, view raw/cleaned signal
│   │   ├── ComparativeAnalysis.tsx ← Side-by-side raw vs cleaned
│   │   ├── ClassificationDashboard.tsx ← Stress prediction + features
│   │   ├── Brain3D.tsx           ← Three.js 3-D electrode map
│   │   ├── PipelineStepper.tsx   ← 4-step pipeline progress
│   │   └── NeuroAssistant.tsx    ← Gemini 2.0 Flash chat
│   │
│   ├── services/
│   │   └── api.ts                ← Axios wrappers for /api/*
│   │
│   ├── store/
│   │   └── useStore.ts           ← Zustand global state
│   │
│   └── lib/
│       ├── eeg-types.ts          ← FrequencyBands, ELECTRODE_POSITIONS
│       └── utils.ts              ← cn() Tailwind merge helper
│
├── features.py                   ← Time / freq / Hjorth / fractal / entropy features
├── extract_features.py           ← Standalone feature extraction script
├── dataset.py                    ← load_all_cleaned_with_features()
├── dataframe.py                  ← Build & save features DataFrame
├── classifier.py                 ← Train XGB+LGBM+CatBoost ensemble, save metrics.json
├── preprocessor.py               ← Single-file MNE preprocessing CLI
├── batch_preprocess.py           ← Batch preprocessing for raw_data/ folder
├── mapping_labels.py             ← Parse .mat filenames → subject/trial labels
├── visualize_bands.py            ← EEG spectral band visualisation
├── variables.py                  ← Shared path / constant definitions
│
├── server.ts                     ← Express dev server + /api proxy to Python
├── vite.config.ts                ← Vite build config
├── tsconfig.json
├── package.json
├── index.html
├── requirements.txt              ← Python dependencies
└── .env.example                  ← GEMINI_API_KEY placeholder
```

---

## Quick Start

### 1 — Python backend

```bash
# from project root
pip install -r requirements.txt

# run the API (port 8000)
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

### 2 — Node frontend

```bash
npm install

# copy .env.example → .env.local, fill in GEMINI_API_KEY
cp .env.example .env.local

# starts Express (port 3000) which:
#   • proxies /api/* → Python :8000
#   • serves the Vite React app
npm run dev
```

Open **http://localhost:3000**

---

## API Contract

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/preprocess` | Upload `.mat` → `{raw_signal, cleaned_signal, channel_names, sfreq}` |
| `POST` | `/api/predict` | `{signal, sfreq}` → `{prediction, probabilities, confidence, topFeatures, bandPowers}` |
| `GET` | `/api/metrics` | `{balanced_accuracy, roc_auc, confusion_matrix, fold_scores}` |
| `GET` | `/api/health` | `{status, model_available}` |

---

## Training Pipeline (offline)

```bash
# 1. Preprocess raw .mat files
python batch_preprocess.py

# 2. Build the features DataFrame
python dataframe.py

# 3. Train the ensemble + save models/metrics.json
python classifier.py
```

After step 3 the `/metrics` endpoint returns **real** cross-validation scores
and the `/predict` endpoint uses the trained VotingClassifier.

---

## Key Bug Fixes Applied

| File | Fix |
|------|-----|
| `backend/main.py` | **Created** — was entirely missing; all frontend API calls returned 502 |
| `backend/__init__.py` | **Created** — required for `python -m uvicorn backend.main:app` |
| `features.py` | Removed unused `import mne_features` + `import pywt` (caused `ImportError`) |
| `extract_features.py` | Fixed double-windowing bug; pre-framed data was being re-windowed inside `extract_all_features` |
| `classifier.py` | Added `json` import + `metrics.json` save after training |
| `NeuroAssistant.tsx` | Changed `"gemini-3-flash-preview"` → `"gemini-2.0-flash"` (model didn't exist) |
| `requirements.txt` | Added `catboost`, `fastapi`, `uvicorn[standard]`, `python-multipart` |
