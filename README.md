<div align="center">

# 🧠 NeuroStress Control Room

### EEG-Based Stress Detection Platform

**Automatic stress classification from EEG signals using the SAM 40 Dataset**

*Signal Processing Project — Academic Year 2025–2026*

[![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)](https://python.org)
[![React](https://img.shields.io/badge/React-19-61DAFB?logo=react)](https://react.dev)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![Dataset](https://img.shields.io/badge/Dataset-SAM%2040-orange)](https://www.sciencedirect.com/science/article/pii/S2352340921010465)

</div>

---

## 📋 Table of Contents

1. [Project Overview](#-project-overview)
2. [SAM 40 Dataset](#-sam-40-dataset)
   - [Dataset Description](#dataset-description)
   - [EEG Recording Setup](#eeg-recording-setup)
   - [Cognitive Stress Tasks](#cognitive-stress-tasks)
   - [Stress Labeling Protocol](#stress-labeling-protocol)
3. [Theoretical Background](#-theoretical-background)
   - [EEG Signal Properties](#eeg-signal-properties)
   - [EEG Frequency Bands](#eeg-frequency-bands)
4. [Signal Processing Pipeline](#-signal-processing-pipeline)
   - [Noise & Artifact Removal](#1-noise--artifact-removal)
   - [Normalization](#2-normalization)
   - [Segmentation & Windowing](#3-segmentation--windowing)
5. [Feature Extraction — Mathematical Formulation](#-feature-extraction--mathematical-formulation)
   - [Time-Domain Features](#a-time-domain-features)
   - [Frequency-Domain Features](#b-frequency-domain-features-welch-psd)
   - [Hjorth Parameters](#c-hjorth-parameters)
   - [Fractal Dimension Features](#d-fractal-dimension-features)
   - [Entropy Features](#e-entropy-features)
   - [Wavelet Energy Features](#f-wavelet-energy-features)
   - [Inter-Hemispheric Asymmetry](#g-inter-hemispheric-asymmetry)
6. [Machine Learning Pipeline](#-machine-learning-pipeline)
7. [Platform Architecture & Features](#-platform-architecture--features)
8. [Project Structure](#-project-structure)
9. [Installation & Quick Start](#-installation--quick-start)
10. [API Reference](#-api-reference)
11. [Results & Performance](#-results--performance)
12. [References](#-references)

---

## 🎯 Project Overview

**NeuroStress Control Room** is an end-to-end brain-computer interface (BCI) platform that automatically detects mental stress from raw EEG signals. It combines advanced neurophysiological signal processing with an ensemble machine learning classifier and a real-time 3D visualization dashboard.

The platform was built on top of the **SAM 40 dataset** — a publicly available, open-access EEG database specifically designed for mental stress research. The full processing chain covers:

- Raw `.mat` EEG ingestion and preprocessing (notch filtering, bandpass, ASR)
- Multi-domain feature extraction (time, frequency, Hjorth, fractal, entropy, wavelet)
- Ensemble classification (XGBoost + LightGBM + CatBoost VotingClassifier)
- Interactive React dashboard with 3D electrode visualization

---

## 📊 SAM 40 Dataset

### Dataset Description

The **SAM 40** (Stress-Affect-Meditation 40) dataset was published by the Department of Information Technology, Gauhati University, India, and is available open-access on ScienceDirect:

> **Citation:** Sharma, M., et al. (2022). *SAM 40: Dataset of 40 Subject EEG Recordings to Monitor the Stress, Affect and Meditation*. Data in Brief, Elsevier. DOI: [10.1016/j.dib.2021.107805](https://www.sciencedirect.com/science/article/pii/S2352340921010465)

| Property | Value |
|---|---|
| **Participants** | 40 healthy subjects |
| **Gender** | 26 males, 14 females |
| **Mean Age** | 21.5 years |
| **EEG Channels** | 32 channels |
| **Sampling Rate** | 128 Hz |
| **Trial Duration** | 25 seconds per trial |
| **Trials per Condition** | 3 trials |
| **Conditions** | Stress (3 tasks) + Relax |
| **Total Recordings** | 480 recordings |

### EEG Recording Setup

EEG data were recorded using the **Emotiv Epoch Flex gel kit** with 32 channels following the international 10–20 system. Standard electrode positions include: `Fp1, Fp2, F3, F4, C3, C4, P3, P4, O1, O2, F7, F8, T7, T8, P7, P8`, and others. CMS and DRL reference electrodes were placed on the left and right mastoids.

![EEG Recording Setup](./assets/1-s2.0-S2352340921010465-gr4_lrg.jpg)

### Cognitive Stress Tasks

The dataset uses three validated psychophysiological stress induction paradigms, each lasting **25 seconds** with **3 trials per participant**:

---

#### 1. 🎨 Stroop Color-Word Test (SCWT)

<div align="center">

*A classic cognitive interference paradigm*

</div>

The Stroop Color-Word Test measures **cognitive interference** and **executive function** under stress. Participants are shown color words (e.g., "RED", "BLUE") printed in a **different ink color** (e.g., the word "RED" printed in blue ink). Subjects must name the **ink color**, not read the word.

**Why it induces stress:** The automatic reading response conflicts with the required color-naming response, creating cognitive overload and mental tension.

**EEG signatures:**
- ↑ Frontal beta power (13–30 Hz) — cognitive effort
- ↑ Theta power (4–8 Hz) at Fz — working memory load
- ↓ Alpha power (8–13 Hz) — reduced relaxation

---

#### 2. 🪞 Mirror Image Recognition Task

<div align="center">

*A spatial cognition and visual processing test*

</div>

Participants view pairs of mirror images and must decide whether they are **symmetric or asymmetric** under time pressure. This task primarily engages parietal and occipital regions involved in spatial processing.

**Why it induces stress:** Time constraints combined with spatial reasoning create mental strain and decision pressure.

**EEG signatures:**
- ↑ Parietal theta power — spatial working memory
- ↑ Occipital beta activity — enhanced visual processing
- Bilateral parietal activation asymmetry

---

#### 3. ➕ Arithmetic Problem Solving Task

<div align="center">

*Mental calculation under time pressure*

</div>

Subjects mentally solve arithmetic problems (addition, subtraction, multiplication) and indicate correctness using a **thumbs up or thumbs down** gesture. Problems increase in difficulty across trials.

**Why it induces stress:** Cognitive demand + performance pressure + time constraints activate the sympathetic nervous system and induce measurable mental stress.

**EEG signatures:**
- ↑ Frontal beta and gamma power — high cognitive load
- ↑ Theta at Fz and FCz — numerical working memory
- ↓ Alpha at parietal regions — active processing

---

### Stress Labeling Protocol

After each trial, participants rate their **subjective stress level** using a self-assessment scale (1–10). The binary classification label is derived from this score:

```
label = 1 (STRESS)     if score > 5
label = 0 (NON-STRESS) if score ≤ 5
```

**Relax condition:** Participants also performed a relaxation baseline (eyes closed, resting). These recordings are assigned a fixed score of 5 → **label = 0** (non-stress baseline).

| Condition | Score Range | Label |
|---|---|---|
| Stroop SCWT | 1–10 (subjective) | score > 5 → 1 |
| Arithmetic | 1–10 (subjective) | score > 5 → 1 |
| Mirror Image | 1–10 (subjective) | score > 5 → 1 |
| Relax (baseline) | fixed = 5 | 0 |

---

## 🔬 Theoretical Background

### EEG Signal Properties

EEG signals are non-stationary, stochastic electrical recordings of brain neural activity measured at the scalp surface. Key properties:

| Property | Value |
|---|---|
| Amplitude range | 1 – 100 µV |
| Frequency range | 0.5 – 100 Hz |
| Nature | Non-stationary, stochastic |
| Noise sensitivity | Very high (power line, muscle, eye, cardiac) |
| Spatial resolution | Low (volume conduction effect) |

### EEG Frequency Bands

The EEG spectrum is partitioned into physiologically meaningful frequency bands, each associated with distinct mental states:

| Band | Symbol | Frequency | Mental State |
|---|---|---|---|
| **Delta** | δ | 0.5 – 4 Hz | Deep sleep, unconscious |
| **Theta** | θ | 4 – 8 Hz | Working memory, drowsiness |
| **Alpha** | α | 8 – 13 Hz | Relaxation, idle state |
| **Beta** | β | 13 – 30 Hz | **Stress, anxiety, mental tension** |
| **Gamma** | γ | 30 – 45 Hz | Conscious awareness, high cognition |

**Key stress biomarkers:**
- **β/α ratio** → elevated during cognitive stress (frontal electrodes)
- **θ/α ratio** → elevated during working memory overload
- **(β + γ) / (δ + θ + α)** → global arousal index

---

## ⚙️ Signal Processing Pipeline

### 1. Noise & Artifact Removal

EEG signals are contaminated by multiple noise sources. A cascaded filtering approach is applied:

| Noise Type | Source | Filter Applied |
|---|---|---|
| Power Line Interference | 50/60 Hz electrical | Notch Filter |
| Eye blinks / slow drift | EOG, DC drift | High-Pass Filter |
| Muscle artifacts | EMG contamination | Low-Pass Filter |
| Electrode artifacts | Bad channels | ASR + Interpolation |

**Mathematical formulations:**

**Notch Filter** — eliminates 50/60 Hz power line interference:

```
y[n] = x[n] − 2·cos(ω₀)·x[n−1] + x[n−2]
```

where ω₀ = 2π·f₀/fₛ, f₀ = 50 Hz, fₛ = 256 Hz

**High-Pass Filter** — removes slow drifts and eye blinks (< 1 Hz):

```
y[n] = α · (y[n−1] + x[n] − x[n−1])
```

where α = τ / (τ + Tₛ), τ = RC time constant

**Low-Pass Filter** — removes EMG artifacts (> 40 Hz):

```
y[n] = α·x[n] + (1 − α)·y[n−1]
```

where α = 2πf_c·Tₛ / (1 + 2πf_c·Tₛ), f_c = 40 Hz

**Artifact Subspace Reconstruction (ASR):**
ASR is applied to detect and remove non-stationary burst artifacts that cannot be removed by filtering alone. It uses PCA-based decomposition on clean baseline windows to set a threshold (cutoff = 15 standard deviations) and reconstructs contaminated segments.

**Bad Channel Detection & Interpolation:**

A channel is marked as "bad" if its z-score of standard deviation exceeds a threshold:

```
z_i = (σᵢ − μ_σ) / std_σ

Bad if: |z_i| > 5  OR  σᵢ < 0.1 · μ_σ
```

Bad channels are interpolated using spherical spline interpolation.

**Average Reference:**
After artifact removal, the EEG is re-referenced to the common average:

```
x'_i[n] = x_i[n] − (1/N) · Σⱼ xⱼ[n]
```

### 2. Normalization

Z-score normalization is applied to make features comparable across subjects and recording sessions:

$$X_{\text{norm}} = \frac{X - \mu}{\sigma}$$

where μ is the channel mean and σ is the channel standard deviation computed over the recording window.

The alternative Min-Max scaling (not used here) would be:

$$X_{\text{norm}} = \frac{X - X_{\min}}{X_{\max} - X_{\min}}$$

**Why Z-score?** It preserves the signal's relative dynamics and is robust to outliers introduced by residual artifacts.

### 3. Segmentation & Windowing

**Nyquist-Shannon Theorem:**
The sampling frequency must satisfy:

```
fₛ > 2 · f_max
```

With fₛ = 256 Hz (2× upsampled from the original 128 Hz), the sampling period is:

```
Tₛ = 1/fₛ = 1/256 ≈ 3.906 ms
```

**Hanning Window:**
Each EEG epoch is segmented into overlapping windows using the Hanning (raised cosine) window function to minimize spectral leakage:

$$w[n] = 0.5 \left(1 - \cos\left(\frac{2\pi n}{N-1}\right)\right), \quad 0 \leq n \leq N-1$$

**Windowing parameters:**

| Parameter | Value | Rationale |
|---|---|---|
| Window length | 1.0 s = 256 samples | Frequency resolution = 1 Hz |
| Overlap | 50% = 128 samples | Recover edge-attenuated information |
| Step size | 128 samples | Balance between time & frequency resolution |

**Why Hanning windows?** A rectangular window creates sharp discontinuities at segment edges, causing spectral leakage (energy spreading to adjacent frequency bins). The Hanning window tapers smoothly to zero at both edges, concentrating energy in the true frequency bins.

---

## 📐 Feature Extraction — Mathematical Formulation

For each 1-second window of shape `(n_channels, 256_samples)`, the following feature groups are computed **per channel** and concatenated into a single feature vector.

**Total features per channel:** 37 (from 10 groups)
**Asymmetry block:** min(5, n_ch//2) × 5 bands

### A. Time-Domain Features

#### Variance (σ²)
Measures signal power — high variance indicates strong neural activity:

$$\sigma^2 = \frac{1}{N} \sum_{i=1}^{N} (x_i - \mu)^2$$

#### Root Mean Square (RMS)
Reflects overall signal amplitude; higher RMS corresponds to stronger neuronal firing:

$$\text{RMS} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} x_i^2}$$

#### Peak-to-Peak Amplitude
Captures extreme amplitude fluctuations, sensitive to residual artifacts:

$$P_{ptp} = \max(x) - \min(x)$$

#### Skewness (γ₁)
Measures distribution asymmetry; non-zero skewness indicates irregular neural patterns:

$$\gamma_1 = \frac{\mathbb{E}[(X - \mu)^3]}{\sigma^3}$$

#### Kurtosis (γ₂)
Measures "peakedness" of signal distribution; high kurtosis indicates sharp spikes or bursts:

$$\gamma_2 = \frac{\mathbb{E}[(X - \mu)^4]}{\sigma^4} - 3$$

---

### B. Frequency-Domain Features (Welch PSD)

The **Discrete Fourier Transform (DFT)** decomposes the signal into its frequency components:

$$X[k] = \sum_{n=0}^{N-1} x[n] \cdot e^{-j\frac{2\pi}{N}kn}$$

The **Welch method** computes a smoothed Power Spectral Density (PSD) by averaging multiple overlapping periodograms:

**Step 1 — Per-segment power spectrum:**

$$P_m[k] = \frac{1}{L \cdot U} \left| \text{FFT}\left(x_m^w[n]\right)[k] \right|^2$$

**Step 2 — Window normalization factor:**

$$U = \frac{1}{L} \sum_{n=0}^{L-1} |w[n]|^2$$

**Step 3 — Average across M overlapping segments (Welch PSD):**

$$P_{\text{Welch}}[k] = \frac{1}{M} \sum_{m=0}^{M-1} P_m[k]$$

where L = 256 (window length), M = number of overlapping segments, w[n] = Hanning window.

**Band Power (absolute):**
The absolute power in each frequency band is estimated by integrating the PSD over the band using the trapezoidal rule:

$$P_{\text{band}} = \int_{f_{\text{low}}}^{f_{\text{high}}} S(f) \, df \approx \sum_{f_k \in [f_{\text{low}}, f_{\text{high}}]} S(f_k) \cdot \Delta f$$

**Relative Band Power:**

$$P_{\text{rel}} = \frac{P_{\text{band}}}{\sum_{\text{all bands}} P_{\text{band}}}$$

**Clinical Stress Ratios:**

| Ratio | Formula | Stress Interpretation |
|---|---|---|
| Beta/Alpha | β/α | ↑ → cognitive arousal/stress |
| Theta/Alpha | θ/α | ↑ → working memory overload |
| Arousal Index | (β+γ)/(δ+θ+α) | ↑ → global mental activation |
| Relaxation Index | α/(θ+β) | ↓ → reduced relaxation |

**Spectral Edge Frequency (SEF):**

The frequency f_p% below which p% of total spectral power is contained:

$$\text{SEF}_{p} = f \text{ such that } \frac{\int_0^f S(f')df'}{\int_0^{f_{\max}} S(f')df'} = \frac{p}{100}$$

We compute SEF₉₀ and SEF₉₅.

---

### C. Hjorth Parameters

Introduced by Hjorth (1970), these three parameters characterize EEG complexity in the time domain using signal derivatives:

**Activity** — signal variance (power):

$$H_{\text{act}} = \sigma^2(x)$$

**Mobility** — mean frequency approximation:

$$H_{\text{mob}} = \sqrt{\frac{\sigma^2(x')}{\sigma^2(x)}}$$

**Complexity** — deviation from a pure sinusoid:

$$H_{\text{comp}} = \frac{H_{\text{mob}}(x')}{H_{\text{mob}}(x)}$$

where x' = dx/dn (first derivative), x'' = d²x/dn² (second derivative).

Higher complexity values during stress indicate more irregular, less predictable brain dynamics.

---

### D. Fractal Dimension Features

Fractal dimension quantifies the self-similarity and complexity of EEG signals — stressed EEG tends to have higher fractal dimension.

**Higuchi Fractal Dimension (HFD):**

For a time series x(1), x(2), ..., x(N) and interval k:

$$L_m(k) = \frac{N-1}{\lfloor (N-m)/k \rfloor \cdot k^2} \sum_{i=1}^{\lfloor (N-m)/k \rfloor} |x(m+ik) - x(m+(i-1)k)|$$

The HFD is the slope of log(L(k)) vs log(1/k):

$$\text{HFD} = -\frac{d\left[\log L(k)\right]}{d\left[\log(1/k)\right]}$$

**Katz Fractal Dimension:**

$$\text{KFD} = \frac{\log(N-1)}{\log(d/L) + \log(N-1)}$$

where L = total path length of the waveform, d = maximum Euclidean distance from the first point.

---

### E. Entropy Features

Entropy measures signal unpredictability — stressed EEG typically shows higher entropy.

**Approximate Entropy (ApEn):**

$$\text{ApEn}(m, r) = \phi^m(r) - \phi^{m+1}(r)$$

where $\phi^m(r) = \frac{1}{N-m+1} \sum_{i=1}^{N-m+1} \ln C_i^m(r)$, r = 0.2·σ (tolerance), m = 2 (embedding dimension).

**Sample Entropy (SampEn):**
Estimated via correlation of adjacent samples (computationally efficient approximation):

$$\text{SampEn} \approx -\ln\left|\rho(x_{t-1}, x_t)\right|$$

where ρ is the Pearson correlation coefficient between lagged signals.

**Spectral Entropy:**

$$H_{\text{spec}} = -\sum_{f} \hat{S}(f) \cdot \ln \hat{S}(f)$$

where $\hat{S}(f) = S(f) / \sum_f S(f)$ is the normalized PSD.

**Singular Value Decomposition (SVD) Entropy:**

$$H_{\text{SVD}} = -\sum_{i} \hat{\sigma}_i \cdot \ln \hat{\sigma}_i$$

where σ̂ᵢ = σᵢ / Σσᵢ are normalized singular values of the delay-embedding matrix.

---

### F. Wavelet Energy Features

A Haar-like multi-resolution decomposition is applied to extract energy at different temporal scales:

For each decomposition level ℓ = 1, ..., 4:

$$E_\ell = \frac{1}{N_\ell} \sum_{n} d_\ell[n]^2$$

where d_ℓ[n] are the detail coefficients at level ℓ, and E₅ is the energy of the final approximation.

This provides a 5-dimensional representation of signal energy distribution across temporal scales (high-frequency → low-frequency).

---

### G. Inter-Hemispheric Asymmetry

Asymmetry between homologous left–right electrode pairs is a key stress biomarker:

$$A_{\text{asym}}(b, p) = \ln\left(\frac{P_{\text{left}}(b, p) + \varepsilon}{P_{\text{right}}(b, p) + \varepsilon}\right)$$

where P(b, p) is the absolute band power in band b for electrode pair p, and ε = 1e-10 prevents log(0).

Computed for up to 5 hemisphere pairs × 5 frequency bands = up to 25 asymmetry features.

---

## 🤖 Machine Learning Pipeline

### Feature Engineering Summary

| Group | Features/Channel | Total (32 channels) |
|---|---|---|
| Time Domain | 5 | 160 |
| Absolute Band Power | 5 | 160 |
| Relative Band Power | 5 | 160 |
| Spectral Ratios | 4 | 128 |
| Spectral Edge Freq. | 2 | 64 |
| Zero-Crossing + Line Length | 2 | 64 |
| Hjorth Parameters | 3 | 96 |
| Fractal Dimension | 2 | 64 |
| Entropy | 4 | 128 |
| Wavelet Energy | 5 | 160 |
| **Asymmetry** | — | 25 |
| **TOTAL** | **37/ch** | **~1,209** |

### Preprocessing

1. **Correlated feature removal** — Features with Pearson correlation > 0.95 are dropped
2. **RobustScaler** — Scales features using median and IQR (robust to outliers)
3. **Feature selection** — `SelectFromModel` with XGBoost, threshold = median importance

### Ensemble Classifier

A soft-voting ensemble of three gradient boosting classifiers:

```
VotingClassifier(
    estimators=[
        ('xgb',  XGBClassifier(n_estimators=1200, lr=0.02, max_depth=5)),
        ('lgb',  LGBMClassifier(n_estimators=1200, lr=0.02, max_depth=5)),
        ('cat',  CatBoostClassifier(iterations=1200, lr=0.02, depth=5))
    ],
    voting='soft'
)
```

**Class imbalance handling:**
```
scale_pos_weight = N_negative / N_positive
```

**Cross-validation:** `StratifiedGroupKFold` (5 folds) with subject-level grouping to prevent data leakage across participants.

**Prediction rule:**

$$\hat{y} = \begin{cases} 1 & \text{if } P(\text{stress}) \geq \theta_{\text{optimal}} \\ 0 & \text{otherwise} \end{cases}$$

where θ_optimal is tuned per training fold (default = 0.5).

---

## 🖥️ Platform Architecture & Features

### Backend (FastAPI — Python)

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Server health + model status |
| `/preprocess` | POST | Upload `.mat` → filtered + cleaned EEG |
| `/extract-features` | POST | Compute all feature groups per channel |
| `/predict` | POST | Run stress classification |
| `/metrics` | GET | Cross-validation performance results |

### Frontend (React 19 + Three.js)

| Tab | Feature |
|---|---|
| **Dashboard** | CV metrics, SAM dataset overview, test illustrations |
| **Signal Analyzer** | Upload `.mat`, view raw vs cleaned EEG with Plotly |
| **Compare** | Side-by-side overlay with noise reduction metrics |
| **Feature Explorer** | Radar charts, band power heatmap, temporal evolution |
| **Classification** | Stress verdict, probability gauge, β/α per channel |
| **3D Brain Map** | Live Three.js electrode map driven by real band powers |
| **Pipeline** | Step-by-step processing progress stepper |

### Technology Stack

| Layer | Technologies |
|---|---|
| **Signal Processing** | MNE-Python, SciPy, NumPy, ASRpy |
| **Machine Learning** | XGBoost, LightGBM, CatBoost, scikit-learn |
| **Backend** | FastAPI, Uvicorn, Python 3.11 |
| **Frontend** | React 19, TypeScript, Three.js, Plotly, Recharts |
| **State Management** | Zustand |
| **Styling** | Tailwind CSS v4 |
| **Build** | Vite 6, Node.js |

---

## 📁 Project Structure

```
neuroctrl/
│
├── backend/                      ← FastAPI Python server
│   ├── __init__.py
│   └── main.py                   ← /preprocess /predict /metrics /health
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
│   │   ├── ComparativeAnalysis.tsx
│   │   ├── ClassificationDashboard.tsx
│   │   ├── Brain3D.tsx           ← Three.js 3-D electrode map
│   │   ├── PipelineStepper.tsx
│   │   └── FeatureVisualizer.tsx
│   │
│   ├── services/api.ts           ← Axios wrappers for /api/*
│   ├── store/useStore.ts         ← Zustand global state
│   └── lib/
│       ├── eeg-types.ts          ← FrequencyBands, ELECTRODE_POSITIONS
│       └── utils.ts
│
├── features.py                   ← All feature extraction functions
├── extract_features.py           ← Standalone feature extraction script
├── dataset.py                    ← load_all_cleaned_with_features()
├── dataframe.py                  ← Build & save features DataFrame
├── classifier.py                 ← Train XGB+LGBM+CatBoost ensemble
├── preprocessor.py               ← Single-file MNE preprocessing CLI
├── batch_preprocess.py           ← Batch preprocessing
├── mapping_labels.py             ← Parse .mat filenames → labels
├── visualize_bands.py            ← Spectral band visualization
├── variables.py                  ← Shared path / constant definitions
│
├── server.ts                     ← Express dev server + /api proxy
├── vite.config.ts
├── tsconfig.json
├── package.json
├── requirements.txt              ← Python dependencies
└── .env.example
```

---

## 🚀 Installation & Quick Start

### Prerequisites

- **Python** 3.11+
- **Node.js** 20+
- **npm** 8+
- **Git**

---

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/neurostress-control-room.git
cd neurostress-control-room
```

---

### 2. Python Backend Setup

```bash
# Create and activate virtual environment
python -m venv env311
source env311/bin/activate          # Linux/macOS
# env311\Scripts\activate           # Windows

# Install Python dependencies
pip install -r requirements.txt
```

**requirements.txt includes:**
```
numpy==1.26.4
scipy==1.11.4
pandas==2.2.2
scikit-learn==1.4.2
torch==2.2.2
matplotlib==3.8.4
xgboost==2.0.3
lightgbm==4.3.0
catboost==1.2.5
joblib==1.3.2
fastapi==0.110.0
uvicorn==0.29.0
python-multipart==0.0.9
mne==1.7.0
pywavelets==1.6.0
Pillow<11
```

---

### 3. Node.js Frontend Setup

```bash
npm install
```

---

### 4. Environment Configuration

```bash
cp .env.example .env.local
```

Edit `.env.local`:

```env
# (Optional) Path to trained .joblib model
NEUROSTRESS_MODEL_PATH=./models/xgb_stress_classifier_ensemble.joblib

# (Optional) Gemini API Key for NeuroAssistant chat
GEMINI_API_KEY=your_gemini_api_key_here

APP_URL=http://localhost:3000
```

---

### 5. (Optional) Train the ML Model

If you have the SAM 40 dataset downloaded, run the full training pipeline:

```bash
# Step 1: Configure data paths
nano variables.py        # Set DIR_RAW, DIR_CLEANED, LABELS_PATH

# Step 2: Preprocess all raw .mat files
python batch_preprocess.py

# Step 3: Build the feature DataFrame
python dataframe.py

# Step 4: Train the ensemble classifier
python classifier.py
```

This generates:
- `models/xgb_stress_classifier_ensemble.joblib` — trained model
- `models/metrics.json` — cross-validation metrics

---

### 6. Start the Application

**Development mode (Frontend + Backend together):**

```bash
npm run dev
```

This starts:
- Express server on `http://localhost:3000` (frontend)
- FastAPI backend on `http://127.0.0.1:8000` (via proxy)

**Or run them separately:**

```bash
# Terminal 1 — Python API
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload

# Terminal 2 — Vite frontend
npx vite
```

Open **http://localhost:3000** in your browser.

---

### 7. Verify Installation

```bash
# Check backend health
python check_backend.py

# Or via HTTP
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "ok",
  "model_available": true,
  "model_source": "trained_model",
  "version": "3.2.0"
}
```

---

### 8. Using the Platform

1. Navigate to **Signal Analyzer** tab
2. Click **Import .MAT File** and upload an EEG file from the SAM 40 dataset
3. The backend preprocesses the signal automatically (filter → ASR → re-reference)
4. Go to **Feature Explorer** → click **Extract Features**
5. Go to **Classification** → click **Run Inference**
6. View the 3D brain map in **3D Brain Map** tab

---

## 📡 API Reference

### POST `/preprocess`

Upload a raw `.mat` EEG file for preprocessing.

**Request:** `multipart/form-data` with field `file`

**Response:**
```json
{
  "raw_signal": [[...], ...],
  "cleaned_signal": [[...], ...],
  "channel_names": ["EEG1", "EEG2", ...],
  "sfreq": 256.0,
  "stats": {
    "raw_std_uv": 45.3,
    "clean_std_uv": 28.1,
    "noise_reduction_pct": 37.9
  }
}
```

### POST `/predict`

```json
// Request
{
  "signal": [[ch1_samples...], [ch2_samples...], ...],
  "sfreq": 256.0
}

// Response
{
  "prediction": 1,
  "probabilities": {"stress": 0.78, "non_stress": 0.22},
  "confidence": 0.84,
  "topFeatures": [{"name": "ch1_beta_power", "importance": 0.12}, ...],
  "bandPowers": [{"delta": 1.2}, {"theta": 2.8}, ...],
  "bandPowersPerCh": [{"channel": "EEG1", "delta": 1.1, "beta": 3.2, ...}],
  "model_source": "trained_model"
}
```

### GET `/metrics`

```json
{
  "balanced_accuracy": {"mean": 0.6347, "std": 0.0304},
  "roc_auc": {"mean": 0.6721, "std": 0.0412},
  "confusion_matrix": [[45, 15], [12, 48]],
  "fold_scores": [0.61, 0.65, 0.62, 0.66, 0.63]
}
```

---

## 📈 Results & Performance

| Metric | Mean | Std Dev |
|---|---|---|
| **Balanced Accuracy** | 63.47% | ±3.04% |
| **ROC-AUC** | 67.21% | ±4.12% |

**Cross-validation:** 5-fold `StratifiedGroupKFold` (subject-level split)

**Note:** The moderate performance reflects the inherent difficulty of EEG-based stress detection — EEG is subject to high inter-individual variability, and stress is a complex, subjective state. The results are consistent with the state-of-the-art on this dataset.

**Areas for improvement:**
- Subject-adaptive (personalized) models
- Deep learning approaches (EEGNet, CNN-LSTM)
- Additional physiological modalities (GSR, HRV, facial EMG)
- Longer recording segments for richer feature extraction

---

## 📚 References

1. **SAM 40 Dataset:**
   Sharma, M., Achuth, P. V., Deb, D., Tiwari, A. K., & Pachori, R. B. (2022). *SAM 40: Dataset of 40 Subject EEG Recordings to Monitor the Stress, Affect and Meditation*. Data in Brief, 40, 107805. https://doi.org/10.1016/j.dib.2021.107805

2. **Stroop Effect:**
   Stroop, J. R. (1935). *Studies of interference in serial verbal reactions*. Journal of Experimental Psychology, 18(6), 643–662.

3. **Hjorth Parameters:**
   Hjorth, B. (1970). *EEG analysis based on time domain properties*. Electroencephalography and Clinical Neurophysiology, 29(3), 306–310.

4. **Welch Method:**
   Welch, P. D. (1967). *The use of fast Fourier transform for the estimation of power spectra*. IEEE Transactions on Audio and Electroacoustics, 15(2), 70–73.

5. **Higuchi Fractal Dimension:**
   Higuchi, T. (1988). *Approach to an irregular time series on the basis of the fractal theory*. Physica D: Nonlinear Phenomena, 31(2), 277–283.

6. **ASR Algorithm:**
   Chang, C. Y., et al. (2020). *Evaluation of artifact subspace reconstruction for automatic artifact components removal in multi-channel EEG recordings*. IEEE Transactions on Biomedical Engineering, 67(4), 1114–1121.

7. **EEG Stress Review:**
   Sharma, N., & Gedeon, T. (2012). *Objective measures, sensors and computational techniques for stress recognition and classification: A survey*. Computer Methods and Programs in Biomedicine, 108(3), 1287–1301.

8. **MNE-Python:**
   Gramfort, A., et al. (2013). *MEG and EEG data analysis with MNE-Python*. Frontiers in Neuroscience, 7, 267.

---

<div align="center">

**Made with 🧠 by the NeuroStress Team**

*Signal Processing Project — Academic Year 2025–2026*

[Dataset](https://www.sciencedirect.com/science/article/pii/S2352340921010465) · [FastAPI Docs](http://localhost:8000/docs) · [Issues](https://github.com/your-username/neurostress-control-room/issues)

</div>