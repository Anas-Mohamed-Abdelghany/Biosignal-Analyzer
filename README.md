# ⚡ SignalViewer — Multi-Domain Signal Analysis Platform

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)
![React](https://img.shields.io/badge/React-18.2%2B-61DAFB.svg)
![Vite](https://img.shields.io/badge/Vite-4.0%2B-646CFF.svg)
![TailwindCSS](https://img.shields.io/badge/Tailwind_CSS-3.0%2B-38B2AC.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E.svg)

**A full-stack platform for high-performance processing, visualization, and AI-powered analysis of multi-domain signal data.**

<!-- FIGURE: Landing Page -->
<!-- ![Landing Page](docs/images/landing.png) -->
> 📸 *Screenshot of the SignalViewer landing page showing all module cards: `docs/images/landing.png`*

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [System Architecture](#-system-architecture)
- [Tech Stack](#-tech-stack)
- [Installation & Deployment](#-installation--deployment)
- [API Reference](#-api-reference)
- [Project Structure](#-project-structure)
- [Module Documentation](#-module-documentation)
  - [Medical — ECG](#-medical-signal-viewer--ecg-analysis)
  - [EEG](#-eeg-signal-viewer--neurological-classification)
  - [Acoustic](#-acoustic-signal-viewer--doppler--drone-detection)
  - [Finance](#-finance-signal-viewer--market-analysis--forecasting)
  - [Microbiome](#-microbiome-signal-viewer--ibd-classification)
- [Complete Screenshot Index](#-complete-screenshot-index--all-modules)

---

## 🌐 Overview

SignalViewer is an enterprise-grade, full-stack platform for the interactive exploration and AI-assisted analysis of signals across five scientific and financial domains. It combines a high-performance asynchronous Python backend with a reactive modern web interface, enabling real-time visualization and machine learning inference directly in the browser.

### Domain Coverage

| Module | Signal Type | AI Task |
|---|---|---|
| 🫀 Medical — ECG | Electrocardiogram (12–20 lead) | Disease classification (5 classes) |
| 🧠 Medical — EEG | 19-channel EEG | Neurological classification (4 classes) |
| 🔊 Acoustic | Audio (WAV, MP3, OGG, FLAC) | Doppler velocity estimation + drone detection |
| 📈 Finance | OHLCV market data | GRU price forecasting |
| 🧬 Microbiome | Longitudinal microbiome CSV | IBD classification (3 classes) |

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        BROWSER (React + Vite)                    │
│                                                                  │
│   Landing ──► Medical ──► Acoustic ──► Finance ──► Microbiome   │
│                                                                  │
│   Plotly.js charts  │  File uploads  │  Real-time playback      │
└──────────────────────────────┬───────────────────────────────────┘
                               │ HTTP REST  (JSON)
                               │ http://localhost:8000
┌──────────────────────────────▼───────────────────────────────────┐
│                        FASTAPI BACKEND                           │
│                                                                  │
│  routes/               services/             models/             │
│  ├─ medical_routes     ├─ medical_service    ├─ ECG CNN          │
│  ├─ eeg_routes         ├─ eeg_service        ├─ ECG RandomForest │
│  ├─ acoustic_routes    ├─ acoustic_service   ├─ EEG CNN          │
│  ├─ finance_routes     ├─ finance_service    ├─ EEG SVM          │
│  └─ bio_routes         └─ bio_service        ├─ Finance GRU (×3) │
│                                              ├─ IBD GRU          │
│  uploads/  (temp — auto-deleted after use)   └─ HMP2 ref CSV    │
└──────────────────────────────────────────────────────────────────┘
```

### Request Lifecycle

```
User uploads file
      │
      ▼
React (FileUpload component)
  POST multipart/form-data
      │
      ▼
FastAPI route
  → validate extension
  → save to uploads/
  → call service.analyze()
      │
      ▼
Service layer
  → lazy-load model (singleton)
  → preprocess signal
  → run inference
  → build response dict
      │
      ▼
Route returns JSONResponse
  → delete temp file (finally block)
      │
      ▼
React renders charts + result cards
```

### Controller–Service Pattern

Every domain enforces a strict two-layer backend structure:

- **Route layer** (`routes/`) — HTTP only: validate input, call service, return JSON, clean up uploads
- **Service layer** (`services/`) — all logic: model loading, preprocessing, feature extraction, inference

This ensures services can be tested independently without running the HTTP server.

---

## 🛠️ Tech Stack

### Backend

| Library | Purpose |
|---|---|
| **FastAPI** | Async REST API framework, auto OpenAPI docs |
| **Uvicorn** | ASGI production server |
| **TensorFlow / Keras** | CNN (ECG, EEG), GRU (Finance, IBD) inference |
| **scikit-learn** | SVM, RandomForest, StandardScaler, LabelEncoder |
| **NumPy** | Array operations, sliding windows, padding |
| **Pandas** | CSV parsing, per-patient data grouping |
| **SciPy** | Skewness, kurtosis for EEG feature extraction |
| **Librosa** | Audio loading, STFT, spectral feature extraction |
| **Joblib** | `.pkl` model serialization / deserialization |
| **Python-multipart** | File upload handling in FastAPI |

### Frontend

| Library | Purpose |
|---|---|
| **React 18** | Component-based UI framework |
| **Vite 4** | Build tool and HMR dev server |
| **Tailwind CSS 3** | Utility-first styling |
| **Plotly.js** | Interactive charts, heatmaps, polar plots, spectrograms |
| **React Router 6** | Client-side routing between modules |

---

## 🚀 Installation & Deployment

### Prerequisites

| Tool | Version |
|---|---|
| Python | `^3.8` |
| Node.js | `^16.x` |
| pip | latest |
| npm | latest |

---

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/signal-viewer.git
cd signal-viewer
```

---

### 2. Backend Setup

```bash
cd Backend

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt

# Start the API server
python app.py
```

> API available at `http://localhost:8000`
> Interactive API docs at `http://localhost:8000/docs`

---

### 3. Frontend Setup

```bash
cd Frontend/app

# Install Node dependencies
npm install

# Start the Vite dev server
npm run dev
```

> Web app available at `http://localhost:5173`

---

### 4. Model Files Setup

Place all trained model and reference files in `Backend/models/`:

```
Backend/models/
├── ecg_model.keras                # ECG — CNN classifier
├── ecg_rf_model.pkl               # ECG — Random Forest classifier
├── eeg_model_final.keras          # EEG — CNN model
├── eeg_svm_model.pkl              # EEG — SVM pipeline (StandardScaler + SVC)
├── train_mean.npy                 # EEG — optional normalization mean
├── train_std.npy                  # EEG — optional normalization std
├── ibd_signal_detector.keras      # Microbiome — Bidirectional GRU
├── hmp2_reference.csv             # Microbiome — any .csv from training data
├── finance_stock_model.keras      # Finance — GRU for stocks
├── finance_currency_model.keras   # Finance — GRU for currencies
└── finance_metal_model.keras      # Finance — GRU for metals
```

> **EEG:** Model input shape is auto-detected via a dummy forward pass at load time — no manual constant adjustment needed.

> **Microbiome:** The service scans `models/` for any `.csv` automatically. No renaming required — just copy your training CSV there as-is.

---

### 5. Verify Setup

```bash
# Test signal processing without the HTTP server
python Backend/test_sim.py
python Backend/plot_sim.py

# Confirm API is running
curl http://localhost:8000/
# → open http://localhost:8000/docs for interactive API explorer
```

---

## 📡 API Reference

Base URL: `http://localhost:8000` — all endpoints are prefixed with `/api/{domain}`.

---

### 🫀 Medical — ECG

| Method | Endpoint | Body | Description |
|---|---|---|---|
| `POST` | `/api/medical/process` | `file: .csv` | ECG CSV → AI classification + signals |
| `POST` | `/api/medical/process-wfdb` | `dat_file`, `meta` JSON, optional `xyz_file` | WFDB binary → classification + signals |

**Response**
```json
{
  "analysis": {
    "ai_model":   { "prediction": "NORM", "confidence": 0.94 },
    "classic_ml": { "prediction": "NORM", "confidence": 0.88 }
  },
  "signals": { "lead_I": [...], "lead_II": [...] },
  "time": [0, 1, 2, ...]
}
```

---

### 🧠 EEG

| Method | Endpoint | Body | Description |
|---|---|---|---|
| `POST` | `/api/eeg/process` | `file: .npy or .csv` | CNN + SVM ensemble prediction |

**Response**
```json
{
  "analysis": {
    "cnn": {
      "prediction": "Depression", "confidence": 0.87,
      "probabilities": { "ADFSU": 0.04, "Depression": 0.87, "REEG-PD": 0.06, "BrainLat": 0.03 },
      "window_agreement": 0.91, "n_windows": 127
    },
    "svm": {
      "prediction": "Depression", "confidence": 0.79,
      "probabilities": { "ADFSU": 0.07, "Depression": 0.79, "REEG-PD": 0.09, "BrainLat": 0.05 }
    },
    "verdict": { "agree": true, "prediction": "Depression", "confidence": 0.87, "tiebreak": null }
  },
  "signals": { "EEG_CH1": [...], "EEG_CH19": [...] },
  "time": [0, 1, 2, ...]
}
```

---

### 🔊 Acoustic

| Method | Endpoint | Body | Description |
|---|---|---|---|
| `POST` | `/api/acoustic/simulate` | `{ frequency, velocity }` JSON | Generate Doppler waveform |
| `GET`  | `/api/acoustic/doppler/datasets` | — | List pre-loaded recordings |
| `GET`  | `/api/acoustic/doppler/analyze/{filename}` | — | Analyze a dataset recording |
| `POST` | `/api/acoustic/doppler/upload` | `file: .wav/.mp3` | Upload audio → velocity analysis |
| `POST` | `/api/acoustic/drone/upload` | `file: audio` | Upload audio → drone classification |

**Doppler upload response**
```json
{
  "waveform":    { "time": [...], "amplitude": [...] },
  "fft":         { "frequencies": [...], "magnitudes": [...] },
  "spectrogram": { "times": [...], "frequencies": [...], "power": [[...]] },
  "doppler": {
    "estimated_velocity_kmh": 67.4, "estimated_frequency_hz": 440,
    "approach_freq_hz": 512, "recede_freq_hz": 388,
    "freq_time_axis": [...], "freq_over_time": [...],
    "algorithm": "STFT Peak Tracking"
  },
  "statistics": { "duration_s": 8.2, "sample_rate": 22050, "rms": 0.142, "snr_db": 18.3, "peak_to_peak": 1.94 }
}
```

**Drone upload response**
```json
{
  "filename": "audio.wav",
  "classification": { "label": "Drone Detected", "confidence": 0.91, "score": 4.2, "reasons": ["High ZCR", "Dominant frequency in rotor band"] },
  "waveform":  { "time": [...], "amplitude": [...] },
  "fft":       { "frequencies": [...], "magnitudes": [...] },
  "features":  { "spectral_centroid": 1842.3, "spectral_bandwidth": 920.1, "spectral_rolloff": 3200.5, "dominant_freq": 210.0, "zero_crossing_rate": 0.082 },
  "statistics": { "duration_s": 4.1, "sample_rate": 44100, "rms": 0.211, "snr_db": 14.7 }
}
```

---

### 📈 Finance

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/finance/history/{asset}` | Historical OHLCV data |
| `GET` | `/api/finance/forecast/{asset}` | GRU price forecast + confidence interval |

**Forecast response**
```json
{
  "asset": "EUR-USD", "horizon": 3,
  "forecast": [1.089, 1.091, 1.088],
  "upper":    [1.094, 1.097, 1.093],
  "lower":    [1.084, 1.085, 1.083],
  "dates":    ["2024-06-10", "2024-06-11", "2024-06-12"]
}
```

---

### 🧬 Microbiome

| Method | Endpoint | Body | Description |
|---|---|---|---|
| `POST` | `/api/bio/analyze` | `file: .csv` | Patient CSV → per-patient IBD predictions |

**Response**
```json
{
  "patients": [
    {
      "participant_id": "PATIENT_001", "num_weeks": 20,
      "diagnosis": "Healthy", "confidence": 92.4,
      "probabilities": { "Healthy": 0.924, "Crohn's Disease": 0.051, "Ulcerative Colitis": 0.025 },
      "top_taxa": [{ "name": "Faecalibacterium prausnitzii", "mean_abundance": 44.2 }],
      "weekly_data": { "weeks": [0, 2, 4], "taxa": ["Faecalibacterium prausnitzii"], "values": [[44.2, 43.1]] },
      "fecalcal": [50.0, 50.0]
    }
  ]
}
```

---

## 📂 Project Structure

```
SignalViewer/
│
├── Backend/
│   ├── app.py                         # FastAPI entry point, route registration, CORS config
│   ├── requirements.txt               # All Python dependencies
│   │
│   ├── routes/
│   │   ├── medical_routes.py          # POST /api/medical/process, /process-wfdb
│   │   ├── eeg_routes.py              # POST /api/eeg/process
│   │   ├── acoustic_routes.py         # POST/GET /api/acoustic/*
│   │   ├── finance_routes.py          # GET /api/finance/history, /forecast
│   │   └── bio_routes.py              # POST /api/bio/analyze
│   │
│   ├── services/
│   │   ├── medical_service.py         # ECG parsing, CNN + RandomForest inference
│   │   ├── eeg_service.py             # EEG sliding window, CNN + SVM ensemble, auto shape-detect
│   │   ├── acoustic_service.py        # Doppler STFT estimation, drone spectral features
│   │   ├── finance_service.py         # OHLCV loading, GRU forecasting per asset class
│   │   └── bio_service.py             # Patient sequencing, IBD GRU inference, scaler auto-fit
│   │
│   ├── models/                        # ← All .keras / .pkl / .npy / .csv files here
│   │   ├── ecg_model.keras
│   │   ├── ecg_rf_model.pkl
│   │   ├── eeg_model_final.keras
│   │   ├── eeg_svm_model.pkl
│   │   ├── train_mean.npy             # optional
│   │   ├── train_std.npy              # optional
│   │   ├── ibd_signal_detector.keras
│   │   ├── hmp2_reference.csv
│   │   ├── finance_stock_model.keras
│   │   ├── finance_currency_model.keras
│   │   └── finance_metal_model.keras
│   │
│   ├── uploads/                       # Temp storage — each file deleted after its request
│   ├── data/                          # Static datasets (Doppler recordings, etc.)
│   ├── test_sim.py                    # Standalone pipeline test (no HTTP server needed)
│   └── plot_sim.py                    # Standalone signal plot test
│
├── Frontend/
│   └── app/
│       ├── src/
│       │   ├── pages/
│       │   │   ├── Landing.jsx        # Module selector — 5 domain cards
│       │   │   ├── Medical.jsx        # ECG + EEG viewer (4 modes, playback, AI results)
│       │   │   ├── Acoustic.jsx       # Doppler simulator + analysis + drone detection
│       │   │   ├── Finance.jsx        # Candlestick + SMA + volume + GRU forecast
│       │   │   └── Microbiome.jsx     # IBD patient CSV analysis, per-patient cards
│       │   │
│       │   └── components/
│       │       ├── Sidebar.jsx        # Shared collapsible left sidebar wrapper
│       │       └── ui/
│       │           ├── ToggleTabs.jsx         # Horizontal tab switcher
│       │           ├── SliderControl.jsx      # Labeled range slider with live value
│       │           ├── FileUpload.jsx         # Drag-and-drop + click file input
│       │           ├── StatCard.jsx           # Titled result card container
│       │           ├── ChannelControl.jsx     # Per-channel visibility / color / thickness
│       │           └── ColormapSelector.jsx   # Plotly colormap dropdown
│       │
│       ├── package.json
│       ├── vite.config.js
│       └── tailwind.config.js
│
└── docs/
    └── images/                        # ← Place all screenshots here
        ├── landing.png
        ├── medical_landing.png
        ├── ecg_continuous_multipanel.png
        ├── ecg_xor.png
        ├── ecg_polar.png
        ├── ecg_trajectory.png
        ├── ecg_ai_results.png
        ├── eeg_tab_selector.png
        ├── eeg_results_cards.png
        ├── eeg_waveform.png
        ├── acoustic_landing.png
        ├── acoustic_simulator.png
        ├── acoustic_analysis.png
        ├── acoustic_drone.png
        ├── finance_overview.png
        ├── finance_candlestick.png
        ├── finance_forecast.png
        ├── microbiome_upload.png
        ├── microbiome_results.png
        ├── microbiome_patient_card.png
        ├── microbiome_timeline.png
        ├── microbiome_probs.png
        └── microbiome_summary.png
```

---

## 📘 Module Documentation

---

# 🫀 Medical Signal Viewer — ECG Analysis

> **Module:** `Medical.jsx` · `medical_routes.py` · `medical_service.py`

---

## Overview

Supports ECG upload and visualization with four interactive modes, animated playback, and dual-model AI classification.

<!-- FIGURE: Medical Module Landing -->
<!-- ![Medical Landing](docs/images/medical_landing.png) -->
> 📸 *Screenshot of the Medical module with signal type selector (ECG / EEG): `docs/images/medical_landing.png`*

---

## Supported File Formats

| Format | Description | Notes |
|---|---|---|
| `.csv` | Comma-separated, one row per sample | Columns = leads |
| `.hea + .dat` | WFDB binary format | Upload both; header parsed for gain/baseline |
| `.xyz` | Frank XYZ lead system | Used for ML model input, not visualized |

---

## Visualization Modes

### 1. Continuous

Real-time scrollable multi-channel waveform display.

<!-- FIGURE: ECG Continuous Multi-Panel View -->
<!-- ![ECG Continuous](docs/images/ecg_continuous_multipanel.png) -->
> 📸 *Screenshot of ECG multi-panel continuous view: `docs/images/ecg_continuous_multipanel.png`*

- **Multi-Panel** — each lead in its own panel
- **Overlay** — all leads superimposed on one chart
- Animated playback: speed 0.25× – 4×
- Zoom: adjustable window 100 – 5000 samples
- Per-channel: toggle visibility, color, line thickness

### 2. XOR Analysis

Bitwise XOR comparison between any two selected leads.

<!-- FIGURE: ECG XOR Mode -->
<!-- ![ECG XOR](docs/images/ecg_xor.png) -->
> 📸 *Screenshot of ECG XOR mode with energy bars: `docs/images/ecg_xor.png`*

- Binarizes both signals (threshold 0.5 after normalization)
- XOR per sample highlights timing disagreements
- XOR energy per chunk (16–256 samples) as bar chart

### 3. Polar Periodicity

Ratio `|Channel A| / |Channel B|` as a polar plot.

<!-- FIGURE: ECG Polar Mode -->
<!-- ![ECG Polar](docs/images/ecg_polar.png) -->
> 📸 *Screenshot of ECG polar periodicity plot: `docs/images/ecg_polar.png`*

- Theta wraps every N samples (configurable)
- Radius normalized to 95th percentile
- Live stats: mean r, std r, p95, revolution count

### 4. Trajectory (Phase Space)

Phase-space trajectory of Channel A vs Channel B.

<!-- FIGURE: ECG Trajectory Mode -->
<!-- ![ECG Trajectory](docs/images/ecg_trajectory.png) -->
> 📸 *Screenshot of ECG trajectory plot: `docs/images/ecg_trajectory.png`*

- Color-encoded by time index (selectable colormap)
- Start (green) and end (red) markers
- Stats: path length, cross-correlation, mean ± std

---

## AI Classification

| Model | Type | Classes |
|---|---|---|
| AI Model | CNN (Deep Learning) | NORM, MI, STTC, CD, HYP |
| Classic ML | Random Forest | NORM, MI, STTC, CD, HYP |

<!-- FIGURE: ECG AI Results Panel -->
<!-- ![ECG AI Results](docs/images/ecg_ai_results.png) -->
> 📸 *Screenshot of AI + Classic ML result cards in sidebar: `docs/images/ecg_ai_results.png`*

---

## Screenshot Index — ECG

| File | What to Capture |
|---|---|
| `docs/images/medical_landing.png` | Module with ECG/EEG selector |
| `docs/images/ecg_continuous_multipanel.png` | Multi-panel continuous view |
| `docs/images/ecg_xor.png` | XOR mode with energy bars |
| `docs/images/ecg_polar.png` | Polar periodicity plot |
| `docs/images/ecg_trajectory.png` | Phase-space trajectory |
| `docs/images/ecg_ai_results.png` | AI + Classic ML result cards |

---
---

# 🧠 EEG Signal Viewer — Neurological Classification

> **Module:** `Medical.jsx` (EEG tab) · `eeg_routes.py` · `eeg_service.py`

---

## Overview

19-channel EEG analysis through a sliding-window CNN + SVM ensemble pipeline for 4-class neurological classification.

<!-- FIGURE: EEG Tab Selector -->
<!-- ![EEG Tab](docs/images/eeg_tab_selector.png) -->
> 📸 *Screenshot of the Medical module with EEG signal type selected: `docs/images/eeg_tab_selector.png`*

---

## Supported File Formats

| Format | Shape | Notes |
|---|---|---|
| `.npy` | `(T, 19)` | Used directly |
| `.npy` | `(19, T)` | Auto-transposed |
| `.npy` | `(N, T, 19)` | Flattened to `(N×T, 19)` |
| `.csv` | `(T, 19)` | Rows = samples, columns = channels |

---

## Processing Pipeline

```
Upload (.npy / .csv)
    │
    ▼
Reshape → (T, 19)
    │
    ▼
Sliding Window — 992 samples, 50% overlap (step = 496)
    │
    ▼
Normalize — per-channel global z-score
    │
    ├──► CNN  →  expand_dims (N,992,19,1)  →  predict  →  soft-vote
    └──► SVM  →  extract features (mean, std, min, max per channel)
                 →  76 features  →  predict_proba  →  soft-vote
    │
    ▼
Verdict — higher-confidence model wins on disagreement
```

---

## Classification Classes

| Index | Class | Condition |
|---|---|---|
| 0 | ADFSU | Attention Deficit / related spectrum |
| 1 | Depression | Major depressive disorder |
| 2 | REEG-PD | Parkinson's Disease resting EEG |
| 3 | BrainLat | BrainLat dataset condition |

---

## Model Details

| Property | CNN | SVM |
|---|---|---|
| **File** | `eeg_model_final.keras` | `eeg_svm_model.pkl` |
| **Input** | `(N, 992, 19, 1)` | `(N, 76)` — auto-detected from `n_features_in_` |
| **Architecture** | Conv2D → MaxPool → Flatten → Dense | StandardScaler → SVC (Pipeline) |
| **Voting** | Soft-vote mean across windows | `predict_proba` soft-vote |

<!-- FIGURE: EEG CNN + SVM Result Cards -->
<!-- ![EEG Results](docs/images/eeg_results_cards.png) -->
> 📸 *Screenshot of EEG CNN Report + SVM Report sidebar cards: `docs/images/eeg_results_cards.png`*

<!-- FIGURE: EEG Multi-Channel Waveform -->
<!-- ![EEG Waveform](docs/images/eeg_waveform.png) -->
> 📸 *Screenshot of the 19-channel EEG waveform in the main panel: `docs/images/eeg_waveform.png`*

---

## Required Model Files

```
Backend/models/
├── eeg_model_final.keras     # required
├── eeg_svm_model.pkl         # required
├── train_mean.npy            # optional — training normalization mean
└── train_std.npy             # optional — training normalization std
```

> If normalization files are absent, per-channel global z-score is computed from the uploaded file automatically.

---

## Screenshot Index — EEG

| File | What to Capture |
|---|---|
| `docs/images/eeg_tab_selector.png` | Medical module with EEG selected |
| `docs/images/eeg_results_cards.png` | CNN + SVM result cards in sidebar |
| `docs/images/eeg_waveform.png` | 19-channel waveform in main panel |

---
---

# 🔊 Acoustic Signal Viewer — Doppler & Drone Detection

> **Module:** `Acoustic.jsx` · `acoustic_routes.py` · `acoustic_service.py`

---

## Overview

Three-tab audio analysis suite: Doppler simulation, real-recording vehicle speed estimation, and drone sound classification.

<!-- FIGURE: Acoustic Module Tab Overview -->
<!-- ![Acoustic Landing](docs/images/acoustic_landing.png) -->
> 📸 *Screenshot of the Acoustic module showing all three tabs: `docs/images/acoustic_landing.png`*

---

## Tab 1 — Doppler Simulator

Generates synthetic Doppler-shifted audio from parameters and plays it back in the browser.

<!-- FIGURE: Doppler Simulator -->
<!-- ![Doppler Simulator](docs/images/acoustic_simulator.png) -->
> 📸 *Screenshot: waveform + frequency chart + audio player: `docs/images/acoustic_simulator.png`*

| Control | Range | Default |
|---|---|---|
| Horn Frequency | 100 – 2000 Hz | 440 Hz |
| Vehicle Speed | 10 – 200 km/h | 80 km/h |

**Charts:** Waveform · Observed Frequency Over Time (with dashed source frequency line) · In-browser WAV audio player

---

## Tab 2 — Doppler Analysis

Analyzes real recordings to estimate vehicle speed from spectral Doppler shift.

<!-- FIGURE: Doppler Analysis -->
<!-- ![Doppler Analysis](docs/images/acoustic_analysis.png) -->
> 📸 *Screenshot: waveform + FFT + Doppler curve + spectrogram: `docs/images/acoustic_analysis.png`*

**Input:** Pre-loaded dataset dropdown or custom `.wav`/`.mp3` upload

**Charts:** Waveform · FFT Spectrum (0–3000 Hz) · Doppler Curve with approach/recede reference lines · Spectrogram heatmap

**Results card:** Estimated speed (km/h), approach/recede frequencies, actual speed (if labeled in dataset), error %, algorithm name

---

## Tab 3 — Drone Detection

Classifies audio as drone or non-drone using spectral feature analysis.

<!-- FIGURE: Drone Detection -->
<!-- ![Drone Detection](docs/images/acoustic_drone.png) -->
> 📸 *Screenshot: drone results with waveform + FFT + spectral features: `docs/images/acoustic_drone.png`*

**Formats:** `.wav`, `.mp3`, `.ogg`, `.flac`

**Charts:** Waveform · FFT (0–5000 Hz) · Spectral Features bar chart

| Badge | Threshold |
|---|---|
| Drone Detected | confidence ≥ 60% |
| Possible Drone | confidence 40–60% |
| No Drone | confidence < 40% |

---

## Screenshot Index — Acoustic

| File | What to Capture |
|---|---|
| `docs/images/acoustic_landing.png` | Module with all 3 tabs |
| `docs/images/acoustic_simulator.png` | Simulator: waveform + frequency + audio player |
| `docs/images/acoustic_analysis.png` | Analysis: 4 charts + results card |
| `docs/images/acoustic_drone.png` | Drone: results + spectral features chart |

---
---

# 📈 Finance Signal Viewer — Market Analysis & Forecasting

> **Module:** `Finance.jsx` · `finance_routes.py` · `finance_service.py`

---

## Overview

Candlestick charting, SMA technical indicators, volume bars, and GRU-based price forecasting across stocks, currencies, and metals.

<!-- FIGURE: Finance Overview -->
<!-- ![Finance Overview](docs/images/finance_overview.png) -->
> 📸 *Screenshot of Finance module with candlestick chart and asset selector: `docs/images/finance_overview.png`*

---

## Asset Registry

| Category | Assets | Forecast Horizon |
|---|---|---|
| 📈 Stocks | ABTX, AAT | 5 days |
| 💱 Currencies | EUR/USD, USD/JPY | 3 days |
| 🪙 Metals | Gold, Silver | 30 days |

---

## Charts

**Candlestick** — OHLC candles with SMA-20 (blue dashed) and SMA-50 (amber dashed) overlays

<!-- FIGURE: Finance Candlestick -->
<!-- ![Finance Candlestick](docs/images/finance_candlestick.png) -->
> 📸 *Screenshot of candlestick chart with SMA overlays: `docs/images/finance_candlestick.png`*

**Volume** — Bar chart below candlestick, color-matched to candle direction

**GRU Forecast** — Historical close line + forecast dashed line + shaded confidence band

<!-- FIGURE: Finance Forecast -->
<!-- ![Finance Forecast](docs/images/finance_forecast.png) -->
> 📸 *Screenshot of GRU forecast with confidence band: `docs/images/finance_forecast.png`*

---

## Required Model Files

```
Backend/models/
├── finance_stock_model.keras
├── finance_currency_model.keras
└── finance_metal_model.keras
```

---

## Screenshot Index — Finance

| File | What to Capture |
|---|---|
| `docs/images/finance_overview.png` | Asset selector + candlestick chart |
| `docs/images/finance_candlestick.png` | Candlestick + SMA-20 + SMA-50 |
| `docs/images/finance_forecast.png` | GRU forecast + confidence band |

---
---
