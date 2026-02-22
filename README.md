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

<!-- FIGURE 1: Landing Page Screenshot -->
<!-- Replace the line below with your actual screenshot -->
<!-- ![Landing Page](docs/images/landing.png) -->
> 📸 *Place a screenshot of the landing page here: `docs/images/landing.png`*

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Live Modules](#-live-modules)
  - [Medical Signal Viewer](#-medical-signal-viewer-ecg--eeg)
  - [Acoustic Signal Viewer](#-acoustic-signal-viewer)
  - [Finance Signal Viewer](#-finance-signal-viewer)
  - [Microbiome Signal Viewer](#-microbiome-signal-viewer)
- [System Architecture](#-system-architecture)
- [Tech Stack](#-tech-stack)
- [Installation](#-installation--deployment)
- [API Reference](#-api-reference)
- [Model Details](#-ml-model-details)
- [File Format Guide](#-file-format-guide)
- [Project Structure](#-project-structure)

---

## 🌐 Overview

SignalViewer is an enterprise-grade, full-stack platform engineered for the interactive exploration and AI-assisted analysis of signals across five scientific and financial domains. It combines a high-performance async Python backend with a reactive modern web interface.

### Core Capabilities

| Capability | Details |
|---|---|
| **Cross-Domain Coverage** | Medical (ECG/EEG), Acoustic, Financial, Microbiome |
| **AI Inference** | CNN + SVM ensemble (EEG), GRU sequence model (IBD), ML classifiers |
| **Visualization Engine** | Plotly.js — interactive charts, heatmaps, spectrograms, polar plots |
| **Real-time Playback** | Animated signal scrubbing with zoom and speed controls |
| **File Support** | CSV, NPY, WAV, MP3, WFDB (.hea + .dat) |
| **API Architecture** | FastAPI async REST, auto-generated OpenAPI docs |

---

## 🧩 Live Modules

### 🫀 Medical Signal Viewer (ECG + EEG)

The most feature-rich module — supports both ECG and EEG signal types with a multi-mode visualization engine and dual-model AI inference.

<!-- FIGURE 2: Medical Module — ECG Multi-Panel View -->
<!-- ![Medical ECG Multi-Panel](docs/images/medical_ecg_multipanel.png) -->
> 📸 *Place a screenshot of the ECG multi-panel view here: `docs/images/medical_ecg_multipanel.png`*

**ECG Features:**
- Upload `.csv`, `.hea + .dat` (WFDB binary), or `.xyz` (Frank lead) files
- Up to 20-lead simultaneous visualization
- **4 viewer modes:** Continuous, XOR Analysis, Polar Periodicity, Trajectory (Phase Space)
- Animated playback with adjustable speed (0.25× – 4×) and zoom
- Multi-panel or overlay display
- Per-channel color, thickness, and visibility controls
- CNN (deep learning) + Random Forest (classic ML) classification

<!-- FIGURE 3: Medical Module — ECG Viewer Modes -->
<!-- ![ECG Viewer Modes](docs/images/medical_ecg_modes.png) -->
> 📸 *Place a side-by-side of the 4 viewer modes (Continuous / XOR / Polar / Trajectory): `docs/images/medical_ecg_modes.png`*

**EEG Features:**
- Upload `.npy` (NumPy array) or `.csv` files
- Accepts shapes: `(T, 19)`, `(19, T)`, or `(N, T, 19)` — auto-reshaped
- Sliding window pipeline: 992-sample windows, 50% overlap
- **CNN + SVM ensemble** with per-window soft voting
- 4-class IBD classification: `ADFSU`, `Depression`, `REEG-PD`, `BrainLat`
- Window agreement score and confidence breakdown

<!-- FIGURE 4: EEG Analysis Results -->
<!-- ![EEG Results Panel](docs/images/medical_eeg_results.png) -->
> 📸 *Place a screenshot of EEG analysis results with CNN/SVM predictions: `docs/images/medical_eeg_results.png`*

| ECG Classes | EEG Classes |
|---|---|
| NORM, MI, STTC, CD, HYP | ADFSU, Depression, REEG-PD, BrainLat |

---

### 🔊 Acoustic Signal Viewer

Three-tab acoustic analysis suite covering simulation, real-signal analysis, and drone classification.

<!-- FIGURE 5: Acoustic Module — Doppler Simulator -->
<!-- ![Doppler Simulator](docs/images/acoustic_simulator.png) -->
> 📸 *Place a screenshot of the Doppler simulator with waveform + frequency charts: `docs/images/acoustic_simulator.png`*

**Tab 1 — Doppler Simulator:**
- Interactive sliders for horn frequency (100–2000 Hz) and vehicle speed (10–200 km/h)
- Backend-generated waveform with audio playback (WAV synthesized in-browser)
- Observed frequency-over-time chart with source frequency reference line

**Tab 2 — Doppler Analysis:**
- Select from pre-loaded dataset recordings or upload your own `.wav` / `.mp3`
- Waveform, FFT spectrum, frequency-over-time Doppler curve, and spectrogram
- Estimated vehicle speed, approach/recede frequencies, SNR, and RMS statistics

<!-- FIGURE 6: Acoustic Module — Doppler Analysis -->
<!-- ![Doppler Analysis](docs/images/acoustic_analysis.png) -->
> 📸 *Place a screenshot showing waveform + spectrogram + Doppler curve: `docs/images/acoustic_analysis.png`*

**Tab 3 — Drone Detection:**
- Upload any audio file (WAV, MP3, OGG, FLAC)
- Spectral feature extraction: centroid, bandwidth, rolloff, dominant frequency, ZCR
- Classification: `Drone Detected` / `Possible Drone` / `No Drone`
- Waveform + FFT + spectral features bar chart

<!-- FIGURE 7: Drone Detection Results -->
<!-- ![Drone Detection](docs/images/acoustic_drone.png) -->
> 📸 *Place a screenshot of drone detection results with feature bar chart: `docs/images/acoustic_drone.png`*

---

### 📈 Finance Signal Viewer

Financial market analysis with candlestick charting, technical indicators, and multi-asset AI forecasting.

<!-- FIGURE 8: Finance Module — Candlestick + SMA -->
<!-- ![Finance Candlestick](docs/images/finance_candlestick.png) -->
> 📸 *Place a screenshot of the candlestick chart with SMA overlays: `docs/images/finance_candlestick.png`*

**Supported Asset Classes:**

| Category | Assets | Forecast Horizon |
|---|---|---|
| 📈 Stocks | ABTX, AAT | 5 days |
| 💱 Currencies | EUR/USD, USD/JPY | 3 days |
| 🪙 Metals | Gold, Silver | 30 days |

**Features:**
- Candlestick OHLC charts with SMA-20 and SMA-50 overlays
- Volume bar chart
- GRU-based price forecasting with confidence intervals
- Historical data viewer with adjustable lookback window
- Statistical summary: mean, std, min/max, daily change

<!-- FIGURE 9: Finance Forecast Chart -->
<!-- ![Finance Forecast](docs/images/finance_forecast.png) -->
> 📸 *Place a screenshot of the GRU forecast with confidence band: `docs/images/finance_forecast.png`*

---

### 🧬 Microbiome Signal Viewer

Longitudinal gut microbiome analysis with IBD classification using a GRU sequence model trained on the HMP2 dataset.

<!-- FIGURE 10: Microbiome Module — Upload + Results -->
<!-- ![Microbiome Results](docs/images/microbiome_results.png) -->
> 📸 *Place a screenshot showing patient cards with diagnosis badges: `docs/images/microbiome_results.png`*

**Features:**
- Upload patient CSV files (multi-patient supported in one file)
- Per-patient longitudinal sequence → GRU model → diagnosis prediction
- 3-class IBD classification: `Healthy`, `Crohn's Disease`, `Ulcerative Colitis`
- Top-5 contributing taxa ranked by mean abundance with bar visualization
- Timeline chart of taxa abundance across weeks
- Probability breakdown per class with confidence bar
- Auto-detects `Participant ID`, `week_num`, and microbiome feature columns

<!-- FIGURE 11: Microbiome — Per-Patient Card Detail -->
<!-- ![Microbiome Patient Card](docs/images/microbiome_patient_card.png) -->
> 📸 *Place a close-up of a single patient card with timeline + probability charts: `docs/images/microbiome_patient_card.png`*

**Diagnosis Color Coding:**

| Diagnosis | Color |
|---|---|
| ✅ Healthy | Green |
| 🔴 Crohn's Disease | Red |
| 🟡 Ulcerative Colitis | Amber |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        FRONTEND (React)                      │
│  Landing → Medical │ Acoustic │ Finance │ Microbiome        │
│  Plotly.js visualizations  │  File uploads  │  AI results   │
└────────────────────────────┬────────────────────────────────┘
                             │ HTTP REST (FastAPI)
                             │ http://localhost:8000
┌────────────────────────────▼────────────────────────────────┐
│                        BACKEND (FastAPI)                     │
│  routes/         services/          models/                  │
│  ├ medical       ├ medical_service  ├ eeg_model_final.keras  │
│  ├ acoustic      ├ eeg_service      ├ eeg_svm_model.pkl      │
│  ├ finance       ├ acoustic_service ├ ibd_signal_detector    │
│  ├ bio           ├ bio_service      ├ finance GRU models     │
│  └ eeg           └ finance_service  └ hmp2_reference.csv     │
└─────────────────────────────────────────────────────────────┘
```

<!-- FIGURE 12: Architecture Diagram -->
<!-- Replace with a proper architecture diagram if you have one -->
<!-- ![Architecture](docs/images/architecture.png) -->
> 📸 *Optionally place a detailed architecture diagram here: `docs/images/architecture.png`*

---

## 🛠️ Tech Stack

### Backend
| Library | Purpose |
|---|---|
| **FastAPI** | Async REST API framework |
| **TensorFlow / Keras** | CNN (EEG), GRU (IBD, Finance) model inference |
| **scikit-learn** | SVM classifier, StandardScaler, LabelEncoder |
| **NumPy / Pandas** | Signal processing and data manipulation |
| **SciPy** | Statistical feature extraction (skewness, kurtosis) |
| **Librosa** | Audio feature extraction for acoustic analysis |
| **Joblib** | Model serialization (.pkl) |

### Frontend
| Library | Purpose |
|---|---|
| **React 18** | Component-based UI framework |
| **Vite** | Build tool and dev server |
| **Tailwind CSS** | Utility-first styling |
| **Plotly.js** | Interactive scientific charts |
| **React Router** | Client-side navigation |

---

## 🚀 Installation & Deployment

### Prerequisites

- **Python** `^3.8`
- **Node.js** `^16.x`
- **pip** and **npm**

### 1. Backend Setup

```bash
cd Backend

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the API server
python app.py
```

> API runs at `http://localhost:8000`
> Interactive docs at `http://localhost:8000/docs`

### 2. Frontend Setup

```bash
cd Frontend/app

# Install dependencies
npm install

# Start the Vite dev server
npm run dev
```

> App runs at `http://localhost:5173`

### 3. Models Setup

Place the following files in `Backend/models/`:

```
Backend/models/
├── eeg_model_final.keras         # EEG CNN model
├── eeg_svm_model.pkl             # EEG SVM model
├── ibd_signal_detector.keras     # Microbiome GRU model
├── hmp2_reference.csv            # HMP2 training reference CSV (any .csv works)
├── finance_stock_model.keras     # Finance GRU — stocks
├── finance_currency_model.keras  # Finance GRU — currencies
└── finance_metal_model.keras     # Finance GRU — metals
```

---

## 📡 API Reference

### Medical / ECG
| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/medical/process` | Upload ECG CSV → AI analysis + signals |
| `POST` | `/api/medical/process-wfdb` | Upload WFDB (.dat + meta + .xyz) → analysis |

### EEG
| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/eeg/process` | Upload `.npy` or `.csv` → CNN+SVM prediction |

### Acoustic
| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/acoustic/simulate` | Generate Doppler waveform from params |
| `GET` | `/api/acoustic/doppler/datasets` | List available Doppler recordings |
| `GET` | `/api/acoustic/doppler/analyze/{filename}` | Analyze a dataset recording |
| `POST` | `/api/acoustic/doppler/upload` | Upload audio → Doppler analysis |
| `POST` | `/api/acoustic/drone/upload` | Upload audio → drone classification |

### Finance
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/finance/history/{asset}` | Get historical OHLC data |
| `GET` | `/api/finance/forecast/{asset}` | Get GRU price forecast |

### Microbiome
| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/bio/analyze` | Upload patient CSV → IBD diagnosis per patient |

---

## 🤖 ML Model Details

### EEG Disease Classification
| Property | Value |
|---|---|
| **Architecture** | CNN (Conv2D → MaxPool → Flatten → Dense) |
| **Ensemble** | CNN soft-vote + SVM (Pipeline: StandardScaler → SVC) |
| **Input** | `(N_windows, 992, 19, 1)` — sliding window, 50% overlap |
| **Classes** | ADFSU, Depression, REEG-PD, BrainLat |
| **Normalization** | Per-channel global z-score across all windows |
| **Output** | Per-class probabilities → argmax + window agreement score |

### IBD Microbiome Classification
| Property | Value |
|---|---|
| **Architecture** | Bidirectional GRU (64 units) + Dropout(0.3) + Dense |
| **Input** | `(1, 45, N_microbe_features)` — padded patient sequence |
| **Classes** | Healthy, Crohn's Disease, Ulcerative Colitis |
| **Training Data** | HMP2 IBD Metagenomics Atlas |
| **Normalization** | StandardScaler fitted on training reference CSV |
| **Class Balancing** | Oversampling to equal class counts |

### Finance Forecasting
| Property | Value |
|---|---|
| **Architecture** | GRU sequence model |
| **Assets** | Stocks (5-day), Currencies (3-day), Metals (30-day) |
| **Features** | OHLCV + multi-pair cross-rates (currencies) |

---

## 📁 File Format Guide

### ECG — CSV
```
Columns: lead_I, lead_II, lead_III, ...   (one row per sample)
```

### ECG — WFDB
```
Upload: .hea (header) + .dat (binary signal) + .xyz (Frank leads, optional)
```

### EEG — NumPy
```
Shape: (T, 19)   — T timesteps, 19 channels
       (19, T)   — auto-transposed
       (N, T, 19) — N segments, auto-flattened to (N*T, 19)
```

### Microbiome — CSV
```
Required columns : Participant ID, week_num (or week/time/visit)
Optional columns : fecalcal, External ID
Remaining columns: microbiome species abundance values
```

---

## 📂 Project Structure

```
SignalViewer/
├── Backend/
│   ├── app.py                    # FastAPI entry point
│   ├── requirements.txt
│   ├── routes/
│   │   ├── medical_routes.py
│   │   ├── eeg_routes.py
│   │   ├── acoustic_routes.py
│   │   ├── finance_routes.py
│   │   └── bio_routes.py
│   ├── services/
│   │   ├── medical_service.py
│   │   ├── eeg_service.py
│   │   ├── acoustic_service.py
│   │   ├── finance_service.py
│   │   └── bio_service.py
│   ├── models/                   # ← Place .keras / .pkl / .csv here
│   ├── uploads/                  # Temp storage for uploaded files
│   └── data/                     # Static datasets
│
├── Frontend/
│   └── app/
│       ├── src/
│       │   ├── pages/
│       │   │   ├── Landing.jsx
│       │   │   ├── Medical.jsx
│       │   │   ├── Acoustic.jsx
│       │   │   ├── Finance.jsx
│       │   │   └── Microbiome.jsx
│       │   └── components/
│       │       ├── Sidebar.jsx
│       │       └── ui/
│       │           ├── ToggleTabs.jsx
│       │           ├── SliderControl.jsx
│       │           ├── FileUpload.jsx
│       │           ├── StatCard.jsx
│       │           ├── ChannelControl.jsx
│       │           └── ColormapSelector.jsx
│       ├── package.json
│       └── vite.config.js
│
└── docs/
    └── images/                   # ← Place all screenshots here
        ├── landing.png
        ├── medical_ecg_multipanel.png
        ├── medical_ecg_modes.png
        ├── medical_eeg_results.png
        ├── acoustic_simulator.png
        ├── acoustic_analysis.png
        ├── acoustic_drone.png
        ├── finance_candlestick.png
        ├── finance_forecast.png
        ├── microbiome_results.png
        └── microbiome_patient_card.png
```

---

## 🧪 Development & Testing

```bash
# Test signal generation pipeline (no HTTP server needed)
python Backend/test_sim.py
python Backend/plot_sim.py

# API documentation (interactive)
http://localhost:8000/docs
```

---

## 📄 License & Attribution

*(Include organizational licensing details or proprietary notices here)*

---

<div align="center">
Built with ⚡ FastAPI · React · TensorFlow · Plotly.js
</div>
