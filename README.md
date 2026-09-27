# Sentinel-2 Land Use & Land Cover (LULC) Classification System

An end-to-end Machine Learning web application for Land Use and Land Cover (LULC) classification using 10m Sentinel-2 multispectral satellite imagery over the Vijayawada region, featuring three comparative model architectures (pixel-level Random Forest, patch-level Random Forest, and patch-level EfficientNetB0) with an interactive React dashboard.

---

## 📌 Project Overview

This project implements an automated Land Use / Land Cover (LULC) classification pipeline for the **Vijayawada, India** agricultural-urban region using multi-temporal **Sentinel-2** satellite imagery paired with ground-truth label tiles for the Vijayawada region.

The system classifies satellite imagery into **6 thematic LULC classes**:
- `0`: **Water** (`#1E88E5` - Blue)
- `1`: **Trees / Forest** (`#2E7D32` - Dark Green)
- `2`: **Crops** (`#81C784` - Light Green)
- `3`: **Grassland** (`#DCE775` - Yellow-Green)
- `4`: **Built-up** (`#E53935` - Red)
- `5`: **Bare land** (`#8D6E63` - Brown)
- `255`: **Invalid / Masked** (`transparent` - Excluded from training & evaluation)

Each Sentinel-2 tile consists of **5 multispectral channels**:
1. **Band 2 (Blue)**: ~490 nm Surface Reflectance
2. **Band 3 (Green)**: ~560 nm Surface Reflectance
3. **Band 4 (Red)**: ~665 nm Surface Reflectance
4. **Band 8 (NIR)**: ~842 nm Surface Reflectance
5. **NDVI**: Normalized Difference Vegetation Index $\frac{\text{B8} - \text{B4}}{\text{B8} + \text{B4}}$

---

## 🧠 Model Architectures & Approaches

The system provides three distinct model architectures accessible via an interactive model selector in the dashboard UI:

```
                      ┌───────────────────────────────────────────────┐
                      │        Sentinel-2 Input GeoTIFF Tile          │
                      │         (B2, B3, B4, B8, NDVI Bands)          │
                      └──────────────────────┬────────────────────────┘
                                             │
               ┌─────────────────────────────┼─────────────────────────────┐
               ▼                             ▼                             ▼
   ┌───────────────────────┐   ┌───────────────────────────┐   ┌───────────────────────────┐
   │    Pixel-Level RF     │   │      Patch-Level RF       │   │  Patch-Level EfficientNet │
   │      (pixel_rf)       │   │        (rf_patch)         │   │   (efficientnet_patch)    │
   ├───────────────────────┤   ├───────────────────────────┤   ├───────────────────────────┤
   │ • 10m Pixel Resolution│   │ • 64x64 Block Resolution  │   │ • 64x64 Block Resolution  │
   │ • 5 Spectral Features │   │ • 25 Statistical Features │   │ • Pretrained ImageNet CNN │
   │ • Full spatial detail │   │ • Fast inference (~0.4s)  │   │ • 3-Channel RGB Input     │
   │ • ~53s per tile       │   │ • Block-level assignment  │   │ • ~10.5s per tile         │
   └───────────┬───────────┘   └─────────────┬─────────────┘   └─────────────┬─────────────┘
               │                             │                               │
               │                             └───────────────┬───────────────┘
               │                                             │
               │                                             ▼
               │                               ┌───────────────────────────┐
               │                               │ Visual Smoothing (RGBA)   │
               │                               │ (Cosmetic blur filter)    │
               └───────────────────────┬───────┴───────────────────────────┘
                                       │
                                       ▼
                       ┌───────────────────────────────┐
                       │   Interactive Dashboard Map   │
                       │     & Per-Class Analytics     │
                       └───────────────────────────────┘
```

### 1. Pixel-Level Random Forest (`pixel_rf`)
- **Granularity**: Native 10m pixel-level resolution.
- **Input Features**: 5 features per pixel (B2, B3, B4, B8, NDVI) standardized with `StandardScaler`.
- **Architecture**: 300 decision trees, unconstrained depth (`max_depth=None`), `min_samples_leaf=3`, `max_samples=0.35`, with balanced class weighting.
- **Key Advantage**: Precise delineations along narrow waterways, roads, and land boundaries without blockiness.

### 2. Patch-Level Random Forest (`rf_patch`)
- **Granularity**: 64x64 pixel block resolution ($640\text{m} \times 640\text{m}$).
- **Input Features**: 25 summary statistical features per block computed across all 5 bands in exact native order (B2, B3, B4, B8, NDVI):
  - $\text{Mean}, \text{Std} (\text{ddof}=0), \text{Min}, \text{Max}, \text{Median}$ directly from raw reflectance.
- **Architecture**: Scikit-Learn `RandomForestClassifier` trained on patch-level feature vectors.
- **Key Advantage**: Ultra-fast execution (~0.4s per tile) for rapid macro-level land cover screening.

### 3. Patch-Level EfficientNetB0 (`efficientnet_patch`)
- **Granularity**: 64x64 pixel block resolution ($640\text{m} \times 640\text{m}$).
- **Input Features**: 3-channel composite (Band 2 Blue, Band 3 Green, Band 4 Red) scaled by flat multiplier ($255.0$) to convert raw reflectance into standard image representation.
- **Architecture**: Pretrained `EfficientNetB0` convolutional backbone (frozen ImageNet base) with GlobalAveragePooling2D and a 6-class softmax classification head.
- **Key Advantage**: Exploits spatial textural representations learned from deep computer vision features.

### 4. Cosmetic Visual Smoothing
- For patch-based models (`rf_patch` and `efficientnet_patch`), an optional Gaussian smoothing filter is applied **exclusively to the rendered RGBA PNG image** to soften 64x64 block boundaries in the dashboard viewer.
- **Integrity Guarantee**: Visual smoothing is strictly cosmetic. The underlying classification arrays, class distributions, pixel counts, percentages, and analytics remain genuine 64x64 block data. Invalid / masked (nodata) pixels remain 100% transparent.

---

## 📂 Repository Structure

```text
lulc-project/
├── backend/
│   ├── app/
│   │   ├── main.py                     # FastAPI application, CORS & lifespan startup loader
│   │   ├── routes/
│   │   │   └── classify.py             # Endpoints: /api/classify, /api/model-info, /api/legend
│   │   ├── services/
│   │   │   ├── model_service.py        # Pixel-level Random Forest inference & scaler service
│   │   │   └── patch_model_service.py  # Patch-based RF & EfficientNetB0 inference service
│   │   └── utils/
│   │       └── raster_utils.py         # GeoTIFF parsing, RGBA color-mapping & visual smoothing
│   ├── ml/
│   │   ├── models/
│   │   │   ├── lulc_rf_model.joblib    # Pixel-level Random Forest model (tracked via Git LFS)
│   │   │   ├── scaler.joblib           # Fitted StandardScaler for pixel features
│   │   │   └── model_metadata.json     # Hyperparameters, feature importances & class mapping
│   │   ├── models_patch/
│   │   │   ├── random_forest_patch.pkl # Patch-level Random Forest model (25 statistical features)
│   │   │   └── efficientnetb0_patch.keras # Patch-level EfficientNetB0 transfer learning model
│   │   ├── outputs/
│   │   │   ├── metrics/test_metrics.json # Test metrics (Accuracy, F1, Cohen's Kappa)
│   │   │   └── maps/legend.png         # Rendered class palette legend
│   │   └── src/
│   │       ├── preprocessing/          # Tile cleaning, nodata filtering & dataset builder
│   │       ├── training/               # Training pipelines for pixel and patch models
│   │       └── evaluation/             # Test evaluation & map reconstruction
│   ├── requirements.txt                # Python backend dependencies (FastAPI, Rasterio, TensorFlow, etc.)
│   ├── test_all_three_models.py        # Verification script for all 3 classification pipelines
│   └── test_phase8_full_suite.py       # Full multi-tile integration test suite
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── UploadPanel.jsx         # Model selector (pixel_rf, rf_patch, efficientnet_patch) & tile uploader
│   │   │   ├── MapViewer.jsx           # Multi-view visualizer (LULC, RGB, Side-by-Side, Swipe)
│   │   │   ├── ModelInfoCard.jsx       # Model telemetry, comparison trade-offs & limitations badge
│   │   │   ├── ClassDistribution.jsx   # Interactive class distribution bar & metrics table
│   │   │   └── Legend.jsx              # Thematic class color legend
│   │   ├── services/
│   │   │   └── api.js                  # Frontend API client for FastAPI
│   │   ├── App.jsx                     # Application root with dynamic state management
│   │   ├── App.css                     # Component styling & dark glassmorphic design
│   │   └── index.css                   # Global theme tokens & typography
│   ├── package.json                    # Node dependencies & Vite scripts
│   └── vite.config.js                  # Vite configuration
│
├── dataset/
│   └── LULCzip/LULC/
│       ├── input/                      # Sentinel-2 input tiles (.tif)
│       └── labels/                     # Matching ground-truth label masks (.tif)
│
├── .gitattributes                      # Git LFS tracking configuration for large model files
├── .gitignore                          # Git ignore rules
└── README.md                           # Project documentation (this file)
```

---

## 🚀 Setup & Local Execution

### Prerequisites
- **Python 3.10+**
- **Git LFS** (for pulling the 458MB pixel-level model)
- **Node.js 18+** & **npm**

---

### 1. Clone & Pull Git LFS Objects

```bash
git clone https://github.com/sarthak-panda-18/LandUseLandCover.git
cd LandUseLandCover
git lfs pull
```

---

### 2. Backend Setup (FastAPI & ML Inference Engines)

```bash
# 1. Navigate to backend directory
cd backend

# 2. Create and activate a Python virtual environment
python -m venv venv

# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
# source venv/bin/activate

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Start the FastAPI backend server
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

- Backend API: `http://127.0.0.1:8000`
- Interactive Swagger UI: `http://127.0.0.1:8000/docs`
- Health Check: `http://127.0.0.1:8000/health`

---

### 3. Frontend Setup (React & Vite Dashboard)

```bash
# 1. Navigate to frontend directory
cd frontend

# 2. Install npm dependencies
npm install

# 3. Start the Vite development server
npm run dev
```

- Interactive Web Dashboard: `http://localhost:5173/`

---

## 📊 Model Performance & Comparative Evaluation

### Comparative Model Summary

| Metric / Dimension | Pixel-Level RF (`pixel_rf`) | Patch-Level RF (`rf_patch`) | Patch-Level EfficientNet (`efficientnet_patch`) |
| :--- | :---: | :---: | :---: |
| **Spatial Detail** | **10m / pixel** (Full detail) | 64x64 block (Coarse) | 64x64 block (Coarse) |
| **Input Representation** | 5 spectral bands (per pixel) | 25 statistical features / block | 3-channel RGB image / block |
| **Overall Accuracy** | **72.84%** | **89.02%** | **87.20%** |
| **Cohen's Kappa ($\kappa$)** | **0.5990** | **0.85** | **0.83** |
| **Macro Average F1** | **64.76%** | **67.00%** | **64.00%** |
| **Weighted Average F1** | **73.80%** | **86.00%** | **85.00%** |
| **Test Sample Support** | **1,923,589 pixels** | **328 blocks** | **328 blocks** |
| **Inference Latency** | ~53.0 s / tile | **~0.4 s / tile** ⚡ | ~10.5 s / tile |
| **Grassland F1** | **8.61%** | **0.00%** (Failure) | **0.00%** (Failure) |

---

### Detailed Per-Class Breakdown

#### 1. Pixel-Level Random Forest (1,923,589 Test Pixels)
| Class ID | Class Name | Precision | Recall | F1-Score | Support (Pixels) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **0** | Water | 88.04% | 86.44% | **87.23%** | 139,986 |
| **1** | Trees / Forest | 58.10% | 71.27% | **64.02%** | 326,955 |
| **2** | Crops | 81.23% | 73.35% | **77.09%** | 994,202 |
| **3** | Grassland | 5.15% | 26.22% | **8.61%** | 10,632 |
| **4** | Built-up | 72.64% | 66.84% | **69.62%** | 400,966 |
| **5** | Bare land | 73.48% | 92.82% | **82.03%** | 50,848 |

#### 2. Patch-Level Random Forest & EfficientNetB0 (328 Test Blocks)
| Class ID | Class Name | RF Patch F1 | EfficientNet F1 | Note |
| :---: | :--- | :---: | :---: | :--- |
| **0** | Water | **94%** | **92%** | High block-level contrast on water bodies |
| **1** | Trees / Forest | **88%** | **85%** | Robust vegetation clustering |
| **2** | Crops | **89%** | **87%** | Dominant regional agricultural land use |
| **3** | Grassland | **0.00%** ⚠️ | **0.00%** ⚠️ | Complete failure due to dataset imbalance |
| **4** | Built-up | **84%** | **82%** | Urban cluster delineation |
| **5** | Bare land | **92%** | **90%** | Distinct high-reflectance spectral signature |

---

## ⚠️ Known Limitations & Model Behavior

1. **Grassland Detection Failure in Patch Models ($F_1 = 0.00$)**:
   - The patch dataset contains severe regional class imbalance with **only 12 total Grassland blocks** across the entire dataset.
   - Consequently, both patch models (`rf_patch` and `efficientnet_patch`) fail completely on Grassland ($F_1 = 0.00$, Precision = 0.00, Recall = 0.00), misclassifying all test Grassland blocks into the dominant **Crops** class.
   - In contrast, the pixel-level model evaluated on 10,632 test Grassland pixels achieves $F_1 = 8.61\%$.

2. **Evaluation Sample Support & Variance**:
   - The patch models report high overall accuracy (87–89%), but were evaluated on a small test split of **328 blocks** (64x64). This carries higher statistical variance compared to the rigorous **1,923,589 pixel** test evaluation conducted for the pixel-level Random Forest model.

3. **Spatial Granularity & Blockiness**:
   - Patch models classify entire 64x64 pixel areas ($640\text{m} \times 640\text{m}$) into a single class. This causes coarse, step-like boundaries along narrow river channels, roads, and mixed urban-rural margins.
   - While visual Gaussian smoothing softens the display output in the UI, it is strictly cosmetic and does not recover true pixel-level sub-block boundaries.

4. **Memory-Safe Regularization for Pixel Trees**:
   - Unconstrained tree depth on 8.97M training pixels across parallel CPU workers requires substantial memory. The pixel model was optimized with `min_samples_leaf=3` and `max_samples=0.35` (~3.14M samples/tree bootstrap), enabling robust full-depth tree growth within standard memory limits.

---

## 📡 API Reference

### 1. `POST /api/classify`
Uploads a Sentinel-2 GeoTIFF tile, runs inference using the selected model, and returns the colorized LULC map with per-class statistics.
- **Request**: Multipart form data:
  - `file`: GeoTIFF file (`.tif`, `.tiff`, up to 200MB)
  - `model`: Optional form field (`pixel_rf`, `rf_patch`, or `efficientnet_patch`; default: `pixel_rf`)
- **Response** (`200 OK`):
```json
{
  "success": true,
  "filename": "Vijayawada_LULC_Input_Tile.tif",
  "model_used": "rf_patch",
  "processing_time_seconds": 0.38,
  "dimensions": { "height": 1294, "width": 915 },
  "total_pixels": 1184010,
  "valid_pixels": 1184010,
  "invalid_pixels": 0,
  "class_distribution": [
    {
      "class_id": 0,
      "class_name": "Water",
      "color_hex": "#1E88E5",
      "pixel_count": 36864,
      "percentage_valid": 3.11
    }
  ],
  "classified_image_base64": "data:image/png;base64,iVBORw0KGgo...",
  "rgb_preview_base64": "data:image/png;base64,iVBORw0KGgo..."
}
```

### 2. `GET /api/model-info`
Returns metadata, hyperparameters, and evaluation metrics for the selected model.
- **Parameters**: `?model=pixel_rf` | `rf_patch` | `efficientnet_patch`
- **Response** (`200 OK`):
```json
{
  "model_id": "rf_patch",
  "model_name": "Random Forest (64x64 Patch)",
  "granularity": "64x64 Block (Patch-level)",
  "test_metrics": {
    "overall_accuracy": 0.8902,
    "cohen_kappa": 0.85,
    "macro_avg_f1": 0.67,
    "weighted_avg_f1": 0.86,
    "grassland_f1": 0.00,
    "test_patches_count": 328
  }
}
```

### 3. `GET /api/legend`
Returns the 6-class visual color palette legend image (`image/png`).

### 4. `GET /api/sample-tiles`
Lists bundled sample Sentinel-2 GeoTIFF tiles for rapid browser testing.

---

## 🧪 Testing

To run the full multi-model test suite:

```bash
cd backend
python test_all_three_models.py
python test_phase8_full_suite.py
```
