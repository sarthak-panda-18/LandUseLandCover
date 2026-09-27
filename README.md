# Sentinel-2 Land Use & Land Cover (LULC) Classification System

An end-to-end Machine Learning web application for pixel-level Land Use and Land Cover (LULC) classification using 10m Sentinel-2 multispectral satellite imagery, powered by a Random Forest classifier and an interactive React dashboard.

---

## 📌 Project Overview

This project implements an automated Land Use / Land Cover classification pipeline for the **Vijayawada, India** agricultural-urban region using multi-temporal **Sentinel-2** satellite data derived from **Google Earth Engine (GEE)** and benchmarked against **ESRI 10m Annual Land Cover** ground truth data.

The system classifies every valid pixel into one of **6 thematic LULC classes**:
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

## 📂 Repository Structure

```text
lulc-project/
├── backend/
│   ├── app/
│   │   ├── main.py                     # FastAPI application, CORS & lifespan startup loader
│   │   ├── routes/
│   │   │   └── classify.py             # Endpoints: /api/classify, /api/model-info, /api/legend
│   │   ├── services/
│   │   │   └── model_service.py        # In-memory Random Forest inference & scaler service
│   │   └── utils/
│   │       └── raster_utils.py         # GeoTIFF raster parsing & RGBA color-mapping
│   ├── ml/
│   │   ├── models/
│   │   │   ├── lulc_rf_model.joblib    # Trained Random Forest classifier (300 trees, depth 25)
│   │   │   ├── scaler.joblib           # Fitted StandardScaler
│   │   │   └── model_metadata.json     # Hyperparameters, feature importances & class mapping
│   │   ├── outputs/
│   │   │   ├── metrics/test_metrics.json # Test metrics (Accuracy, F1, Cohen's Kappa)
│   │   │   └── maps/legend.png         # Rendered class palette legend
│   │   └── src/
│   │       ├── preprocessing/          # Tile cleaning, nodata filtering & dataset builder
│   │       ├── training/               # Grid search & Random Forest training pipeline
│   │       └── evaluation/             # Test evaluation & map reconstruction
│   ├── requirements.txt                # Python backend dependencies
│   ├── test_api_endpoints.py           # Unit & endpoint test script
│   └── test_phase8_full_suite.py       # Full 6-tile integration test suite
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── UploadPanel.jsx         # Drag-and-drop GeoTIFF uploader with radar pulse animation
│   │   │   ├── ClassifiedMapView.jsx   # Multi-view map visualizer (LULC, RGB, Side-by-Side)
│   │   │   ├── ClassBreakdownChart.jsx # Multi-segment distribution bar & per-class stats
│   │   │   └── ModelStatsPanel.jsx     # Model performance telemetry & feature importances
│   │   ├── pages/
│   │   │   └── Dashboard.jsx           # Main unified dashboard view
│   │   ├── services/
│   │   │   └── api.js                  # Frontend API client for FastAPI
│   │   ├── App.jsx                     # Application root
│   │   ├── App.css                     # Component styling & dark glassmorphic design
│   │   └── index.css                   # Global theme tokens & typography
│   ├── package.json                    # Node dependencies & Vite scripts
│   └── vite.config.js                  # Vite configuration
│
├── dataset/
│   └── LULCzip/LULC/
│       ├── input/                      # 6 Sentinel-2 input tiles (.tif)
│       └── labels/                     # Matching ESRI ground truth masks (.tif)
│
└── README.md                           # Project documentation (this file)
```

---

## 🚀 Setup & Local Execution

### Prerequisites
- **Python 3.10+**
- **Node.js 18+** & **npm**

---

### 1. Backend Setup (FastAPI & ML Inference Engine)

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

### 2. Frontend Setup (React & Vite Dashboard)

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

## 📊 Model Performance & Evaluation

The Random Forest model was trained on **8,976,744 pixels** and evaluated on an independent test split of **1,923,589 pixels**.

### Overall Test Metrics

| Metric | Score | Note |
| :--- | :---: | :--- |
| **Overall Test Accuracy** | **72.84%** 🟢 | Exact pixel classification across 6 classes (+7.24% improvement) |
| **Cohen's Kappa ($\kappa$)** | **0.5990** 🟢 | Substantial statistical agreement above chance (near 0.60) |
| **Weighted Average F1** | **73.80%** 🟢 | Support-weighted mean reflecting true spatial distribution |
| **Macro Average F1** | **64.76%** 🟢 | Unweighted mean across all 6 classes |

### Per-Class Test Breakdown

| Class ID | Class Name | Precision | Recall | F1-Score | Test Support (Pixels) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **0** | Water | 88.04% | 86.44% | **87.23%** | 139,986 |
| **1** | Trees / Forest | 58.10% | 71.27% | **64.02%** | 326,955 |
| **2** | Crops | 81.23% | 73.35% | **77.09%** 🟢 | 994,202 |
| **3** | Grassland | 5.15% | 26.22% | **8.61%** | 10,632 |
| **4** | Built-up | 72.64% | 66.84% | **69.62%** | 400,966 |
| **5** | Bare land | 73.48% | 92.82% | **82.03%** 🟢 | 50,848 |

### Spectral Feature Importances

1. **Band 4 (Red)**: `23.19%` (Crucial for separating vegetation from bare soil and urban surfaces)
2. **Band 8 (NIR)**: `22.08%` (Key for canopy reflectance and water boundary delineation)
3. **Band 3 (Green)**: `19.78%` (Chlorophyll absorption and vegetation vigour)
4. **Band 2 (Blue)**: `18.61%` (Water clarity and atmospheric scattering contrast)
5. **NDVI**: `16.34%` (Normalized vegetative biomass density)

---

## 📡 API Reference

### 1. `POST /api/classify`
Uploads a Sentinel-2 GeoTIFF tile, runs inference, and returns classified map and statistics.
- **Request**: Multipart form data with file field `file` (`.tif` or `.tiff`, up to 200MB).
- **Response** (`200 OK`):
```json
{
  "success": true,
  "filename": "Vijayawada_LULC_Input_2025-0000002048-0000004096.tif",
  "file_size_mb": 29.62,
  "processing_time_seconds": 21.76,
  "dimensions": { "height": 1294, "width": 915 },
  "total_pixels": 1184010,
  "valid_pixels": 1184010,
  "invalid_pixels": 0,
  "valid_percentage": 100.0,
  "masked_percentage": 0.0,
  "class_distribution": [
    {
      "class_id": 0,
      "class_name": "Water",
      "color_hex": "#1E88E5",
      "pixel_count": 35361,
      "percentage_valid": 2.99,
      "percentage_total": 2.99
    },
    {
      "class_id": 2,
      "class_name": "Crops",
      "color_hex": "#81C784",
      "pixel_count": 443543,
      "percentage_valid": 37.46,
      "percentage_total": 37.46
    }
  ],
  "classified_image_base64": "data:image/png;base64,iVBORw0KGgo...",
  "rgb_preview_base64": "data:image/png;base64,iVBORw0KGgo..."
}
```

### 2. `GET /api/model-info`
Returns model training hyperparameters, feature importances, and evaluation metrics.
- **Response** (`200 OK`):
```json
{
  "model_type": "RandomForestClassifier",
  "training_samples": 8976744,
  "validation_samples": 1923588,
  "hyperparameters": {
    "n_estimators": 300,
    "max_depth": null,
    "min_samples_leaf": 3,
    "max_samples": 0.35,
    "class_weight": "balanced",
    "random_state": 42
  },
  "test_metrics": {
    "overall_accuracy": 0.7284,
    "cohen_kappa": 0.5990,
    "macro_avg_f1": 0.6476,
    "weighted_avg_f1": 0.7380
  }
}
```

### 3. `GET /api/legend`
Returns the visual class palette legend image.
- **Response** (`200 OK`): `image/png` binary stream.

### 4. `GET /health`
Returns backend service health and model warmup status.
- **Response** (`200 OK`):
```json
{
  "status": "ok",
  "model_loaded": true,
  "service": "LULC Classification Backend",
  "version": "1.0.0"
}
```

---

## ⚠️ Known Limitations & Model Behavior

1. **Extreme Class Imbalance (93.5× Ratio)**:
   - The dataset reflects real-world spatial distributions where agricultural **Crops** (994,202 test pixels) heavily dominate over sparse **Grassland** (10,632 test pixels).
   - Although `class_weight='balanced'` and unlimited tree depth (`max_depth=None`) improved precision from $2.74\%$ to **$5.15\%$**, Grassland F1 remains modest (**8.61%**) due to single-date spectral ambiguity between pasture/grazing land and fallow cropland pixels.

2. **Memory-Safe Regularization for Full-Depth Trees**:
   - Pure unconstrained trees (`max_depth=None`, `min_samples_leaf=1`, `max_samples=None`) on 8.97M training pixels across multiple CPU cores exhausted 16GB RAM during parallel training. The model was optimized with `min_samples_leaf=3` and `max_samples=0.35` (~3.14M samples/tree bootstrap), enabling full-depth unconstrained branching within system memory limits.

3. **Pixel-Level vs. Contextual Spatial Models**:
   - The Random Forest model classifies pixels independently based on spectral reflectance and NDVI. Future extensions can incorporate spatial neighborhood context using Convolutional Neural Networks (e.g., U-Net, DeepLabV3+) or Vision Transformers (ViT) to capture textural and geometric patterns.


---

## 🧪 Integration Testing

To execute the automated end-to-end integration test suite across all 6 tiles:

```bash
cd backend
.\venv\Scripts\python test_phase8_full_suite.py
```
