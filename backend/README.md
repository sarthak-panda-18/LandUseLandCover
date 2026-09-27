# LULC Classification - Backend & ML Engine

FastAPI backend service and machine learning pipeline for Land Use and Land Cover (LULC) classification based on Sentinel-2 satellite imagery.

---

## Directory Structure

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application with health check
│   ├── routes/                 # API endpoints (to be implemented)
│   ├── services/               # Inference & business logic (to be implemented)
│   └── utils/                  # Raster & satellite helpers (to be implemented)
├── ml/
│   ├── data/
│   │   ├── input/              # 5-band Sentinel-2 GeoTIFF tiles
│   │   └── labels/             # Land-cover ground-truth masks
│   ├── notebooks/              # Jupyter notebooks for data analysis & experiments
│   ├── src/
│   │   ├── preprocessing/      # Data normalization, feature engineering
│   │   ├── training/           # Random Forest model training logic
│   │   └── evaluation/         # Accuracy, Confusion Matrix, Classification Reports
│   └── models/                 # Saved model binaries (.joblib)
├── requirements.txt            # Python dependencies
└── README.md                   # This documentation
```

---

## Setup & Execution

### 1. Create Virtual Environment
```bash
python -m venv venv
```

### 2. Activate Virtual Environment
- **Windows (PowerShell)**:
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```
- **Windows (CMD)**:
  ```cmd
  .\venv\Scripts\activate.bat
  ```
- **macOS / Linux**:
  ```bash
  source venv/bin/activate
  ```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Development Server
```bash
uvicorn app.main:app --reload --port 8000
```

### 5. Health Check
Open browser or send GET request:
`http://localhost:8000/health`
Expected response:
```json
{"status": "ok"}
```
