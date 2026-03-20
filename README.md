# 🌦️ Nimbus — Weather Intelligence Web App
### ELE 4409 Mini Project

## Project Structure
```
weather_app/
├── main.py              ← FastAPI backend
├── index.html           ← Web frontend
├── requirements.txt     ← Python dependencies
│
│   (copy from Colab after training)
├── best_model.pkl
├── scaler_img.pkl
├── scaler_tab.pkl
├── label_encoder.pkl
├── shap_background.pkl
└── model_meta.json
```

## Setup (one time)

### Step 1 — Copy model files
Copy these 6 files from your Google Drive (weather_model_artifacts/) into this folder:
- best_model.pkl
- scaler_img.pkl
- scaler_tab.pkl
- label_encoder.pkl
- shap_background.pkl
- model_meta.json

### Step 2 — Get WeatherAPI key (free)
1. Go to https://www.weatherapi.com → Sign up free
2. Copy your API key
3. Open main.py → replace YOUR_WEATHERAPI_KEY on line 27

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Run the backend
```bash
uvicorn main:app --reload --port 8000
```

### Step 5 — Open the frontend
Open index.html directly in your browser (double-click it).

---

## How it works

### City Mode (🏙)
1. User types any city name
2. FastAPI fetches live weather from WeatherAPI.com
3. Builds 11 tabular features (8 raw + 3 engineered)
4. Runs through trained KNN model
5. Computes SHAP explanations
6. Returns prediction + confidence + live readings

### Image Mode (🖼)
1. User uploads a sky photograph
2. FastAPI extracts 39 visual features using OpenCV + GLCM
   (brightness, saturation, histograms, cloud coverage, texture, edges)
3. Runs through trained KNN model
4. Computes SHAP explanations
5. Returns prediction + confidence + extracted feature values

---

## API Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/model-info` | Model metadata |
| POST | `/predict/city` | Predict from city name (form: city=Chennai) |
| POST | `/predict/image` | Predict from sky photo (multipart: file=image) |
