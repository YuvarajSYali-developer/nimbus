"""
🌦️ Multimodal Weather Classifier — FastAPI Backend
ELE 4409 Mini Project

Run:
    pip install fastapi uvicorn joblib scikit-learn shap numpy requests pandas opencv-python scikit-image Pillow python-multipart
    uvicorn main:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import joblib, json, numpy as np, pandas as pd
import requests, os, shap, cv2, warnings
from io import BytesIO
from PIL import Image
from skimage.feature import graycomatrix, graycoprops
warnings.filterwarnings("ignore")

BASE = os.path.dirname(__file__)

def load(name):
    path = os.path.join(BASE, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name} not found in {BASE}")
    return joblib.load(path)

# ── Load artifacts ────────────────────────────────────────────────────────────
try:
    model      = load("best_model.pkl")
    scaler_img = load("scaler_img.pkl")
    scaler_tab = load("scaler_tab.pkl")
    le         = load("label_encoder.pkl")
    background = load("shap_background.pkl")

    with open(os.path.join(BASE, "model_meta.json")) as f:
        meta = json.load(f)

    IMG_FEATURE_COLS = meta["img_feature_cols"]
    TAB_FEATURE_COLS = meta["tab_feature_cols"]
    ALL_FEATURE_COLS = meta["all_feature_cols"]
    CLASSES          = meta["classes"]
    IS_COMBINED      = meta["is_combined"]

    # ── Determine which features the best model expects ───────────────────────
    model_name_lower = meta['model_name'].lower()
    if 'image only' in model_name_lower:
        MODEL_FEATURE_COLS = IMG_FEATURE_COLS
        print(f"ℹ️  Best model is Image-Only → expects {len(IMG_FEATURE_COLS)} features")
    elif 'tabular only' in model_name_lower:
        MODEL_FEATURE_COLS = TAB_FEATURE_COLS
        print(f"ℹ️  Best model is Tabular-Only → expects {len(TAB_FEATURE_COLS)} features")
    else:
        MODEL_FEATURE_COLS = ALL_FEATURE_COLS
        print(f"ℹ️  Best model is Combined → expects {len(ALL_FEATURE_COLS)} features")

    # ── Rebuild SHAP background to match model's expected feature count ───────
    print("🔄 Building SHAP explainer...")
    bg_cols = list(background.columns) if hasattr(background, 'columns') else []

    if len(bg_cols) != len(MODEL_FEATURE_COLS):
        print(f"⚠️  Background has {len(bg_cols)} features but model expects {len(MODEL_FEATURE_COLS)}")
        print("   Rebuilding background with correct feature shape (zeros)...")
        bg_data = pd.DataFrame(
            np.zeros((50, len(MODEL_FEATURE_COLS))),
            columns=MODEL_FEATURE_COLS
        )
    else:
        bg_data = background[MODEL_FEATURE_COLS]

    explainer = shap.KernelExplainer(model.predict_proba, bg_data)
    print(f"✅ Backend ready — model: {meta['model_name']} | accuracy: {meta['accuracy']}")

except Exception as e:
    raise RuntimeError(f"❌ Startup failed.\n{e}")

WEATHER_API_KEY = "b7feeadc4e1e4358a35110705262003"   # 🔁 Replace with your key from weatherapi.com
WEATHER_API_URL = "http://api.weatherapi.com/v1/current.json"
EMOJI = {"sunny":"☀️","rain":"🌧️","cloudy":"☁️","fog":"🌫️","snow":"❄️"}

app = FastAPI(title="🌦️ Weather Classifier", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Image feature extraction (mirrors notebook exactly) ───────────────────────
def extract_image_features(img_array: np.ndarray, img_size=(128,128)) -> dict:
    img_bgr  = cv2.resize(img_array, img_size)
    img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_hsv  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    f = {}
    f['brightness_mean'] = float(np.mean(img_hsv[:,:,2]))
    f['brightness_std']  = float(np.std(img_hsv[:,:,2]))
    f['saturation_mean'] = float(np.mean(img_hsv[:,:,1]))
    f['saturation_std']  = float(np.std(img_hsv[:,:,1]))
    f['red_mean']        = float(np.mean(img_rgb[:,:,0]))
    f['green_mean']      = float(np.mean(img_rgb[:,:,1]))
    f['blue_mean']       = float(np.mean(img_rgb[:,:,2]))
    f['blue_dominance']  = f['blue_mean'] - f['red_mean']

    for ch_idx, ch_name in enumerate(['r','g','b']):
        hist = cv2.calcHist([img_rgb],[ch_idx],None,[8],[0,256]).flatten()
        hist = hist / (hist.sum() + 1e-7)
        for i, v in enumerate(hist):
            f[f'hist_{ch_name}_{i}'] = float(v)

    _, cloud_mask = cv2.threshold(img_gray, 180, 255, cv2.THRESH_BINARY)
    f['cloud_coverage'] = float(np.sum(cloud_mask > 0)) / (img_size[0]*img_size[1])
    _, dark_mask = cv2.threshold(img_gray, 60, 255, cv2.THRESH_BINARY_INV)
    f['dark_ratio'] = float(np.sum(dark_mask > 0)) / (img_size[0]*img_size[1])

    glcm = graycomatrix(img_gray.astype(np.uint8), distances=[5], angles=[0],
                        levels=256, symmetric=True, normed=True)
    f['texture_contrast']    = float(graycoprops(glcm,'contrast')[0,0])
    f['texture_correlation'] = float(graycoprops(glcm,'correlation')[0,0])
    f['texture_homogeneity'] = float(graycoprops(glcm,'homogeneity')[0,0])
    f['texture_energy']      = float(graycoprops(glcm,'energy')[0,0])

    edges = cv2.Canny(img_gray, 50, 150)
    f['edge_density'] = float(np.sum(edges > 0)) / (img_size[0]*img_size[1])
    return f

def build_tab_features(w: dict) -> np.ndarray:
    heat_index  = w["temperature"] - (0.55 - 0.0055*w["humidity"]) * (w["temperature"] - 14.5)
    temp_spread = w["temperature"] - w["feels_like"]
    hum_wind    = w["humidity"] * w["wind"] / 100
    return np.array([[
        w["precipitation"], w["temperature"], w["feels_like"],
        w["humidity"], w["wind"], w["pressure"], w["visibility"], w["uv_index"],
        heat_index, temp_spread, hum_wind
    ]])

def get_shap(feat_df: pd.DataFrame, pred_idx: int):
    try:
        sv   = explainer.shap_values(feat_df, nsamples=80)
        cs   = sv[pred_idx][0]
        cols = list(feat_df.columns)
        contribs = sorted([
            {"feature": cols[i].replace("_"," "),
             "shap_value": round(float(cs[i]),4),
             "direction": "increases" if cs[i]>0 else "decreases",
             "modality": "image" if cols[i] in IMG_FEATURE_COLS else "sensor"}
            for i in range(len(cols))
        ], key=lambda x: abs(x["shap_value"]), reverse=True)[:6]

        img_s = sum(abs(c["shap_value"]) for c in contribs if c["modality"]=="image")
        tab_s = sum(abs(c["shap_value"]) for c in contribs if c["modality"]=="sensor")
        tot   = img_s + tab_s + 1e-9
        return {"top_features": contribs,
                "image_pct": round(img_s/tot*100,1),
                "sensor_pct": round(tab_s/tot*100,1)}
    except Exception as ex:
        return {"top_features":[], "image_pct":0, "sensor_pct":0, "error":str(ex)}

def fetch_weather(city: str) -> dict:
    try:
        r = requests.get(WEATHER_API_URL,
                         params={"key":WEATHER_API_KEY,"q":city,"aqi":"no"},
                         timeout=10)
        r.raise_for_status()
        d = r.json(); c = d["current"]; l = d["location"]
        return {
            "city":l["name"],"country":l["country"],"local_time":l["localtime"],
            "precipitation":c["precip_mm"],"temperature":c["temp_c"],
            "feels_like":c["feelslike_c"],"humidity":c["humidity"],
            "wind":c["wind_kph"],"pressure":c["pressure_mb"],
            "visibility":c["vis_km"],"uv_index":c["uv"],
            "condition_raw":c["condition"]["text"],
            "wind_dir":c["wind_dir"],"cloud_pct":c["cloud"],
        }
    except requests.exceptions.HTTPError as e:
        code = e.response.status_code if e.response else 0
        if code == 400:
            raise HTTPException(404, f"City '{city}' not found.")
        raise HTTPException(502, str(e))
    except Exception as e:
        raise HTTPException(502, f"Weather fetch failed: {e}")

def make_prediction(feat_df: pd.DataFrame):
    pred_idx   = int(model.predict(feat_df.values)[0])
    pred_label = le.inverse_transform([pred_idx])[0]
    proba      = model.predict_proba(feat_df.values)[0]
    scores     = {CLASSES[i]: round(float(proba[i])*100,1) for i in range(len(CLASSES))}
    shap_res   = get_shap(feat_df, pred_idx)
    return pred_label, round(float(proba[pred_idx])*100,1), scores, shap_res

# ── Routes ────────────────────────────────────────────────────────────────────
from fastapi.responses import FileResponse

@app.get("/")
def serve_frontend():
    return FileResponse(os.path.join(BASE, "index.html"))

@app.get("/health")
def health():
    return {
        "status": "running",
        "model": meta["model_name"],
        "accuracy": meta["accuracy"]
    }


@app.get("/model-info")
def model_info():
    return {**meta,"model_feature_count":len(MODEL_FEATURE_COLS)}

@app.post("/predict/city")
async def predict_city(city: str = Form(...)):
    """Predict using live sensor data from WeatherAPI."""
    w       = fetch_weather(city)
    tab_vec = build_tab_features(w)
    tab_s   = scaler_tab.transform(tab_vec)

    # Build feature vector matching what the model expects
    if 'image only' in meta['model_name'].lower():
        # Image-only model: use zero image features (no photo provided)
        img_s   = np.zeros((1, len(IMG_FEATURE_COLS)))
        feat_df = pd.DataFrame(img_s, columns=IMG_FEATURE_COLS)
        note    = "City mode uses sensor data. For best image-only results, upload a sky photo."
    elif 'tabular only' in meta['model_name'].lower():
        feat_df = pd.DataFrame(tab_s, columns=TAB_FEATURE_COLS)
        note    = ""
    else:
        img_s    = np.zeros((1, len(IMG_FEATURE_COLS)))
        combined = np.hstack([img_s, tab_s])
        feat_df  = pd.DataFrame(combined, columns=ALL_FEATURE_COLS)
        note     = ""

    pred, conf, scores, shap_res = make_prediction(feat_df)
    top_feat = shap_res["top_features"][0]["feature"] if shap_res["top_features"] else "weather features"

    return JSONResponse({
        "mode":"city",
        "city":w["city"],"country":w["country"],"local_time":w["local_time"],
        "prediction":pred,"emoji":EMOJI.get(pred,"🌤️"),"confidence":conf,
        "confidence_scores":scores,
        "explanation":f"Prediction driven mainly by {top_feat}.",
        "shap":shap_res,
        "note":note,
        "live_readings":{k:v for k,v in w.items() if k not in ["city","country","local_time"]},
        "model_used":meta["model_name"],
    })

@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    """Predict from an uploaded sky photo using image feature extraction."""
    contents = await file.read()
    img_pil  = Image.open(BytesIO(contents)).convert("RGB")
    img_bgr  = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    feats   = extract_image_features(img_bgr)
    img_vec = np.array([[feats[c] for c in IMG_FEATURE_COLS]])
    img_s   = scaler_img.transform(img_vec)

    if 'image only' in meta['model_name'].lower():
        feat_df = pd.DataFrame(img_s, columns=IMG_FEATURE_COLS)
    elif 'tabular only' in meta['model_name'].lower():
        # tabular-only can't use image, use zeros
        feat_df = pd.DataFrame(np.zeros((1, len(TAB_FEATURE_COLS))), columns=TAB_FEATURE_COLS)
    else:
        tab_zeros = np.zeros((1, len(TAB_FEATURE_COLS)))
        feat_df   = pd.DataFrame(np.hstack([img_s, tab_zeros]), columns=ALL_FEATURE_COLS)

    pred, conf, scores, shap_res = make_prediction(feat_df)
    top_feat = shap_res["top_features"][0]["feature"] if shap_res["top_features"] else "image features"

    return JSONResponse({
        "mode":"image",
        "filename":file.filename,
        "prediction":pred,"emoji":EMOJI.get(pred,"🌤️"),"confidence":conf,
        "confidence_scores":scores,
        "explanation":f"Sky image classified mainly by {top_feat}.",
        "shap":shap_res,
        "image_features":{
            "brightness_mean": round(feats["brightness_mean"],2),
            "saturation_mean": round(feats["saturation_mean"],2),
            "cloud_coverage":  round(feats["cloud_coverage"],3),
            "dark_ratio":      round(feats["dark_ratio"],3),
            "blue_dominance":  round(feats["blue_dominance"],2),
            "edge_density":    round(feats["edge_density"],3),
        },
        "model_used":meta["model_name"],
    })
