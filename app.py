from flask import Flask, render_template, request
import pandas as pd
import joblib
import os
import shap
import numpy as np
from utils.feature_extraction import extract_features

app = Flask(__name__)

# ==========================================
# 1. LOAD ASSETS
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_model_assets():
    try:
        model = joblib.load(os.path.join(BASE_DIR, "xgboost_phishing_model.pkl"))
        feature_order = joblib.load(os.path.join(BASE_DIR, "feature_columns.pkl"))
        explainer = joblib.load(os.path.join(BASE_DIR, "shap_explainer.pkl"))
        return model, feature_order, explainer
    except Exception as e:
        print(f"Error Loading Assets: {e}")
        return None, None, None

model, FEATURE_ORDER, explainer = load_model_assets()

# ==========================================
# 2. WHITELIST LOGIC
# ==========================================
WHITELIST_DOMAINS = ['facebook.com', 'google.com', 'instagram.com', 'microsoft.com', 'apple.com', 'github.com', 'unisayogya.ac.id']

def check_whitelist(url):
    url_lower = url.lower()
    for domain in WHITELIST_DOMAINS:
        if domain in url_lower:
            return True
    return False

# ==========================================
# ROUTES
# ==========================================
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    url = request.form.get("url", "").strip()
    if not url: return "Silakan masukkan URL!"

    # --- JALUR CEPAT: WHITELIST ---
    if check_whitelist(url):
        return render_template(
            "result.html",
            url=url,
            label="Aman",
            confidence=0.01,
            shap_values={}, 
            status_color="success"
        )

    try:
        # 1. Ekstraksi & Sinkronisasi
        features = extract_features(url)
        X = pd.DataFrame([features])
        X = X[FEATURE_ORDER] 

        # 2. Prediksi
        proba = model.predict_proba(X)[0][1]
        confidence = round(float(proba) * 100, 2)

        # 3. Thresholding
        if proba >= 0.9:
            label, status_color = "Phishing", "danger"
        elif proba >= 0.5:
            label, status_color = "Suspicious (Perlu Dicek)", "warning"
        else:
            label, status_color = "Aman", "success"

        # 4. SHAP (Explainable AI)
        shap_values_raw = explainer.shap_values(X)
        
        # Logika pengambilan nilai SHAP untuk XGBoost (Binary Classification)
        if hasattr(shap_values_raw, "shape"):
            if len(shap_values_raw.shape) == 2:
                s_val = shap_values_raw[0]
            else:
                s_val = shap_values_raw
        else:
            s_val = shap_values_raw[0] if isinstance(shap_values_raw, list) else shap_values_raw

        # 5. MAPPING KE BAHASA INDONESIA
        feature_mapping = {
            "UrlLength": "Panjang URL",
            "HostnameLength": "Panjang Nama Domain",
            "PathLength": "Panjang Folder/Path",
            "NumDots": "Jumlah Titik (.)",
            "NumDash": "Jumlah Tanda Hubung (-)",
            "NumNumericChars": "Jumlah Karakter Angka",
            "AtSymbol": "Penggunaan Simbol @",
            "HttpsInHostname": "HTTPS Palsu di Domain",
            "IsIpAddress": "Menggunakan Alamat IP",
            "SubdomainLevel": "Tingkat Subdomain",
            "AbnormalSubdomain": "Subdomain Mencurigakan",
            "IsAcademicDomain": "Domain Akademik (.ac.id)",
            "NumSensitiveWords": "Jumlah Kata Sensitif"
        }
        # Olah data SHAP menjadi dictionary terjemahan
        shap_dict = {}
        for feat, val in zip(FEATURE_ORDER, s_val):
            if "Unnamed" not in feat:
                nama_indo = feature_mapping.get(feat, feat)
                shap_dict[nama_indo] = round(float(val), 4)

        # Urutkan berdasarkan pengaruh absolut terbesar
        sorted_shap = dict(sorted(shap_dict.items(), key=lambda item: abs(item[1]), reverse=True))

        # 6. KIRIM KE RESULT (Hanya satu return di sini)
        return render_template(
            "result.html",
            url=url,
            label=label,
            confidence=confidence,
            shap_values=sorted_shap,
            status_color=status_color
        )

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return f"Sistem error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)