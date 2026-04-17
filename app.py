from flask import Flask, render_template, request
import pandas as pd
import joblib
import numpy as np
import os

# Import fungsi ekstraksi fitur buatanmu
# Pastikan file ini ada di folder utils/feature_extraction.py
from utils.feature_extraction import extract_features

app = Flask(__name__)

# ==========================================
# 1. LOAD ASSETS (MODEL & EXPLAINER)
# ==========================================
# Gunakan path yang aman agar tidak error saat dideploy
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "xgboost_phishing_model.pkl"))

# Load explainer. Jika tadi kamu simpan dengan nama 'shap_explainer.pkl'
try:
    explainer = joblib.load(os.path.join(BASE_DIR, "shap_explainer.pkl"))
except:
    # Fallback jika file belum ada, buat explainer baru dari model
    import shap
    explainer = shap.TreeExplainer(model)

# Load urutan fitur agar sinkron dengan model (Sangat Penting!)
try:
    FEATURE_ORDER = joblib.load(os.path.join(BASE_DIR, "feature_columns.pkl"))
except:
    # Manual fallback jika file pkl tidak ada
    FEATURE_ORDER = [
        "NumDots", "UrlLength", "NumDash", "AtSymbol", "IpAddress", 
        "HttpsInHostname", "PathLevel", "PathLength", "NumNumericChars"
    ]

# ==========================================
# ROUTES
# ==========================================
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    url = request.form.get("url", "").strip()
    
    if not url:
        return "Silakan masukkan URL!"

    try:
        # 1. Ekstraksi Fitur
        features = extract_features(url)
        
        # 2. Buat DataFrame
        X = pd.DataFrame([features])

        # --- LOGIKA PENYELAMAT (ANTI ERROR 'Unnamed: 0') ---
        # Cek apakah FEATURE_ORDER mengandung kolom sampah
        # Jika model minta 'Unnamed: 0' tapi di X tidak ada, kita buatkan dengan nilai 0
        for col in FEATURE_ORDER:
            if col not in X.columns:
                X[col] = 0  # Isi dengan 0 agar model tidak error
        
        # Pastikan urutan kolom sesuai keinginan model
        X = X[FEATURE_ORDER] 
        # --------------------------------------------------

        # 3. Prediksi
        proba_all = model.predict_proba(X)[0]
        phishing_prob = float(proba_all[1])
        confidence = round(phishing_prob * 100, 2)

        # 4. Logika Label
        if phishing_prob >= 0.7:
            label = "Phishing"
            status_color = "danger"
        elif phishing_prob >= 0.4:
            label = "Suspicious"
            status_color = "warning"
        else:
            label = "Aman"
            status_color = "success"

        # 5. SHAP
        shap_values_raw = explainer.shap_values(X)
        if isinstance(shap_values_raw, list):
            s_val = shap_values_raw[1][0]
        else:
            s_val = shap_values_raw[0]

        shap_dict = {}
        for feat, val in zip(FEATURE_ORDER, s_val):
            # Jangan tampilkan 'Unnamed: 0' di tabel hasil web agar tidak bingung
            if feat != "Unnamed: 0":
                shap_dict[feat] = round(float(val), 4)

        sorted_shap = dict(sorted(shap_dict.items(), key=lambda item: abs(item[1]), reverse=True))

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
        return f"Terjadi kesalahan saat memproses URL: {str(e)}"

# ==========================================
# RUN APP
# ==========================================
if __name__ == "__main__":
    app.run(debug=True, port=5000)