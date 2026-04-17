import pandas as pd
import joblib
import shap
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
import os

# ===============================
# 1. LOAD DATA
# ===============================
print("[1/5] Loading dan membersihkan data...")
# Di train_model.py, saat load data:
df = pd.read_csv("Phising_Detection_Dataset.csv")

# Hapus kolom yang tidak diinginkan sebelum training
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

# Atau gunakan ini untuk hapus semua yang berawalan 'Unnamed'
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
# Bersihkan nama kolom jika ada tanda koma di depan (seperti di screenshot kamu)
df.columns = [c.replace(',', '').strip() for c in df.columns]

# Hilangkan baris dengan label Phising yang kosong (Penyebab error awal)
df = df.dropna(subset=['Phising'])

# Pastikan label Phising adalah integer (0 atau 1)
df['Phising'] = df['Phising'].astype(int)

# Pisahkan Fitur (X) dan Target (y)
X = df.drop(columns=['Phising'])
y = df['Phising']

# Tangani nilai kosong di fitur dengan median
if X.isnull().values.any():
    X = X.fillna(X.median())

print(f"Dataset dimuat: {X.shape[0]} baris, {X.shape[1]} fitur.")

# ===============================
# 2. SPLIT DATA
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ===============================
# 3. TRAIN MODEL (XGBOOST)
# ===============================
print("[2/5] Melatih model...")
neg = y_train.value_counts()[0]
pos = y_train.value_counts()[1]
scale_pos_weight = neg / pos

model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric='logloss'
)

model.fit(X_train, y_train)

# ===============================
# 4. EVALUASI
# ===============================
print("\n[3/5] Evaluasi Model:")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# ===============================
# 5. SIMPAN ASSET & VISUAL SHAP
# ===============================
print("[4/5] Menyimpan model...")
joblib.dump(model, "xgboost_phishing_model.pkl")
joblib.dump(X.columns.tolist(), "feature_columns.pkl")

print("[5/5] Membuat grafik SHAP...")
explainer = shap.TreeExplainer(model)
# Ambil 500 sampel data test untuk penjelasan visual
shap_values = explainer.shap_values(X_test[:500])

plt.figure(figsize=(12, 8))
# Membuat grafik summary plot (XAI)
shap.summary_plot(shap_values, X_test[:500], show=False)
plt.title("Fitur Paling Berpengaruh dalam Deteksi Phishing")
plt.tight_layout()
plt.savefig("shap_summary_plot.png")

print("\n✅ SELESAI!")
print("- Model disimpan: xgboost_phishing_model.pkl")
print("- Grafik SHAP disimpan: shap_summary_plot.png")