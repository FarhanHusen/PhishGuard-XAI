import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from utils.feature_extraction import extract_features 
from tqdm import tqdm
import os
import shap

# ==========================================
# 1. LOAD DATASET
# ==========================================
file_path = "raw_urls.csv"
if not os.path.exists(file_path):
    print(f"Error: File '{file_path}' tidak ditemukan!")
    exit()

df_raw = pd.read_csv(file_path)

# Identifikasi Kolom (Berdasarkan dataset: 'url' dan 'target')
url_col = 'url'
label_col = 'target'

# Bersihkan data dari nilai kosong
df_raw = df_raw.dropna(subset=[url_col, label_col])

print(f"Memproses {len(df_raw)} data URL...")

# ==========================================
# 2. EKSTRAKSI FITUR (SINKRON)
# ==========================================
feature_list = []
labels_list = []

for index, row in tqdm(df_raw.iterrows(), total=df_raw.shape[0], desc="Ekstraksi Fitur"):
    try:
        # Ekstrak fitur
        feat = extract_features(str(row[url_col]))
        feature_list.append(feat)
        # Simpan label hanya jika ekstraksi berhasil
        labels_list.append(row[label_col])
    except Exception as e:
        continue

X = pd.DataFrame(feature_list)
y = pd.Series(labels_list)

# Simpan urutan kolom untuk app.py
FEATURE_COLUMNS = X.columns.tolist()
joblib.dump(FEATURE_COLUMNS, "feature_columns.pkl")

# ==========================================
# 3. SPLIT DATA
# ==========================================
# Menggunakan stratify agar distribusi kelas tetap seimbang
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==========================================
# 4. TRAINING XGBOOST (DENGAN PENALTI KETAT)
# ==========================================
print("\nMelatih model XGBoost dengan parameter Anti-Overfitting...")

model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    
    # PARAMETER ANTI-OVERFITTING
    reg_lambda=3.0,
    reg_alpha=1.0,
    gamma=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    
    # PINDAHKAN EARLY STOPPING KE SINI
    early_stopping_rounds=10, 
    
    random_state=42,
    eval_metric='logloss'
)

# Di bagian .fit(), hapus early_stopping_rounds
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)

# ==========================================
# 5. EVALUASI
# ==========================================
y_pred = model.predict(X_test)
print("\n--- LAPORAN KLASIFIKASI ---")
print(classification_report(y_test, y_pred))
print(f"Akurasi Akhir: {round(accuracy_score(y_test, y_pred) * 100, 2)}%")

# ==========================================
# 6. SIMPAN ASSETS
# ==========================================
print("\nMenyimpan assets...")

# Simpan Model Utama
joblib.dump(model, "xgboost_phishing_model.pkl")

# Simpan SHAP Explainer
explainer = shap.TreeExplainer(model)
joblib.dump(explainer, "shap_explainer.pkl")

print("SUKSES! Model lebih bijak (Generalize) sekarang.")