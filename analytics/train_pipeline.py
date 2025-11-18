# analytics/train_pipeline.py

import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Carrega o dataset (ajuste o caminho se necessário)
df = pd.read_csv("data/dataset_amazonia.csv")

X = df[["chuva_mm", "pm25", "pm10", "vento_m_s", "frp", "focos"]]
y = df["risco_label"]

pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler()),
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
pipeline.fit(X_train, y_train)

joblib.dump(pipeline, "models/amazonsafe_pipeline.joblib")

print("✅ Modelo salvo em models/amazonsafe_pipeline.joblib")
