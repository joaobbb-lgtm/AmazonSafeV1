import os, joblib, numpy as np

MODEL_PATH = os.getenv("MODEL_PATH", "models/risk_rf.joblib")
MODEL_VERSION = os.getenv("MODEL_VERSION", "rf_v1")

_DEF_COLS = ["focos","frp_med","pm25","pm10","pm_ratio","temp","rain_mm_24h","uvi","secos"]

def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)

def predict(feat: dict):
    """
    Retorna (label, {label:proba,...}) ou None se não houver modelo.
    """
    bundle = load_model()
    if not bundle:
        return None
    clf = bundle["model"]
    cols = bundle.get("cols", _DEF_COLS)
    labels = bundle.get("labels", ["verde","amarelo","vermelho"])

    x = np.array([[feat.get(c, 0.0) for c in cols]], dtype=float)
    proba = clf.predict_proba(x)[0].tolist()
    idx = int(np.argmax(proba))
    return labels[idx], dict(zip(labels, [round(p, 3) for p in proba]))
