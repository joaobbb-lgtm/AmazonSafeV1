# ai_model.py
from __future__ import annotations
import os, math, json, statistics as st
from typing import Dict, Any, Optional, List, Tuple

import joblib

try:
    # estes imports só são exigidos no TREINO; inferência não precisa deles carregados
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    import pandas as pd
except Exception:
    # ambiente de produção pode não ter sklearn/pandas instalados;
    # a inferência com modelo já treinado (joblib) funciona sem eles.
    pass

# -----------------------------
# Config
# -----------------------------
MODEL_DIR = os.environ.get("AMZSAFE_MODEL_DIR", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "modelo_alerta.pkl")
LABELS = ["verde", "amarelo", "vermelho"]

# -----------------------------
# Helpers
# -----------------------------
def _safe_float(x, default=None):
    try:
        if x is None: return default
        v = float(x)
        if math.isnan(v) or math.isinf(v): return default
        return v
    except Exception:
        return default

def _frp_mean(focos_list: List[Dict[str, Any]]) -> Optional[float]:
    vals = []
    for f in focos_list or []:
        v = _safe_float(f.get("frp"))
        if v is not None:
            vals.append(v)
    return (sum(vals)/len(vals)) if vals else None

def extract_features_from_payload(payload: Dict[str, Any]) -> Tuple[list[float], dict]:
    """
    Extrai features NUMÉRICAS do dicionário completo retornado por /api/alertas_update.
    Retorna (vetor_features, dict_debug)
    """
    # chuva 24h (o backend já calcula precip_24h_mm no breakdown quando open-meteo está ativo)
    precip_24h = _safe_float(
        payload.get("score", {}).get("breakdown", {}).get("precip_24h_mm"),
        # fallback: valor "current" da normalização, se existir
        _safe_float(payload.get("weather", {}).get("features", {}).get("precipitation"))
    )

    # qualidade do ar
    pm25 = _safe_float(payload.get("air", {}).get("pm25")) \
        or _safe_float(payload.get("air", {}).get("features", {}).get("pm25"))
    pm10 = _safe_float(payload.get("air", {}).get("pm10")) \
        or _safe_float(payload.get("air", {}).get("features", {}).get("pm10"))

    # vento
    wind = _safe_float(
        payload.get("weather", {}).get("features", {}).get("wind_speed_10m")
    )

    # focos e FRP médio
    focos_count = _safe_float(payload.get("focos", {}).get("count"))
    focos_list = payload.get("focos", {}).get("focos") or []
    frp_med = _frp_mean(focos_list)

    # faltas viram 0 (modelo simples e robusto)
    feats = [
        precip_24h or 0.0,
        pm25 or 0.0,
        pm10 or 0.0,
        wind or 0.0,
        frp_med or 0.0,
        focos_count or 0.0,
    ]
    info = {
        "rain_mm_24h": precip_24h,
        "pm25": pm25,
        "pm10": pm10,
        "wind": wind,
        "frp_med": frp_med,
        "focos": focos_count,
    }
    return feats, info

# -----------------------------
# Inferência
# -----------------------------
_loaded_model = None

def load_model(path: str = MODEL_PATH):
    global _loaded_model
    if _loaded_model is None and os.path.exists(path):
        _loaded_model = joblib.load(path)
    return _loaded_model

def has_model() -> bool:
    return load_model() is not None

def predict(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Retorna dict:
      {
        "modelo": "rf_v1",
        "label": "amarelo",
        "proba": {"verde": 0.12, "amarelo": 0.73, "vermelho": 0.15},
        "features": {...}
      }
    Caso não haja modelo carregado, retorna {"modelo":"rules_v1","label":<heurística simples>...}
    """
    feats, finfo = extract_features_from_payload(payload)

    mdl = load_model()
    if mdl is None:
        # fallback muito simples (mantém sistema operante)
        # regra: se focos>100 ou pm25>35 → "vermelho"; else se focos>10 ou pm25>12 → "amarelo"
        focos = finfo.get("focos") or 0
        pm25 = finfo.get("pm25") or 0
        if focos > 100 or pm25 >= 35:
            label = "vermelho"
            proba = {"verde": 0.0, "amarelo": 0.2, "vermelho": 0.8}
        elif focos > 10 or pm25 >= 12:
            label = "amarelo"
            proba = {"verde": 0.2, "amarelo": 0.7, "vermelho": 0.1}
        else:
            label = "verde"
            proba = {"verde": 0.75, "amarelo": 0.2, "vermelho": 0.05}
        return {"modelo": "rules_v1", "label": label, "proba": proba, "features": finfo}

    # inferência com modelo treinado
    proba_arr = mdl.predict_proba([feats])[0]
    idx = int(max(range(len(proba_arr)), key=lambda i: proba_arr[i]))
    label = LABELS[idx]
    proba = {LABELS[i]: float(proba_arr[i]) for i in range(len(LABELS))}
    return {"modelo": "rf_v1", "label": label, "proba": proba, "features": finfo}

# -----------------------------
# Logging p/ re-treino (opcional)
# -----------------------------
LOG_PATH = os.path.join(MODEL_DIR, "alerts_log.csv")

def log_example(payload: Dict[str, Any], label: Optional[str] = None):
    """
    Anexa uma linha CSV com features + rótulo (se existir).
    Útil para construir dataset histórico.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    feats, finfo = extract_features_from_payload(payload)
    lbl = label or (payload.get("score", {}).get("level")) or "verde"
    line = {
        "rain_mm_24h": finfo.get("rain_mm_24h") or 0,
        "pm25": finfo.get("pm25") or 0,
        "pm10": finfo.get("pm10") or 0,
        "wind": finfo.get("wind") or 0,
        "frp_med": finfo.get("frp_med") or 0,
        "focos": finfo.get("focos") or 0,
        "label": lbl
    }
    header = ",".join(line.keys())
    row = ",".join(str(line[k]) for k in line.keys())
    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, "w", encoding="utf-8") as f:
            f.write(header + "\n")
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(row + "\n")

# -----------------------------
# Treino (executar offline/cron)
# -----------------------------
def train_from_csv(csv_path: str, save_to: str = MODEL_PATH) -> Dict[str, Any]:
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    os.makedirs(os.path.dirname(save_to), exist_ok=True)
    df = pd.read_csv(csv_path)

    X = df[["rain_mm_24h", "pm25", "pm10", "wind", "frp_med", "focos"]].fillna(0.0)
    le = LabelEncoder()
    y = le.fit_transform(df["label"].fillna("verde"))

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_tr, y_tr)
    acc = float(clf.score(X_te, y_te))

    joblib.dump(clf, save_to)
    return {"saved_to": save_to, "accuracy": acc, "samples": int(len(df))}
