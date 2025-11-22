
# ============================================================
# MÓDULO 1 — IMPORTS ESSENCIAIS E CONFIGURAÇÕES INICIAIS
# ============================================================

import os
import sys
import math
import json
import time
import datetime as dt
import io


# FastAPI
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware

# Tipos
from typing import Optional, Dict, Any

# DB (opcional – mantido porque o resto do código pode depender)
from sqlmodel import create_engine, SQLModel

# HTTP / Data
import requests
import pandas as pd

# IA
import joblib
import numpy as np

# ============================================================
# Banco de Dados — Mantido por compatibilidade
# ============================================================

DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
DB_URL = DATABASE_URL if DATABASE_URL else "sqlite:///./amazonsafe.db"
engine = create_engine(DB_URL, pool_pre_ping=True)

# ============================================================
# Constantes globais e chaves de APIs
# ============================================================

# Coordenadas padrão (Belém)
DEFAULT_LAT = float(os.getenv("DEFAULT_LAT", "-1.4558"))
DEFAULT_LON = float(os.getenv("DEFAULT_LON", "-48.5039"))

# INPE Queimadas (CSV)
INPE_CSV_BASE = os.getenv(
    "INPE_CSV_BASE",
    "https://dataserver-coids.inpe.br/queimadas/queimadas/focos/csv"
).rstrip("/")

INPE_DEFAULT_SCOPE = os.getenv("INPE_DEFAULT_SCOPE", "diario").lower()
INPE_DEFAULT_REGION = os.getenv("INPE_DEFAULT_REGION", "Brasil")

# Timeout
HTTP_TIMEOUT = int(os.getenv("HTTP_TIMEOUT", "60"))

CACHE_TTL_SEC = int(os.getenv("CACHE_TTL_SEC", "600"))

# ============================================================
# MÓDULO 2 — HELPERS GERAIS (cache, datas, haversine)
# ============================================================

UTC = dt.timezone.utc

def now_utc():
    return dt.datetime.now(UTC)

def iso_utc(ts: dt.datetime) -> str:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    return ts.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

# Cache TTL simples
def ttl_cache(ttl_seconds: int = 300):
    def deco(fn):
        store = {}
        def wrapper(*args, **kwargs):
            key = (fn.__name__, args, tuple(sorted(kwargs.items())))
            now = time.time()
            if key in store:
                val, ts = store[key]
                if now - ts < ttl_seconds:
                    return val
            val = fn(*args, **kwargs)
            store[key] = (val, now)
            return val
        wrapper._cache = store
        return wrapper
    return deco

# Distância geográfica
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2-lat1)
    dlmb = math.radians(lon2-lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dlmb/2)**2
    return 2 * R * math.asin(math.sqrt(a))

# ============================================================
# MÓDULO 3 — GEOCODING + COLETORES HTTP + NORMALIZAÇÃO
# ============================================================

import re
import io
import math
import datetime as dt
import requests
import pandas as pd
from typing import Optional, Dict, Any

# ---------------------------------------
# CONSTANTES E CONFIGURAÇÕES
# ---------------------------------------

GEOCODE_UA = "AmazonSafe/3 (+https://amazonsafe-api.onrender.com)"
OPENAQ_API_KEY = os.getenv("OPENAQ_API_KEY", "f6713d7b945cc5d989cdc08bcb44b62c0f343f11e0f1080555d0b768283ce101")
HTTP_TIMEOUT = int(os.getenv("HTTP_TIMEOUT", "60"))

# ============================================================
# 3.1 — GEOCODING (Nominatim + Open-Meteo fallback)
# ============================================================

UF2STATE = {
    "AC":"Acre","AL":"Alagoas","AP":"Amapá","AM":"Amazonas","BA":"Bahia",
    "CE":"Ceará","DF":"Distrito Federal","ES":"Espírito Santo","GO":"Goiás",
    "MA":"Maranhão","MT":"Mato Grosso","MS":"Mato Grosso do Sul","MG":"Minas Gerais",
    "PA":"Pará","PB":"Paraíba","PR":"Paraná","PE":"Pernambuco","PI":"Piauí",
    "RJ":"Rio de Janeiro","RN":"Rio Grande do Norte","RS":"Rio Grande do Sul",
    "RO":"Rondônia","RR":"Roraima","SC":"Santa Catarina","SP":"São Paulo",
    "SE":"Sergipe","TO":"Tocantins"
}

def _split_city_state(q: str):
    s = q.strip()
    m = re.split(r"\s*[,;-]\s*|\s{2,}", s, maxsplit=1)
    return (m[0].strip(), m[1].strip()) if len(m)==2 else (s, None)

def _normalize_state(st):
    if not st:
        return (None, None)
    up = st.upper()
    if up in UF2STATE:
        return (up, UF2STATE[up])
    for uf, name in UF2STATE.items():
        if up.lower() == name.lower():
            return (uf, name)
    return (None, st)

def _geocode_nominatim(q, state_name):
    try:
        r = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": q, "format":"jsonv2", "addressdetails":1, "limit":5, "countrycodes":"br"},
            headers={"User-Agent": GEOCODE_UA},
            timeout=HTTP_TIMEOUT
        )
        r.raise_for_status()
        arr = r.json() or []
        if not arr:
            return None

        best, score = None, -1
        for it in arr:
            lat, lon = it.get("lat"), it.get("lon")
            if not lat or not lon:
                continue
            s = 1
            if state_name and state_name.lower() in (it.get("address") or {}).get("state","").lower():
                s += 2
            if s > score:
                best, score = it, s

        if best:
            return {
                "lat": float(best["lat"]),
                "lon": float(best["lon"]),
                "display_name": best.get("display_name"),
                "source": "nominatim"
            }
    except:
        pass
    return None

def geocode_city(raw_q: str):
    if not raw_q.strip():
        return None

    city, st = _split_city_state(raw_q)
    uf, state_name = _normalize_state(st)

    # 1) Tenta Nominatim
    q = f"{city}, {state_name}, Brasil" if state_name else city
    res = _geocode_nominatim(q, state_name)
    if res:
        return res

    # 2) Fallback Open-Meteo geocoder
    try:
        r = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 10, "language": "pt", "country": "BR"},
            timeout=HTTP_TIMEOUT
        )
        arr = (r.json() or {}).get("results") or []
        if arr:
            best = arr[0]
            return {
                "lat": best["latitude"],
                "lon": best["longitude"],
                "display_name": f"{best['name']}, {best.get('admin1','')}",
                "source": "open-meteo"
            }
    except:
        pass

    return None

# ============================================================
# 3.2 — HTTP SESSION COM RETRY
# ============================================================

from requests.adapters import HTTPAdapter
try:
    from urllib3.util.retry import Retry
except:
    Retry = None

def make_retrying_session(
    total: int = 3,
    backoff: float = 0.5,
    status: tuple = (408, 429, 500, 502, 503, 504),
):
    s = requests.Session()
    if Retry is not None:
        r = Retry(
            total=total,
            read=total,
            connect=total,
            backoff_factor=backoff,
            status_forcelist=status,
            allowed_methods=frozenset({"GET", "HEAD", "OPTIONS"}),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=r, pool_connections=20, pool_maxsize=50)
        s.mount("http://", adapter)
        s.mount("https://", adapter)
    return s

_HTTP = make_retrying_session()

def http_get(url, *, params=None, headers=None, timeout=HTTP_TIMEOUT):
    to = timeout if isinstance(timeout, (int, float, tuple)) else HTTP_TIMEOUT
    if isinstance(to, (int, float)):
        to = (min(5, to), to)
    resp = _HTTP.get(url, params=params, headers=headers, timeout=to)
    resp.raise_for_status()
    return resp

# ============================================================
# 3.3 — Open-Meteo Forecast + Air Quality
# ============================================================

def safe_mean(values):
    if not isinstance(values, list):
        return None
    clean = [v for v in values if isinstance(v, (int, float))]
    return sum(clean) / len(clean) if clean else None


@ttl_cache(ttl_seconds=CACHE_TTL_SEC)
def get_meteo(lat: float, lon: float, retries: int = 5):
    """
    Coleta:
    - temperatura, umidade, ponto orvalho
    - pressão
    - vento, direção, rajadas
    - precipitação
    - radiação solar global e direta
    - evapotranspiração
    - solo temperatura e solo umidade
    - PM10, PM2.5, O3, NO2, SO2, CO
    - UV index
    """

    hourly_vars = ",".join([
        "temperature_2m",
        "relativehumidity_2m",
        "dewpoint_2m",
        "surface_pressure",
        "windspeed_10m",
        "winddirection_10m",
        "windgusts_10m",
        "precipitation",
        "shortwave_radiation",
        "direct_normal_irradiance",
        "soil_temperature_0cm",
        "soil_moisture_0_to_1cm",
        "evapotranspiration"
    ])

    url_weather = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&hourly={hourly_vars}"
        "&timezone=America%2FSao_Paulo"
    )

    url_air = (
        "https://air-quality-api.open-meteo.com/v1/air-quality"
        f"?latitude={lat}&longitude={lon}"
        "&hourly=pm10,pm2_5,ozone,nitrogen_dioxide,"
        "sulphur_dioxide,carbon_monoxide,uv_index"
        "&timezone=America%2FSao_Paulo"
    )

    for attempt in range(1, retries + 1):
        try:
            # Clima
            r1 = requests.get(url_weather, timeout=20)
            r1.raise_for_status()
            w = r1.json().get("hourly", {})

            # Qualidade do ar
            r2 = requests.get(url_air, timeout=20)
            r2.raise_for_status()
            a = r2.json().get("hourly", {})

            clima = {}

            # Forecast
            for var in hourly_vars.split(","):
                clima[var] = safe_mean(w.get(var))

            # Air quality
            clima["pm10"] = safe_mean(a.get("pm10"))
            clima["pm25"] = safe_mean(a.get("pm2_5"))
            clima["o3"]   = safe_mean(a.get("ozone"))
            clima["no2"]  = safe_mean(a.get("nitrogen_dioxide"))
            clima["so2"]  = safe_mean(a.get("sulphur_dioxide"))
            clima["co"]   = safe_mean(a.get("carbon_monoxide"))
            clima["uv"]   = safe_mean(a.get("uv_index"))

            return clima

        except Exception as e:
            print(f"[METEO] Falha {attempt}/{retries}: {e}")
            time.sleep(2)

    print("[METEO] Falhou após todas tentativas.")
    return {}

# ============================================================
# 3.5 — INPE (Queimadas)
# ============================================================

from io import BytesIO

def parse_float_safe(x):
    try:
        return float(str(x).replace(",", "."))
    except:
        return None

def _deg_per_km_lat():
    return 1.0 / 111.32

def _deg_per_km_lon(lat: float):
    return 1.0 / (111.32 * max(0.01, math.cos(math.radians(lat))))

def bbox_from_center(lat, lon, raio_km):
    dlat = raio_km * _deg_per_km_lat()
    dlon = raio_km * _deg_per_km_lon(lat)
    return (lon - dlon, lat - dlat, lon + dlon, lat + dlat)

def inpe_fetch_csv(scope="diario", region="Brasil", timeout=HTTP_TIMEOUT):
    today = dt.datetime.utcnow().date()
    ref = today

    base = INPE_CSV_BASE.rstrip("/")

    if scope == "diario":
        url = f"{base}/diario/{region}/focos_diario_br_{ref.strftime('%Y%m%d')}.csv"
    else:
        url = f"{base}/mensal/{region}/focos_mensal_br_{ref.strftime('%Y%m')}.csv"

    r = http_get(url, timeout=timeout)
    try:
        df = pd.read_csv(BytesIO(r.content), encoding="utf-8")
    except:
        df = pd.read_csv(BytesIO(r.content), encoding="latin1")

    return {"df": df, "url": url, "ref": str(ref)}

def _canonical_inpe_columns(df):
    cols = {c.lower(): c for c in df.columns}

    def pick(*names):
        for n in names:
            if n in cols:
                return cols[n]
        return None

    c_lat = pick("latitude","lat","y")
    c_lon = pick("longitude","lon","x")
    c_dt  = pick("datahora","data_hora")
    c_sat = pick("satelite","satellite")
    c_frp = pick("frp","radiative_power")

    out = pd.DataFrame()
    out["latitude"] = df[c_lat] if c_lat else None
    out["longitude"] = df[c_lon] if c_lon else None
    out["datahora"] = df[c_dt] if c_dt else None
    out["satelite"] = df[c_sat] if c_sat else None
    out["frp"] = df[c_frp] if c_frp else None
    return out

def _json_safe(v):
    try:
        if pd.isna(v):
            return None
    except:
        pass
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v

@ttl_cache(ttl_seconds=CACHE_TTL_SEC)
def inpe_focos_near(lat, lon, raio_km=150, scope="diario", region="Brasil", limit=1000, timeout=HTTP_TIMEOUT):
    payload = inpe_fetch_csv(scope=scope, region=region, timeout=timeout)
    df = payload["df"]
    norm = _canonical_inpe_columns(df)

    norm["latitude"] = norm["latitude"].map(parse_float_safe)
    norm["longitude"] = norm["longitude"].map(parse_float_safe)

    minx, miny, maxx, maxy = bbox_from_center(lat, lon, float(raio_km))
    mask = (
        (norm["longitude"].notna()) &
        (norm["latitude"].notna()) &
        (norm["longitude"] >= minx) &
        (norm["longitude"] <= maxx) &
        (norm["latitude"] >= miny) &
        (norm["latitude"] <= maxy)
    )
    sub = norm[mask].copy()

    try:
        sub["dist_km"] = sub.apply(lambda r: haversine_km(lat, lon, r["latitude"], r["longitude"]), axis=1)
        sub = sub[sub["dist_km"] <= float(raio_km)]
    except:
        pass

    if limit:
        sub = sub.head(int(limit))

    focos = []
    for _, r in sub.iterrows():
        focos.append({
            "latitude": _json_safe(r.get("latitude")),
            "longitude": _json_safe(r.get("longitude")),
            "datahora": _json_safe(r.get("datahora")),
            "satelite": _json_safe(r.get("satelite")),
            "frp": _json_safe(r.get("frp")),
            "dist_km": _json_safe(r.get("dist_km")),
        })

    meta = {
        "source": "inpe_csv",
        "url": payload["url"],
        "reference": payload["ref"],
        "bbox": {"minlon": minx, "minlat": miny, "maxlon": maxx, "maxlat": maxy},
        "count": len(focos),
        "timestamp_utc": dt.datetime.utcnow().isoformat()+"Z",
    }

    return {
        "features": {"focos": focos, "count": len(focos), "meta": meta},
        "payload": {"csv_url": payload["url"]},
    }

# ============================================================
# 3.6 — NORMALIZADORES
# ============================================================

def normalize_meteo(data: dict) -> dict:
    out = {}
    for k, v in data.items():
        try:
            out[k] = float(v) if v is not None else None
        except:
            out[k] = None
    return out

# ============================================================
# MÓDULO 4 — Funções auxiliares INPE (bbox / parsing)
# ============================================================

def _deg_per_km_lat():
    return 1.0 / 111.32

def _deg_per_km_lon(lat: float):
    return 1.0 / (111.32 * max(0.01, math.cos(math.radians(lat))))

def bbox_from_center(lat, lon, raio_km):
    dlat = raio_km * _deg_per_km_lat()
    dlon = raio_km * _deg_per_km_lon(lat)
    return (lon - dlon, lat - dlat, lon + dlon, lat + dlat)

def parse_float_safe(x):
    try:
        return float(str(x).replace(",", "."))
    except:
        return None

# ============================================================
# MÓDULO 5 — Inicialização da API FastAPI
# ============================================================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Configuração do AmazonSafe API carregada com sucesso.")

# ============================================================
# MÓDULO 6 — Persistência (SQLModel + helpers + utilitários)
# ============================================================

from typing import Optional, Dict, Any
from sqlmodel import SQLModel, Field, Session, select
import json
import datetime as dt
import os
import time
import math

UTC = dt.timezone.utc
def _now_utc():
    return dt.datetime.now(UTC)

# ------------------------------------------------------------
# MODELO UNIFICADO — WeatherObs (clima + solo + ar)
# ------------------------------------------------------------

class WeatherObs(SQLModel, table=True):
    __tablename__ = "weatherobs"
    __table_args__ = {"extend_existing": True}

    id: Optional[int] = Field(default=None, primary_key=True)

    lat: float
    lon: float
    fonte: str = "open-meteo"   # origem dos dados

    # -----------------------------
    # Clima (Open-Meteo Forecast)
    # -----------------------------
    temperatura: Optional[float] = None
    umidade: Optional[float] = None
    ponto_orvalho: Optional[float] = None
    pressao: Optional[float] = None

    vento_m_s: Optional[float] = None
    vento_dir: Optional[float] = None
    rajadas: Optional[float] = None

    chuva_mm: Optional[float] = None
    rad_solar: Optional[float] = None
    rad_direta: Optional[float] = None

    solo_temp_0cm: Optional[float] = None
    solo_umid_0_1cm: Optional[float] = None
    evapotranspiracao: Optional[float] = None

    # -----------------------------
    # Qualidade do ar (Open-Meteo AQ)
    # -----------------------------
    pm10: Optional[float] = None
    pm25: Optional[float] = None
    o3: Optional[float] = None
    no2: Optional[float] = None
    so2: Optional[float] = None
    co: Optional[float] = None
    uv: Optional[float] = None

    observed_at: dt.datetime = Field(default_factory=_now_utc)
    raw_json: Optional[str] = None


class FireObs(SQLModel, table=True):
    __tablename__ = "fireobs"
    __table_args__ = {"extend_existing": True}

    id: Optional[int] = Field(default=None, primary_key=True)
    lat: float
    lon: float
    fonte: str = "inpe_csv"
    payload: Optional[str] = None
    observed_at: dt.datetime = Field(default_factory=_now_utc)


class AlertObs(SQLModel, table=True):
    __tablename__ = "alertobs"
    __table_args__ = {"extend_existing": True}

    id: Optional[int] = Field(default=None, primary_key=True)
    lat: float
    lon: float
    tipo: str
    payload: Optional[str] = None
    observed_at: dt.datetime = Field(default_factory=_now_utc)


print("DB pronto: tabelas criadas/atualizadas em", DB_URL)

# ------------------------------------------------------------
# HELPER PARA SALVAR WEATHER COMPLETO
# ------------------------------------------------------------

def save_weather(lat: float, lon: float, fonte: str,
                 clima: Dict[str, Any],
                 raw: Dict[str, Any] | None = None) -> int:
    """
    Salva no banco um registro completo de clima + solo + ar.
    'clima' deve ser exatamente o dict vindo de normalize_meteo(get_meteo()).
    """

    # Monta o registro expandindo automaticamente as keys
    rec = WeatherObs(
        lat=lat,
        lon=lon,
        fonte=fonte,
        **clima,
        observed_at=_now_utc(),
        raw_json=json.dumps(raw or clima, ensure_ascii=False),
    )

    with Session(engine) as sess:
        sess.add(rec)
        sess.commit()
        sess.refresh(rec)
        return rec.id


def save_fire(lat: float, lon: float, fonte: str = "inpe_csv",
              payload: Dict[str, Any] | None = None) -> int:

    rec = FireObs(
        lat=lat,
        lon=lon,
        fonte=fonte,
        payload=json.dumps(payload or {}, ensure_ascii=False),
        observed_at=_now_utc(),
    )

    with Session(engine) as sess:
        sess.add(rec)
        sess.commit()
        sess.refresh(rec)
        return rec.id


# ------------------------------------------------------------
# CONSULTAS (GETS SIMPLES)
# ------------------------------------------------------------

def get_last_weather():
    with Session(engine) as sess:
        return sess.exec(
            select(WeatherObs).order_by(WeatherObs.id.desc()).limit(1)
        ).first()

def get_last_fire():
    with Session(engine) as sess:
        return sess.exec(
            select(FireObs).order_by(FireObs.id.desc()).limit(1)
        ).first()


# ------------------------------------------------------------
# ALERTAS — eventos, score, armazenamento NDJSON
# (Mantidos sem alteração)
# ------------------------------------------------------------

try:
    _LAST_LEVEL
except NameError:
    _LAST_LEVEL = {}

try:
    _LAST_NOTIFY
except NameError:
    _LAST_NOTIFY = {}

NOTIFY_DEBOUNCE_SEC = int(os.getenv("NOTIFY_DEBOUNCE_SEC", "600"))
WEBHOOK_URL = os.getenv("ALERTS_WEBHOOK_URL")


def _ensure_alerts_dir():
    d = "./runtime_data/alerts"
    os.makedirs(d, exist_ok=True)
    return d


def save_alert_score(alert_id: str, score: float, level: str,
                     alert_obs: Dict[str, Any],
                     params: Dict[str, Any] | None = None):

    try:
        d = _ensure_alerts_dir()
        rec = {
            "ts": _now_utc().isoformat().replace("+00:00", "Z"),
            "alert_id": alert_id,
            "score": score,
            "level": level,
            "alert_obs": alert_obs,
            "params": params or {},
        }
        with open(os.path.join(d, "alerts.ndjson"), "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception as e:
        print("[save_alert_score] WARN:", e)


def _persist_level_event(alert_id: str, old_level: str, new_level: str, payload: dict):

    try:
        d = _ensure_alerts_dir()
        rec = {
            "ts": _now_utc().isoformat().replace("+00:00", "Z"),
            "alert_id": alert_id,
            "from": old_level,
            "to": new_level,
            "payload": payload,
        }
        with open(os.path.join(d, "level_events.ndjson"), "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception as e:
        print("[level_event] WARN:", e)


def _notify_level_change(alert_id: str, old_level: str, new_level: str,
                         score: float, obs: dict):

    now = time.time()
    last = _LAST_NOTIFY.get(alert_id, 0)
    if now - last < NOTIFY_DEBOUNCE_SEC:
        return

    _LAST_NOTIFY[alert_id] = now

    msg = {
        "alert_id": alert_id,
        "from": old_level,
        "to": new_level,
        "score": score,
        "when": _now_utc().isoformat().replace("+00:00", "Z"),
        "obs": {
            k: obs.get(k)
            for k in ("severity", "duration", "frequency", "impact")
        },
    }

    try:
        if WEBHOOK_URL:
            requests.post(WEBHOOK_URL, json=msg, timeout=8)
        else:
            d = _ensure_alerts_dir()
            with open(os.path.join(d, "notifications.ndjson"), "a", encoding="utf-8") as f:
                f.write(json.dumps(msg, ensure_ascii=False) + "\n")
    except Exception as e:
        print("[notify] WARN:", e)


def handle_level_transition(alert_id: str, new_level: str, score: float,
                            alert_obs: Dict[str, Any],
                            extra: Dict[str, Any] | None = None,
                            notify_on_bootstrap: bool = False):

    old_level = (_LAST_LEVEL.get(alert_id) or {}).get("level")
    first_time = (old_level is None)

    if new_level != old_level:
        _persist_level_event(alert_id, old_level, new_level, {
            "score": score,
            "obs": alert_obs,
            **(extra or {}),
        })

        if (not first_time) or notify_on_bootstrap:
            _notify_level_change(alert_id, old_level, new_level, score, alert_obs)

        _LAST_LEVEL[alert_id] = {"level": new_level, "ts": _now_utc()}


# ------------------------------------------------------------
# HELPERS GERAIS (Mantidos)
# ------------------------------------------------------------

def safe_number(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

def level_to_color(level: str) -> str:
    lv = (level or "").lower()
    if lv.startswith("verm"):
        return "#ef4444"
    if lv.startswith("amar"):
        return "#eab308"
    return "#22c55e"

def _now_unix():
    return int(time.time())

def _coalesce(*vals, default=None):
    for v in vals:
        if v is not None:
            return v
    return default

def _is_valid_pm25(x):
    try:
        xf = float(x)
        return math.isfinite(xf) and xf >= 1.0
    except:
        return False


# ============================================================
# MÓDULO 7 — Sistema de Risco (Atualizado) + Helpers de Localização
# ============================================================

import time
from fastapi import HTTPException

# ------------------------------------------------------------
# PARÂMETROS DO SISTEMA DE RISCO ATUALIZADO
# ------------------------------------------------------------

PM25_LIMIT = 35        # µg/m³
PM10_LIMIT = 50        # µg/m³
CHUVA_BAIXA_MM = 5.0   # mm

WEIGHT_PM25 = 40
WEIGHT_PM10 = 30
WEIGHT_SECO = 30

THRESHOLD_YELLOW = 40
THRESHOLD_RED    = 70


def score_risk(meteo: dict):
    """
    Sistema de risco simplificado baseado no dataset unificado:
    Entrada:
        meteo = dict retornado por normalize_meteo(get_meteo())

    Variáveis usadas:
        pm25, pm10, chuva_mm

    Retorna:
        score (0 a 100)
        level ("Verde", "Amarelo", "Vermelho")
    """

    pm25 = meteo.get("pm25") or 0
    pm10 = meteo.get("pm10") or 0
    chuva = meteo.get("chuva_mm") or 0

    score = 0

    if pm25 >= PM25_LIMIT:
        score += WEIGHT_PM25

    if pm10 >= PM10_LIMIT:
        score += WEIGHT_PM10

    if chuva <= CHUVA_BAIXA_MM:
        score += WEIGHT_SECO

    score = max(0, min(100, int(score)))

    if score >= THRESHOLD_RED:
        level = "Vermelho"
    elif score >= THRESHOLD_YELLOW:
        level = "Amarelo"
    else:
        level = "Verde"

    return score, level


# ------------------------------------------------------------
# HELPERS DE LOCALIZAÇÃO (mantidos, apenas reorganizados)
# ------------------------------------------------------------

def _as_float_or_none(x):
    if x is None:
        return None
    if isinstance(x, str):
        x = x.strip()
        if not x:
            return None
    try:
        return float(x)
    except:
        return None


def _resolve_location(cidade: Optional[str], lat: Optional[float], lon: Optional[float]):
    """
    Regras:
    1) Se cidade foi passada → geocodifica
    2) Se há lat/lon → usa diretamente
    3) Senão → fallback para DEFAULT_LAT/LON
    """

    lat = _as_float_or_none(lat)
    lon = _as_float_or_none(lon)

    if cidade:
        info = geocode_city(cidade)
        if not info:
            raise HTTPException(
                status_code=404,
                detail=f"Não foi possível geocodificar '{cidade}'."
            )
        return float(info["lat"]), float(info["lon"]), {
            "resolved_by": "geocode",
            "display_name": info.get("display_name"),
        }

    if lat is not None and lon is not None:
        return lat, lon, {"resolved_by": "direct_params"}

    # fallback: Belém
    return float(DEFAULT_LAT), float(DEFAULT_LON), {"resolved_by": "default"}


# ============================================================
# 🧩 MÓDULO 8 — IA AmazonSafe v10 (RandomForest)
# ============================================================

from pydantic import BaseModel
import joblib
import numpy as np
from fastapi import HTTPException
import datetime as dt
import pandas as pd

# ============================================================
# 8.0 — CARREGAMENTO DO PIPELINE FINAL AmazonSafe v10
# ============================================================

MODEL_PATH = "models/amazonsafe_pipeline_v10.joblib"

try:
    modelo_pipeline = joblib.load(MODEL_PATH)
    print(f"[IA] Modelo AmazonSafe v10 carregado de {MODEL_PATH}")
except Exception as e:
    print(f"[IA] ERRO ao carregar modelo AmazonSafe v10: {e}")
    modelo_pipeline = None

# ============================================================
# 8.1 — FEATURES EXATAS UTILIZADAS NO TREINAMENTO
# ============================================================
# Essas são TODAS as colunas do dataset_scaled v10 após remover:
# cidade, uf, focos_50km, focos_150km, focos_300km

MODEL_FEATURES = [
    "latitude",
    "longitude",
    "temperatura",
    "umidade",
    "ponto_orvalho",
    "pressao",
    "vento_m_s",
    "vento_dir",
    "rajadas",
    "chuva_mm",
    "rad_solar",
    "rad_direta",
    "solo_temp_0cm",
    "solo_umid_0_1cm",
    "evapotranspiracao",
    "pm10",
    "pm25",
    "o3",
    "no2",
    "so2",
    "co",
    "uv"
]

# ============================================================
# 8.2 — MODELO DO PAYLOAD /api/risk
# ============================================================

class RiskInput(BaseModel):
    latitude: float
    longitude: float
    temperatura: float
    umidade: float
    ponto_orvalho: float
    pressao: float
    vento_m_s: float
    vento_dir: float
    rajadas: float
    chuva_mm: float
    rad_solar: float
    rad_direta: float
    solo_temp_0cm: float
    solo_umid_0_1cm: float
    evapotranspiracao: float
    pm10: float
    pm25: float
    o3: float
    no2: float
    so2: float
    co: float
    uv: float


# ============================================================
# 8.3 — ENDPOINT OFICIAL /api/risk (PREVISÃO)
# ============================================================

@app.post("/api/risk", tags=["IA"], summary="Previsão focos_300km e risco ambiental (v10)")
def api_risk(data: RiskInput):

    if modelo_pipeline is None:
        raise HTTPException(status_code=500, detail="Modelo v10 não carregado")

    entrada = data.model_dump()

    # monta o vetor na ordem exata do treinamento
    try:
        row = [entrada[col] for col in MODEL_FEATURES]
    except KeyError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Variável ausente no payload: {str(e)}"
        )

    X = np.array([row], dtype=float)

    # previsão
    try:
        focos_prev = float(modelo_pipeline.predict(X)[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao gerar previsão: {e}")

    # classificação simples
    if focos_prev <= 50:
        risco = "baixo"
    elif focos_prev <= 150:
        risco = "medio"
    else:
        risco = "alto"

    return {
        "modelo": {
            "path": MODEL_PATH,
            "features": MODEL_FEATURES,
        },
        "entrada": entrada,
        "focos_previstos_300km": focos_prev,
        "risco_classificado": risco,
    }

# ============================================================
# 8.4 — COLETA CLIMÁTICA DO ARCHIVE (para estudos e validação)
# ============================================================

def collect_weather_archive(lat: float, lon: float, ref_date: str, timeout: int = 20):
    """
    Busca clima horário no Open-Meteo ARCHIVE e calcula a média diária
    das features usadas no treinamento do modelo.
    """

    vars_str = ",".join([
        "temperature_2m",
        "relative_humidity_2m",
        "dewpoint_2m",
        "surface_pressure",
        "wind_speed_10m",
        "wind_direction_10m",
        "wind_gusts_10m",
        "precipitation",
        "shortwave_radiation",
        "direct_normal_irradiance",
        "soil_temperature_0cm",
        "soil_moisture_0_to_1cm",
        "evapotranspiration"
    ])

    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={ref_date}&end_date={ref_date}"
        f"&hourly={vars_str}"
        "&timezone=UTC"
    )

    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        h = data.get("hourly", {}) or {}

        clima = {}
        for var in h:
            serie = h[var]
            serie_ok = [
                x for x in serie
                if x is not None and isinstance(x, (int, float)) and not pd.isna(x)
            ]
            clima[var] = float(sum(serie_ok) / len(serie_ok)) if serie_ok else None

        return clima

    except Exception as e:
        print(f"[IA Archive] ERRO: {e}")
        return None

# ============================================================
# 8.5 — FOCOS POR RAIO (COMPARAÇÃO REAL x PREVISTO)
# ============================================================

def focos_por_raios_backend(lat: float, lon: float):
    """
    Conta focos reais em 50 / 150 / 300 km.
    Não entra no modelo — apenas comparação.
    """
    try:
        data = inpe_focos_near(lat, lon, raio_km=300)
        focos = data["features"]["focos"]
    except:
        focos = []

    f50 = f150 = f300 = 0

    for f in focos:
        d = f.get("dist_km")
        try:
            d = float(d)
        except:
            continue

        if d <= 50: f50 += 1
        if d <= 150: f150 += 1
        if d <= 300: f300 += 1

    return {
        "focos_50km": f50,
        "focos_150km": f150,
        "focos_300km": f300,
    }

# ============================================================
# 8.6 — ENDPOINT /api/model_score (USO COM REF_DATE)
# ============================================================

class AIScoreRequest(BaseModel):
    cidade: str | None = None
    lat: float | None = None
    lon: float | None = None
    ref_date: str | None = None

@app.post("/api/model_score", tags=["IA"], summary="Predição histórica (Archive)")
def api_model_score(body: AIScoreRequest):

    # resolve local
    if body.cidade:
        lat, lon, meta = _resolve_location(body.cidade, None, None)
    elif body.lat and body.lon:
        lat, lon = body.lat, body.lon
        meta = {"resolved_by": "direct_params"}
    else:
        lat, lon = DEFAULT_LAT, DEFAULT_LON
        meta = {"resolved_by": "default"}

    ref_date = body.ref_date or dt.date.today().strftime("%Y-%m-%d")

    clima = collect_weather_archive(lat, lon, ref_date)
    if clima is None:
        raise HTTPException(status_code=502, detail="Falha ao obter clima via Archive")

    # montar X usando as features atuais
    row = []
    for col in MODEL_FEATURES:
        v = clima.get(col)
        row.append(0.0 if v is None or not isinstance(v, (int, float)) else float(v))

    X = np.array([row], dtype=float)

    predicted = float(modelo_pipeline.predict(X)[0])

    if predicted < 50:
        risco = "baixo"
    elif predicted < 150:
        risco = "medio"
    else:
        risco = "alto"

    focos_real = focos_por_raios_backend(lat, lon)

    return {
        "ok": True,
        "local": meta,
        "ref_date": ref_date,
        "clima_archive": clima,
        "focos_reais": focos_real,
        "predicao": {
            "focos_previstos_300km": predicted,
            "risco": risco,
        }
    }


# ============================================================
# 🧩 MÓDULO 9 — IA LEVE + Scoring Inteligente (MAD + Score Avançado)
# ============================================================

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime, timezone
import math
import statistics as stats
from sqlmodel import Session, select

# -----------------------------------------
# Limiares do score inteligente
# -----------------------------------------
THRESHOLDS = {
    "green_lt": 0.33,
    "yellow_lt": 0.66,
}

WEIGHTS = {
    "severity": 0.25,
    "duration": 0.25,
    "frequency": 0.25,
    "impact": 0.15,
    "rainfall": 0.10,
}

@dataclass
class ScoreResult:
    score: float
    level: str
    breakdown: Dict[str, float]


# -----------------------------------------
# Helpers internos
# -----------------------------------------
def _clip01(x: float) -> float:
    """Garante que o valor fique dentro do intervalo 0..1."""
    try:
        return max(0.0, min(1.0, float(x)))
    except Exception:
        return 0.0


def _classify_level(score: float) -> str:
    """Converte score final em nível textual."""
    if score < THRESHOLDS["green_lt"]:
        return "verde"
    if score < THRESHOLDS["yellow_lt"]:
        return "amarelo"
    return "vermelho"


def _norm_pm(pm_val: Optional[float], pm_kind: str) -> float:
    """
    Normaliza PM em faixa 0..1.
    - pm2.5 -> referência 75
    - pm10  -> referência 150
    """
    if pm_val is None:
        return 0.0
    ref = 75.0 if pm_kind == "pm25" else 150.0
    return _clip01(pm_val / ref)


def _hours_since(iso_str: Optional[str]) -> Optional[float]:
    """Tempo em horas desde a última atualização do sensor."""
    if not iso_str:
        return None
    try:
        t = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        return max(0.0, (now - t).total_seconds() / 3600.0)
    except Exception:
        return None


# -----------------------------------------
# Risco Pluviométrico Inteligente
# -----------------------------------------
def _rainfall_risk_u(mm: Optional[float]) -> float:
    """
    Converte precipitação (mm/24h) para risco 0..1.
    0 mm  = 0.0 (nenhum risco)
    50 mm = 1.0 (risco extremo)
    """
    if mm is None:
        return 0.0
    try:
        mm = float(mm)
    except Exception:
        return 0.0

    if mm <= 0:
        return 0.0
    if mm >= 50:
        return 1.0
    return mm / 50.0


def _rainfall_index(mm: Optional[float]) -> float:
    """Alias semântica — mantém consistência de nome."""
    return _rainfall_risk_u(mm)


# -----------------------------------------
# SCORE INTELIGENTE PRINCIPAL
# -----------------------------------------
def compute_alert_score(alert_obs: Dict[str, Any],
                        weights: Dict[str, float] = WEIGHTS) -> ScoreResult:
    """
    Score avançado baseado nos 5 pilares:
    - severity (0..1)
    - duration (0..1)
    - frequency (0..1)
    - impact (0..1)
    - rainfall (chuva_mm/precipitação 24h -> 0..1)

    Retorna ScoreResult com score, nível e detalhamento.
    """

    sev = _clip01(alert_obs.get("severity", 0))
    dur = _clip01(alert_obs.get("duration", 0))
    freq = _clip01(alert_obs.get("frequency", 0))
    imp = _clip01(alert_obs.get("impact", 0))

    # Busca precipitação: suporta chaves novas (chuva_mm) e legadas
    precip_24h = (
        alert_obs.get("precip_24h")
        or alert_obs.get("chuva_mm")
        or alert_obs.get("precipitation")
        or (alert_obs.get("meta", {}).get("precipitation"))
        or 0.0
    )

    rainfall = _rainfall_index(precip_24h)

    score = (
        sev * weights.get("severity", 0.25) +
        dur * weights.get("duration", 0.25) +
        freq * weights.get("frequency", 0.25) +
        imp * weights.get("impact", 0.15) +
        rainfall * weights.get("rainfall", 0.10)
    )

    level = _classify_level(score)

    return ScoreResult(
        score=round(score, 4),
        level=level,
        breakdown={
            "severity": sev,
            "duration": dur,
            "frequency": freq,
            "impact": imp,
            "precip_index": rainfall,
            "precip_24h_mm": float(precip_24h),
        },
    )


# ============================================================
# IA Leve: Detectores de Outliers (MAD) usando WeatherObs
# ============================================================

def _round_coord(v: float, ndigits: int = 3) -> float:
    """Arredonda coordenada base para consulta local."""
    return round(float(v), ndigits)


def _mad(values: List[float]):
    """Retorna (mediana, MAD) ou None se lista vazia."""
    if not values:
        return None
    med = stats.median(values)
    devs = [abs(x - med) for x in values]
    mad = stats.median(devs)
    return med, mad


def pm_outlier_flags(
    lat: float,
    lon: float,
    pm25: Optional[float],
    pm10: Optional[float],
    k: float = 5.0,
    lookback: int = 50
) -> Tuple[bool, bool]:
    """
    Detecta se pm25/pm10 são outliers usando MAD:

    - Busca últimas N medições próximas na tabela WeatherObs
      (histórico local de qualidade do ar)
    - Compara com mediana ± k * MAD * 1.4826

    Retorna:
        (flag_pm25_outlier, flag_pm10_outlier)
    """

    if pm25 is None and pm10 is None:
        return False, False

    latk = _round_coord(lat)
    lonk = _round_coord(lon)

    from .main import WeatherObs  # se estiver em arquivo separado, ajuste o import
    # No seu caso, como está tudo no mesmo main.py, você pode remover essa linha
    # e garantir que WeatherObs já foi definido acima no arquivo.

    with Session(engine) as s:
        rows = s.exec(
            select(WeatherObs)
            .where(WeatherObs.lat.between(latk - 0.001, latk + 0.001))
            .where(WeatherObs.lon.between(lonk - 0.001, lonk + 0.001))
            .order_by(WeatherObs.observed_at.desc())
            .limit(lookback)
        ).all()

    pm25_hist = [r.pm25 for r in rows if r.pm25 is not None]
    pm10_hist = [r.pm10 for r in rows if r.pm10 is not None]

    p25_flag = False
    p10_flag = False

    if pm25 is not None and len(pm25_hist) >= 10:
        med, mad = _mad(pm25_hist)
        if mad and abs(pm25 - med) > k * 1.4826 * mad:
            p25_flag = True

    if pm10 is not None and len(pm10_hist) >= 10:
        med, mad = _mad(pm10_hist)
        if mad and abs(pm10 - med) > k * 1.4826 * mad:
            p10_flag = True

    return p25_flag, p10_flag


# ============================================================
# 🧩 MÓDULO 10 — COLETORES ATUAIS PARA O DASHBOARD (v10)
# ============================================================

def collect_weather_now(lat: float, lon: float) -> dict:
    """
    Coleta o clima atual usando o novo get_meteo() oficial.
    Retorna todas as variáveis do dataset v10 em um formato padronizado.
    """

    try:
        clima = get_meteo(lat, lon)
    except Exception as e:
        return {
            "ok": False,
            "erro": str(e),
            "fonte": "open-meteo",
            "features": {}
        }

    # Normaliza os dados em floats (já vem do normalize_meteo)
    clima_norm = normalize_meteo(clima)

    return {
        "ok": True,
        "fonte": "open-meteo",
        "latitude": lat,
        "longitude": lon,
        "features": clima_norm,
    }


def collect_focos_now(lat: float, lon: float) -> dict:
    """
    Retorna focos reais próximos em 50, 150 e 300 km.
    """

    try:
        focos = focos_por_raios_backend(lat, lon)
    except Exception as e:
        return {
            "ok": False,
            "erro": str(e),
        }

    return {
        "ok": True,
        "latitude": lat,
        "longitude": lon,
        "focos": focos
    }


def collect_dashboard_bundle(lat: float, lon: float) -> dict:
    """
    Pacote simplificado para o front-end (dashboard).
    Retorna clima atual + focos reais num único payload.
    """

    clima = collect_weather_now(lat, lon)
    focos = collect_focos_now(lat, lon)

    return {
        "ok": True,
        "coords": {"lat": lat, "lon": lon},
        "clima": clima,
        "focos": focos,
    }

# ============================================================
# 🧩 MÓDULO 11 — ENDPOINT /api/data (Dashboard v10)
# ============================================================

class DataRequest(BaseModel):
    cidade: str | None = None
    lat: float | None = None
    lon: float | None = None


@app.post("/api/data", tags=["Dashboard"], summary="Dados completos para o dashboard AmazonSafe v10")
def api_data(req: DataRequest):

    # ------------------------------------------------------------
    # 1) Resolver localização
    # ------------------------------------------------------------
    try:
        lat, lon, loc_meta = _resolve_location(
            cidade=req.cidade,
            lat=req.lat,
            lon=req.lon,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Erro ao resolver localização: {e}"
        )

    # ------------------------------------------------------------
    # 2) Clima atual (usando o novo get_meteo)
    # ------------------------------------------------------------
    try:
        clima_raw = get_meteo(lat, lon)
        clima = normalize_meteo(clima_raw)
    except Exception as e:
        clima = {}
        clima_raw = {"erro": str(e)}

    # ------------------------------------------------------------
    # 3) Focos reais via backend (50 / 150 / 300 km)
    # ------------------------------------------------------------
    try:
        focos = focos_por_raios_backend(lat, lon)
    except Exception as e:
        focos = {"erro": str(e)}

    # ------------------------------------------------------------
    # 4) Retorno consolidado no novo padrão v10
    # ------------------------------------------------------------
    return {
        "ok": True,
        "local": {
            "entrada_cidade": req.cidade,
            "lat": lat,
            "lon": lon,
            "resolved_by": loc_meta.get("resolved_by"),
            "display_name": loc_meta.get("display_name"),
        },
        "clima_atual": {
            "fonte": "open-meteo",
            "features": clima,
            "raw": clima_raw
        },
        "focos_reais": focos,
    }
# ============================================================
# 🧩 MÓDULO 12 — ENDPOINT /api/data_auto (Dashboard + IA v10)
# ============================================================

class DataAutoRequest(BaseModel):
    cidade: str | None = None
    lat: float | None = None
    lon: float | None = None


@app.post("/api/data_auto", tags=["Dashboard"], summary="Dados completos + IA (AmazonSafe v10)")
def api_data_auto(req: DataAutoRequest):

    # ------------------------------------------------------------
    # 1) Resolver localização
    # ------------------------------------------------------------
    try:
        lat, lon, loc_meta = _resolve_location(
            cidade=req.cidade,
            lat=req.lat,
            lon=req.lon
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Erro ao resolver localização: {e}"
        )

    # ------------------------------------------------------------
    # 2) Clima atual (Open-Meteo Forecast)
    # ------------------------------------------------------------
    try:
        clima_raw = get_meteo(lat, lon)
        clima = normalize_meteo(clima_raw)
    except Exception as e:
        clima_raw = {"erro": str(e)}
        clima = {}

    # ------------------------------------------------------------
    # 3) Focos reais (50 / 150 / 300 km)
    # ------------------------------------------------------------
    try:
        focos_reais = focos_por_raios_backend(lat, lon)
    except Exception as e:
        focos_reais = {"erro": str(e)}

    # ------------------------------------------------------------
    # 4) IA — Preparar vetor para o modelo
    # ------------------------------------------------------------
    if modelo_pipeline is None:
        raise HTTPException(status_code=500, detail="Modelo IA não carregado")

    try:
        row = [
            clima.get("temperature_2m"),
            clima.get("relativehumidity_2m"),
            clima.get("dewpoint_2m"),
            clima.get("surface_pressure"),
            clima.get("windspeed_10m"),
            clima.get("winddirection_10m"),
            clima.get("windgusts_10m"),
            clima.get("precipitation"),
            clima.get("shortwave_radiation"),
            clima.get("direct_normal_irradiance"),
            clima.get("evapotranspiration"),
            clima.get("soil_temperature_0cm"),
            clima.get("soil_moisture_0_to_1cm"),
            focos_reais.get("focos_50km"),
            focos_reais.get("focos_150km"),
            focos_reais.get("focos_300km"),
        ]

        X = np.array([row], dtype=float)

        focos_prev = float(modelo_pipeline.predict(X)[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao executar IA: {e}")

    # ------------------------------------------------------------
    # 5) Riscos: simples e híbrido
    # ------------------------------------------------------------
    # Risco simples (baseado apenas nos tercis)
    if focos_prev <= 50:
        risco_simples = "baixo"
    elif focos_prev <= 150:
        risco_simples = "medio"
    else:
        risco_simples = "alto"

    # Score híbrido
    temp = clima.get("temperature_2m") or 0
    score_h = (temp / 40) + (focos_prev / 300)

    if score_h <= 1:
        risco_h = "baixo"
    elif score_h <= 2:
        risco_h = "medio"
    else:
        risco_h = "alto"

    # ------------------------------------------------------------
    # 6) Retorno consolidado para o front
    # ------------------------------------------------------------
    return {
        "ok": True,
        "local": {
            "cidade": req.cidade,
            "lat": lat,
            "lon": lon,
            "resolved_by": loc_meta.get("resolved_by"),
            "display_name": loc_meta.get("display_name"),
        },
        "clima_atual": {
            "features": clima,
            "raw": clima_raw,
        },
        "focos_reais": focos_reais,
        "ia": {
            "modelo": MODEL_PATH,
            "focos_previstos_300km": focos_prev,
            "risco_simples": risco_simples,
            "score_hibrido": round(score_h, 3),
            "risco_hibrido": risco_h,
        }
    }

# ============================================================
# 🧩 MÓDULO 13 — Atualização de Alertas + Score Inteligente
# ============================================================

class AlertUpdateRequest(BaseModel):
    cidade: str | None = None
    lat: float | None = None
    lon: float | None = None

    raio_km: int = 150
    air_radius_m: int = 10000
    weather_provider: str = "open-meteo"

    severity: float | None = None
    duration: float | None = None
    frequency: float | None = None
    impact: float | None = None


@app.post("/api/alertas_update", tags=["Alertas"], summary="Atualiza alertas e calcula score inteligente")
def api_alertas_update(req: AlertUpdateRequest):
    """
    Fluxo:
      1) resolve localização
      2) coleta clima atual
      3) coleta qualidade do ar
      4) coleta focos INPE
      5) monta estrutura alert_obs
      6) computa score inteligente
      7) salva em NDJSON (companhia do sistema legacy)
      8) controla transição de nível
      9) devolve resposta consolidada
    """

    # --------------------------------------------------------
    # 1) Resolucao da localizacao
    # --------------------------------------------------------
    try:
        lat, lon, loc_meta = _resolve_location(
            cidade=req.cidade,
            lat=req.lat,
            lon=req.lon,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"Erro ao resolver localização: {e}")

    # --------------------------------------------------------
    # 2) Clima atual
    # --------------------------------------------------------
    try:
        clima_now = collect_weather_now(...)
        clima_feat = clima_now.get("features") or {}
    except Exception as e:
        clima_now = {"erro": str(e)}
        clima_feat = {}

    # --------------------------------------------------------
    # 3) Qualidade do ar
    # --------------------------------------------------------
    try:
        air_now   = {"fonte": "none", "features": {"pm25": None, "pm10": None}}
        air_feat = air_now.get("features") or {}
    except Exception as e:
        air_now = {"erro": str(e)}
        air_feat = {}

    # --------------------------------------------------------
    # 4) Focos INPE
    # --------------------------------------------------------
    try:
        focos_raw = inpe_focos_near(
            lat=lat,
            lon=lon,
            raio_km=req.raio_km,
            scope="diario",
            region="Brasil",
            limit=3000,
            timeout=HTTP_TIMEOUT,
        )
        focos_lista = (focos_raw.get("features") or {}).get("focos") or []
    except Exception as e:
        focos_raw = {"erro": str(e)}
        focos_lista = []

    focos_limpos = []
    for f in focos_lista:
        try:
            dist = float(f.get("dist_km"))
        except Exception:
            dist = None

        focos_limpos.append({
            "lat": f.get("lat"),
            "lon": f.get("lon"),
            "dist_km": dist,
            "uf": f.get("uf"),
            "municipio": f.get("municipio"),
            "frp": f.get("frp"),
            "satelite": f.get("satelite"),
            "data_hora_gmt": f.get("data_hora_gmt"),
        })

    # --------------------------------------------------------
    # 5) Montagem do dicionário alert_obs
    # --------------------------------------------------------
    alert_obs = {
        # variáveis inseridas manualmente
        "severity": req.severity,
        "duration": req.duration,
        "frequency": req.frequency,
        "impact": req.impact,

        # clima e precipitação
        "precipitation": clima_feat.get("precipitation"),

        "meta": {
            "precipitation": clima_feat.get("precipitation"),
        },

        # ar
        "pm25": air_feat.get("pm25"),
        "pm10": air_feat.get("pm10"),

        # foco (quantidade total)
        "focos_total": len(focos_limpos),
    }

    # --------------------------------------------------------
    # 6) Score Inteligente
    # --------------------------------------------------------
    score_res = compute_alert_score(alert_obs)

    # --------------------------------------------------------
    # 7) Persistência NDJSON
    # --------------------------------------------------------
    alert_id = (
        req.cidade.lower().replace(" ", "_")
        if req.cidade else f"{lat:.4f},{lon:.4f}"
    )

    save_alert_score(
        alert_id=alert_id,
        score=score_res.score,
        level=score_res.level,
        alert_obs=alert_obs,
        params={
            "lat": lat,
            "lon": lon,
            "cidade": req.cidade,
            "weather_provider": req.weather_provider,
        },
    )

    # --------------------------------------------------------
    # 8) Transição de nível
    # --------------------------------------------------------
    handle_level_transition(
        alert_id=alert_id,
        new_level=score_res.level,
        score=score_res.score,
        alert_obs=alert_obs,
    )

    # --------------------------------------------------------
    # 9) Retorno final
    # --------------------------------------------------------
    return {
        "ok": True,
        "local": {
            "cidade": req.cidade,
            "lat": lat,
            "lon": lon,
            "resolved_by": loc_meta.get("resolved_by"),
            "display_name": loc_meta.get("display_name"),
        },
        "clima": clima_now,
        "air_quality": air_now,
        "focos": {
            "total": len(focos_limpos),
            "items": focos_limpos,
        },
        "alert_obs": alert_obs,
        "score": {
            "valor": score_res.score,
            "nivel": score_res.level,
            "detalhes": score_res.breakdown,
        },
    }

# ============================================================
# 🧩 MÓDULO 14 — Rotas Administrativas / Diagnóstico (v2)
# ============================================================

from fastapi import APIRouter
from sqlmodel import Session, select

router_admin = APIRouter(prefix="/admin", tags=["Admin"])

# ------------------------------------------------------------
# 14.1 — Status geral
# ------------------------------------------------------------
@router_admin.get("/status", summary="Status geral do backend")
def admin_status():
    return {
        "ok": True,
        "service": "AmazonSafe Backend",
        "version": "1.0",
        "model_loaded": modelo_pipeline is not None,
        "db_url": DB_URL,
        "cache_ttl_sec": CACHE_TTL_SEC,
        "default_scope": INPE_DEFAULT_SCOPE,
        "default_region": INPE_DEFAULT_REGION,
    }

# ------------------------------------------------------------
# 14.2 — Diagnóstico do modelo IA
# ------------------------------------------------------------
@router_admin.get("/model_state", summary="Diagnóstico do modelo IA")
def admin_model_state():
    if modelo_pipeline is None:
        return {"loaded": False, "msg": "Modelo IA não carregado."}

    feats = list(getattr(modelo_pipeline, "feature_names_in_", []))
    return {
        "loaded": True,
        "model_path": MODEL_PATH,
        "n_features": len(feats),
        "feature_names": feats,
    }

# ------------------------------------------------------------
# 14.3 — Últimos registros persistidos
# ------------------------------------------------------------
@router_admin.get("/last_records", summary="Últimas observações persistidas")
def admin_last_records():
    try:
        w = get_last_weather()
        f = get_last_fire()
    except Exception as e:
        return {"ok": False, "error": str(e)}

    return {
        "ok": True,
        "last_weather": w.dict() if w else None,
        "last_fire": f.dict() if f else None,
    }

# ------------------------------------------------------------
# 14.4 — Teste de conexão com serviços externos (NOVO)
# ------------------------------------------------------------
@router_admin.get("/external_check", summary="Verifica serviços externos (Open-Meteo / INPE)")
def admin_external_check():
    status = {}

    # Teste Open-Meteo (forecast)
    try:
        clima = get_meteo(DEFAULT_LAT, DEFAULT_LON)
        status["open_meteo"] = "ok" if clima else "fail"
    except Exception as e:
        status["open_meteo"] = f"error: {e}"

    # Teste INPE focos
    try:
        d = inpe_focos_near(
            DEFAULT_LAT,
            DEFAULT_LON,
            50,
            INPE_DEFAULT_SCOPE,
            INPE_DEFAULT_REGION,
            10,
            timeout=10
        )
        status["inpe"] = "ok" if d else "fail"
    except Exception as e:
        status["inpe"] = f"error: {e}"

    # OpenAQ REMOVIDO
    status["openaq"] = "not_available"

    return status

# ------------------------------------------------------------
# 14.5 — Teste Geocoder
# ------------------------------------------------------------
@router_admin.get("/geocode_test", summary="Testa geocodificação")
def admin_geocode_test(cidade: str):
    info = geocode_city(cidade)
    if not info:
        return {"ok": False, "msg": f"Não foi possível geocodificar '{cidade}'"}
    return {"ok": True, "result": info}

# ------------------------------------------------------------
# 14.6 — Limpeza de cache
# ------------------------------------------------------------
@router_admin.post("/clear_cache", summary="Limpa caches internos (somente local)")
def admin_clear_cache():
    global _CACHE
    try:
        _CACHE.clear()
    except Exception:
        _CACHE = {}

    return {"ok": True, "cache_size": len(_CACHE)}

# ------------------------------------------------------------
# Registrar router
# ------------------------------------------------------------
app.include_router(router_admin)


# ============================================================
# 🧩 MÓDULO 15 — Logging, Auditoria e Telemetria Interna
# ============================================================

import json
import time
import os
from fastapi import Request
from typing import Dict, Any
from datetime import datetime
from fastapi import APIRouter

router_logs = APIRouter(prefix="/logs", tags=["Logs"])

# ------------------------------------------------------------
# 15.1 — Garantir diretório de logs
# ------------------------------------------------------------
def _ensure_logs_dir():
    d = "./runtime_data"
    os.makedirs(d, exist_ok=True)
    return d

LOG_PATH = os.path.join(_ensure_logs_dir(), "events.ndjson")

# ------------------------------------------------------------
# 15.2 — Função central de registro de log
# ------------------------------------------------------------
def log_event(event_type: str, message: str, extra: Dict[str, Any] | None = None):
    """
    Grava um evento de auditoria/telemetria no formato NDJSON.
    event_type: ex. "api_call", "error", "warning", "model_inference"
    message: msg principal
    extra: metadados adicionais (dict)
    """
    rec = {
        "ts": datetime.utcnow().isoformat() + "Z",
        "type": event_type,
        "message": message,
        "extra": extra or {},
    }

    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[LOG_ERROR] Falha ao escrever log: {e}")


# ------------------------------------------------------------
# 15.3 — Middleware opcional para registrar chamadas de API
# ------------------------------------------------------------
@app.middleware("http")
async def audit_api_calls(request: Request, call_next):
    """
    Middleware leve que registra:
    - Rota acessada
    - Método HTTP
    - Tempo de execução
    - Código de resposta
    """
    start = time.time()
    path = request.url.path
    method = request.method

    # Executa rota
    response = await call_next(request)
    duration = (time.time() - start) * 1000  # ms

    # Registra evento
    log_event(
        "api_call",
        f"{method} {path}",
        {
            "duration_ms": round(duration, 2),
            "status_code": response.status_code,
            "client": request.client.host if request.client else None,
        }
    )
    return response


# ------------------------------------------------------------
# 15.4 — Funções utilitárias para logs críticos
# ------------------------------------------------------------
def log_error(message: str, extra: Dict[str, Any] | None = None):
    log_event("error", message, extra)


def log_warning(message: str, extra: Dict[str, Any] | None = None):
    log_event("warning", message, extra)


def log_model_inference(lat: float, lon: float, result: Any, ref_date: str):
    log_event(
        "model_inference",
        "Inferência do modelo executada",
        {
            "lat": lat,
            "lon": lon,
            "ref_date": ref_date,
            "result": result,
        }
    )


# ------------------------------------------------------------
# 15.5 — Endpoint administrativo: últimos N logs
# ------------------------------------------------------------
@router_logs.get("/tail", summary="Lê os últimos eventos de log")
def tail_logs(n: int = 50):
    """
    Lê os últimos N registros do arquivo de logs.
    """
    if not os.path.exists(LOG_PATH):
        return {"ok": False, "msg": "Nenhum log encontrado."}

    try:
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            lines = f.readlines()[-n:]
            events = [json.loads(line) for line in lines]
    except Exception as e:
        return {"ok": False, "error": str(e)}

    return {
        "ok": True,
        "count": len(events),
        "events": events,
        "path": LOG_PATH,
    }


# ------------------------------------------------------------
# 15.6 — Endpoint para registrar um log manual (debug/teste)
# ------------------------------------------------------------
@router_logs.post("/push", summary="Registro manual de log (debug)")
def push_log(event_type: str, message: str):
    log_event(event_type, message)
    return {"ok": True, "msg": "Log registrado."}


# ------------------------------------------------------------
# 15.7 — Registrar router na API principal
# ------------------------------------------------------------
app.include_router(router_logs)

# ============================================================
# 🧩 MÓDULO 16 — Healthcheck, Métricas e Autoverificação (v2)
# ============================================================

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse
import os
import time
import json
import numpy as np

router_health = APIRouter(prefix="/system", tags=["Sistema"])




# ------------------------------------------------------------
# 16.0 — Helper UTC
# ------------------------------------------------------------
def now_utc():
    return _now_utc()   # usa o helper oficial do módulo 6


# ------------------------------------------------------------
# 16.1 — /health básico
# ------------------------------------------------------------
@router_health.get("/health", summary="Healthcheck básico")
def api_health():
    return {
        "status": "ok",
        "timestamp": now_utc().isoformat().replace("+00:00", "Z"),
        "model_loaded": modelo_pipeline is not None,
        "db_url": DB_URL,
    }
# ------------------------------------------------------------
# 16.1A — Healthcheck raiz (compatibilidade com Render)
# ------------------------------------------------------------

@app.get("/health", tags=["Sistema"], summary="Healthcheck raiz (compatibilidade)")
def root_health():
    return {"ok": True, "status": "online", "path": "/system/health"}
# ------------------------------------------------------------
# 16.2 — Teste rápido do modelo IA
# ------------------------------------------------------------
@router_health.get("/health/model", summary="Testa o modelo carregado")
def api_health_model():
    if modelo_pipeline is None:
        return {"ok": False, "error": "Modelo IA não carregado"}

    try:
        n = len(getattr(modelo_pipeline, "feature_names_in_", MODEL_FEATURES))
        test_x = np.zeros((1, n))
        pred = modelo_pipeline.predict(test_x)[0]
        return {"ok": True, "prediction_test": float(pred)}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ------------------------------------------------------------
# 16.3 — Teste de provedores externos
# ------------------------------------------------------------
@router_health.get("/health/providers", summary="Testa Open-Meteo / INPE")
def api_health_providers():

    results = {}

    # Open-Meteo Forecast
    try:
        clima = get_meteo(DEFAULT_LAT, DEFAULT_LON)
        results["open_meteo"] = {
            "ok": True,
            "temp": clima.get("temperature_2m")
        }
    except Exception as e:
        results["open_meteo"] = {"ok": False, "error": str(e)}

    # OpenAQ removido
    results["openaq"] = {"ok": False, "error": "OpenAQ removido do projeto"}

    # INPE
    try:
        d = inpe_focos_near(
            DEFAULT_LAT, DEFAULT_LON,
            raio_km=50,
            scope=INPE_DEFAULT_SCOPE,
            region=INPE_DEFAULT_REGION,
            timeout=5
        )
        count = len((d.get("features") or {}).get("focos") or [])
        results["inpe"] = {"ok": True, "count": count}
    except Exception as e:
        results["inpe"] = {"ok": False, "error": str(e)}

    return {"ok": True, "providers": results}


# ------------------------------------------------------------
# 16.4 — Teste de Geocodificação
# ------------------------------------------------------------
@router_health.get("/health/geocode", summary="Testa geocodificação")
def api_health_geocode():
    try:
        info = geocode_city("Belém, PA")
        return {"ok": True, "result": info}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ------------------------------------------------------------
# 16.5 — Teste de escrita em disco
# ------------------------------------------------------------
@router_health.get("/health/disk", summary="Testa escrita no disco persistente")
def api_health_disk():
    try:
        d = "./runtime_data/healthcheck"
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, "test.txt")
        ts = now_utc().isoformat()

        with open(path, "w", encoding="utf-8") as f:
            f.write(ts)

        return {"ok": True, "file": path, "content": ts}

    except Exception as e:
        return {"ok": False, "error": str(e)}


# ------------------------------------------------------------
# 16.6 — Teste SQLite
# ------------------------------------------------------------
@router_health.get("/health/db", summary="Testa conexão com SQLite")
def api_health_db():
    try:
        with engine.connect() as conn:
            conn.execute("SELECT 1")
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ------------------------------------------------------------
# 16.7 — /metrics estilo Prometheus
# ------------------------------------------------------------
@router_health.get("/metrics", summary="Métricas simples estilo Prometheus")
def api_metrics():

    metrics = {
        "uptime_seconds": time.time() - START_TIME,
        "model_loaded": 1 if modelo_pipeline is not None else 0,
        "cache_ttl_seconds": CACHE_TTL_SEC,
        "default_lat": DEFAULT_LAT,
        "default_lon": DEFAULT_LON,
    }

    lines = [f"{k} {v}" for k, v in metrics.items()]
    return PlainTextResponse("\n".join(lines))


# ------------------------------------------------------------
# 16.8 — Registrar router
# ------------------------------------------------------------
app.include_router(router_health)

# ------------------------------------------------------------
# 16.9 — Marcar início do servidor
# ------------------------------------------------------------
START_TIME = time.time()


# ============================================================
# 🧩 MÓDULO FINAL — Execução (Local / Render)
# ============================================================

import os
import sys
import uvicorn

if __name__ == "__main__":
    """
    Permite rodar localmente com:
        python main.py

    Observação:
    👉 No Render, este bloco é ignorado.
    👉 O Render executa automaticamente:
        uvicorn main:app --host 0.0.0.0 --port $PORT
    conforme definido na plataforma.
    """
    port = int(os.getenv("PORT", 8000))

    print("\n==============================================")
    print("🚀 AmazonSafe API iniciando localmente")
    print(f"➡ Porta: {port}")
    print("➡ Ambiente:", os.getenv("ENV", "development"))
    print("==============================================\n")

    try:
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=port,
            reload=True,       # hot reload no ambiente local
            log_level="info",
        )
    except Exception as e:
        print("❌ Erro ao iniciar o servidor:", e)
        sys.exit(1)


# ============================================================
# Nota:
# No Render NÃO executar python main.py.
# A plataforma usa automaticamente:
#   uvicorn main:app --host 0.0.0.0 --port $PORT
# ============================================================
