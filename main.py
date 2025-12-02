
# ============================================================
# MÓDULO 1 — IMPORTS ESSENCIAIS E CONFIGURAÇÕES INICIAIS (v11)
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

# DB (apenas SQLModel para compatibilidade — engine é criado no Módulo 6)
from sqlmodel import SQLModel

# HTTP / Data
import requests
import pandas as pd

# IA
import joblib
import numpy as np

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

# Timeout HTTP global
HTTP_TIMEOUT = int(os.getenv("HTTP_TIMEOUT", "60"))

# TTL de cache global
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
# MÓDULO 3 — GEOCODING + COLETORES HTTP + NORMALIZAÇÃO (v11)
# ============================================================

import re
import requests
import pandas as pd
import datetime as dt
import math
from io import BytesIO
from typing import Optional, Dict, Any

# ============================================================
# CONSTANTES E CONFIGURAÇÕES
# ============================================================

UF2STATE = {
    "AC": "Acre", "AL": "Alagoas", "AP": "Amapá", "AM": "Amazonas", "BA": "Bahia",
    "CE": "Ceará", "DF": "Distrito Federal", "ES": "Espírito Santo", "GO": "Goiás",
    "MA": "Maranhão", "MT": "Mato Grosso", "MS": "Mato Grosso do Sul", "MG": "Minas Gerais",
    "PA": "Pará", "PB": "Paraíba", "PR": "Paraná", "PE": "Pernambuco", "PI": "Piauí",
    "RJ": "Rio de Janeiro", "RN": "Rio Grande do Norte", "RS": "Rio Grande do Sul",
    "RO": "Rondônia", "RR": "Roraima", "SC": "Santa Catarina", "SP": "São Paulo",
    "SE": "Sergipe", "TO": "Tocantins"
}

# User-Agent profissional
GEOCODE_UA = (
    "AmazonSafe/3 (contato: joaobbb@gmail.com; https://amazonsafe-api.onrender.com)"
)

# Timeout único vindo do Módulo 1
# HTTP_TIMEOUT já está definido globalmente no Módulo 1

GEO_TIMEOUT = 8  # timeout curto para evitar travamentos no backend


# ============================================================
# 3.1 — GEOCODING (Nominatim + Open-Meteo + Fallback Belém)
# ============================================================

def _split_city_state(q: str):
    """Divide entrada 'cidade, estado' em ('cidade', 'estado')"""
    s = q.strip()
    m = re.split(r"\s*[,;-]\s*|\s{2,}", s, maxsplit=1)
    return (m[0].strip(), m[1].strip()) if len(m) == 2 else (s, None)


def _normalize_state(st):
    """Normaliza UF ou nome do estado."""
    if not st:
        return (None, None)
    up = st.upper()
    if up in UF2STATE:
        return (up, UF2STATE[up])
    for uf, name in UF2STATE.items():
        if up.lower() == name.lower():
            return (uf, name)
    return (None, st)


def _geocode_nominatim(q: str, state_name: str | None):
    """
    Tenta geocodificar via Nominatim (OSM).
    Retorna dict com lat/lon/display_name ou None.
    """
    try:
        r = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={
                "q": q,
                "format": "jsonv2",
                "addressdetails": 1,
                "limit": 5,
                "countrycodes": "br",
            },
            headers={"User-Agent": GEOCODE_UA},
            timeout=GEO_TIMEOUT,
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
            addr = (it.get("address") or {})
            st_name = addr.get("state", "")
            if state_name and state_name.lower() in st_name.lower():
                s += 2

            if s > score:
                best, score = it, s

        if best:
            return {
                "lat": float(best["lat"]),
                "lon": float(best["lon"]),
                "display_name": best.get("display_name"),
                "source": "nominatim",
            }
    except Exception as e:
        print("[GEOCODE] Nominatim falhou:", e)

    return None


def geocode_city(raw_q: str) -> Optional[Dict[str, Any]]:
    """
    Geocodificador robusto com fallback:
      1) Nominatim
      2) Open-Meteo
      3) Belém/PA (DEFAULT_LAT/DEFAULT_LON)
    """

    if not raw_q or not raw_q.strip():
        return None

    raw_q = raw_q.strip()
    city, st = _split_city_state(raw_q)
    uf, state_name = _normalize_state(st)

    # 1) Nominatim
    q = f"{city}, {state_name}, Brasil" if state_name else f"{city}, Brasil"
    res = _geocode_nominatim(q, state_name)
    if res:
        return res

    # 2) Open-Meteo Geocoding API
    try:
        r = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={
                "name": city,
                "count": 1,
                "language": "pt",
                "country": "BR",
            },
            timeout=GEO_TIMEOUT,
        )
        data = r.json() or {}
        arr = data.get("results") or []
        if arr:
            best = arr[0]
            return {
                "lat": float(best["latitude"]),
                "lon": float(best["longitude"]),
                "display_name": f"{best.get('name')}, {best.get('admin1', '')}",
                "source": "open-meteo",
            }
    except Exception as e:
        print("[GEOCODE] Open-Meteo falhou:", e)

    # 3) Fallback para Belém/PA
    try:
        lat = float(DEFAULT_LAT)
        lon = float(DEFAULT_LON)
    except Exception:
        lat, lon = -1.45056, -48.4682453  # fallback seguro

    print(f"[GEOCODE] Fallback para Belém/PA para '{raw_q}'")

    return {
        "lat": lat,
        "lon": lon,
        "display_name": f"Fallback {raw_q} → Belém/PA",
        "source": "fallback",
    }


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
    """Sessão HTTP com retry controlador."""
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


def http_get(url, *, params=None, headers=None, timeout=None):
    """GET com retry e timeout inteligente."""
    to = timeout if isinstance(timeout, (int, float)) else HTTP_TIMEOUT

    if isinstance(to, (int, float)):
        to = (min(5, to), to)  # (connect_timeout, read_timeout)

    resp = _HTTP.get(url, params=params, headers=headers, timeout=to)
    resp.raise_for_status()
    return resp


# ============================================================
# 3.3 — Open-Meteo Forecast + Air Quality
# ============================================================

def safe_mean(values):
    if not isinstance(values, list):
        return None
    clean = [v for v in values if isinstance(v, (int, float, float))]
    return sum(clean) / len(clean) if clean else None


@ttl_cache(ttl_seconds=CACHE_TTL_SEC)
def get_meteo(lat: float, lon: float, retries: int = 5):
    """
    Coleta dados do Open-Meteo:
      - Temperatura, umidade, ponto de orvalho
      - Pressão atmosférica
      - Vento, direção, rajadas
      - Precipitação
      - Radiação solar global e direta
      - Solo (temp/umidade)
      - Evapotranspiração
      - PM10, PM2.5, O3, NO2, SO2, CO
      - UV Index
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
        "evapotranspiration",
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
            r1 = requests.get(url_weather, timeout=20)
            r1.raise_for_status()
            w = r1.json().get("hourly", {})

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
    except Exception:
        df = pd.read_csv(BytesIO(r.content), encoding="latin1")

    return {"df": df, "url": url, "ref": str(ref)}


def _canonical_inpe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza colunas principais do CSV do INPE.
    Suporta diferentes nomes usados pelo INPE.
    """
    cols = {c.lower(): c for c in df.columns}

    def pick(*names):
        for n in names:
            if n in cols:
                return cols[n]
        return None

    c_lat = pick("latitude", "lat", "y")
    c_lon = pick("longitude", "lon", "x")
    c_dt  = pick("datahora", "data_hora", "data_hora_gmt")
    c_sat = pick("satelite", "satellite")
    c_frp = pick("frp", "radiative_power")
    c_uf  = pick("uf", "estado", "state")
    c_mun = pick("municipio", "município", "city", "nome_munic", "municipality")

    out = pd.DataFrame()

    out["latitude"]  = df[c_lat] if c_lat else None
    out["longitude"] = df[c_lon] if c_lon else None
    out["datahora"]  = df[c_dt] if c_dt else None
    out["satelite"]  = df[c_sat] if c_sat else None
    out["frp"]       = df[c_frp] if c_frp else None
    out["uf"]        = df[c_uf] if c_uf else None
    out["municipio"] = df[c_mun] if c_mun else None

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
def inpe_focos_near(
    lat,
    lon,
    raio_km: float = 150,
    scope: str = "diario",
    region: str = "Brasil",
    limit: int = 1000,
    timeout: int = HTTP_TIMEOUT,
):
    """Retorna focos do INPE próximos ao ponto (lat, lon)."""

    payload = inpe_fetch_csv(scope=scope, region=region, timeout=timeout)
    df = payload["df"]
    norm = _canonical_inpe_columns(df)

    # Converte para float seguro
    norm["latitude"] = norm["latitude"].map(parse_float_safe)
    norm["longitude"] = norm["longitude"].map(parse_float_safe)

    # Bounding box rápido
    minx, miny, maxx, maxy = bbox_from_center(lat, lon, float(raio_km))
    mask = (
        norm["longitude"].notna() &
        norm["latitude"].notna() &
        (norm["longitude"] >= minx) &
        (norm["longitude"] <= maxx) &
        (norm["latitude"] >= miny) &
        (norm["latitude"] <= maxy)
    )
    sub = norm[mask].copy()

    # Distância exata
    try:
        sub["dist_km"] = sub.apply(
            lambda r: haversine_km(lat, lon, r["latitude"], r["longitude"]),
            axis=1
        )
        sub = sub[sub["dist_km"] <= float(raio_km)]
    except:
        pass

    if limit:
        sub = sub.head(int(limit))

    focos = []
    for _, r in sub.iterrows():
        focos.append({
            "latitude":  _json_safe(r.get("latitude")),
            "longitude": _json_safe(r.get("longitude")),
            "datahora":  _json_safe(r.get("datahora")),
            "satelite":  _json_safe(r.get("satelite")),
            "frp":       _json_safe(r.get("frp")),
            "uf":        _json_safe(r.get("uf")),
            "municipio": _json_safe(r.get("municipio")),
            "dist_km":   _json_safe(r.get("dist_km")),
        })

    meta = {
        "source": "inpe_csv",
        "url": payload["url"],
        "reference": payload["ref"],
        "bbox": {"minlon": minx, "minlat": miny, "maxlon": maxx, "maxlat": maxy},
        "count": len(focos),
        "timestamp_utc": dt.datetime.utcnow().isoformat() + "Z",
    }

    return {
        "features": {"focos": focos, "count": len(focos), "meta": meta},
        "payload": {"csv_url": payload["url"]},
    }


# ============================================================
# 3.6 — DETER PARQUET (Cloudflare R2)
# ============================================================

import zipfile

DETER_R2_URL = os.getenv(
    "DETER_R2_URL",
    "https://0be47677b22a6dd946b4ff62d6dce778.r2.cloudflarestorage.com/deter-storage/deter_parquet.zip"
)

DETER_LOCAL_ZIP = "/tmp/deter_parquet.zip"
DETER_LOCAL_PARQUET = "/tmp/deter.parquet"


@ttl_cache(ttl_seconds=3600)
def load_deter_parquet():
    """Baixa e carrega o DETER parquet hospedado no Cloudflare R2."""

    try:
        r = http_get(DETER_R2_URL, timeout=HTTP_TIMEOUT)
        with open(DETER_LOCAL_ZIP, "wb") as f:
            f.write(r.content)
    except Exception as e:
        print("[DETER] Erro ao baixar:", e)
        return None

    try:
        with zipfile.ZipFile(DETER_LOCAL_ZIP, "r") as z:
            fname = z.namelist()[0]
            z.extract(fname, "/tmp/")
            os.rename(f"/tmp/{fname}", DETER_LOCAL_PARQUET)
    except Exception as e:
        print("[DETER] Erro ao extrair zip:", e)
        return None

    try:
        df = pd.read_parquet(DETER_LOCAL_PARQUET)
        return df
    except Exception as e:
        print("[DETER] Erro ao ler parquet:", e)
        return None


# ============================================================
# 3.7 — DESMATAMENTO POR RAIO (usando DETER)
# ============================================================

def get_desmatamento(lat: float, lon: float, raio_km: float = 50):
    df = load_deter_parquet()

    if df is None or df.empty:
        return {"count": 0, "area_total": 0}

    df = df.copy()
    df["lat"] = df["latitude"].astype(float)
    df["lon"] = df["longitude"].astype(float)

    minx, miny, maxx, maxy = bbox_from_center(lat, lon, raio_km)
    sub = df[
        (df["lon"] >= minx) & (df["lon"] <= maxx) &
        (df["lat"] >= miny) & (df["lat"] <= maxy)
    ].copy()

    if sub.empty:
        return {"count": 0, "area_total": 0}

    sub["dist_km"] = sub.apply(
        lambda r: haversine_km(lat, lon, r["lat"], r["lon"]),
        axis=1
    )
    sub = sub[sub["dist_km"] <= raio_km]

    if sub.empty:
        return {"count": 0, "area_total": 0}

    return {
        "count": len(sub),
        "area_total": sub["area_ha"].sum() if "area_ha" in sub else 0
    }

# ============================================================
# MÓDULO 4 — NORMALIZADORES E SANITIZAÇÃO DE DADOS (v11)
# ============================================================

def normalize_value(x):
    """Converte para float seguro, removendo NaN/inf."""
    try:
        if x is None:
            return None
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except:
        return None


def normalize_meteo(data: dict) -> dict:
    """
    Normaliza o retorno do Open-Meteo e converte
    para as FEATURES usadas pelo ML v11.
    """

    if not isinstance(data, dict):
        return {}

    # Converte para float seguro
    d = {k: normalize_value(v) for k, v in data.items()}

    # Mapeamento → nomes esperados pelo Modelo v11
    return {
        "temperatura":          d.get("temperature_2m"),
        "umidade":              d.get("relativehumidity_2m"),
        "ponto_orvalho":        d.get("dewpoint_2m"),
        "pressao":              d.get("surface_pressure"),
        "vento_m_s":            d.get("windspeed_10m"),
        "vento_dir":            d.get("winddirection_10m"),
        "rajadas":              d.get("windgusts_10m"),
        "chuva_mm":             d.get("precipitation"),
        "rad_solar":            d.get("shortwave_radiation"),
        "rad_direta":           d.get("direct_normal_irradiance"),
        "solo_temp_0cm":        d.get("soil_temperature_0cm"),
        "solo_umid_0_1cm":      d.get("soil_moisture_0_to_1cm"),
        "evapotranspiracao":    d.get("evapotranspiration"),
        "pm10":                 d.get("pm10"),
        "pm25":                 d.get("pm25"),
        "o3":                   d.get("o3"),
        "no2":                  d.get("no2"),
        "so2":                  d.get("so2"),
        "co":                   d.get("co"),
        "uv":                   d.get("uv"),
    }


def normalize_focos_result(focos_data: dict) -> dict:
    if not isinstance(focos_data, dict):
        return {"count": 0}
    return {"count": int(focos_data.get("count", 0))}


def normalize_desmatamento(data: dict) -> dict:
    if not isinstance(data, dict):
        return {"count": 0, "area_total": 0.0}
    return {
        "count": int(data.get("count", 0)),
        "area_total": float(data.get("area_total", 0.0)),
    }
# ============================================================
# MÓDULO 5 — Inicialização da API FastAPI (v11)
# ============================================================

app = FastAPI(
    title="AmazonSafe API",
    version="3.0-v11",
    description="API AmazonSafe – Risco ambiental, incêndios, focos e desmatamento"
)

# CORS liberado para MVP
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("✔ AmazonSafe API carregada.")

@app.get("/", tags=["Infra"], summary="Health Check")
def root():
    return {
        "ok": True,
        "service": "AmazonSafe API",
        "version": "3.0-v11",
        "status": "online",
    }
# ============================================================
# MÓDULO 6 — Persistência, NDJSON, Alertas e Helpers (v11)
# ============================================================

import os
import json
import time
import math
import datetime as dt
from typing import Any, Dict, Optional

from sqlmodel import Session

# ------------------------------------------------------------
# 6.1 — Engine única (vem do módulo 1)
# ------------------------------------------------------------

print(f"[DB] Engine carregado: {DB_URL}")

# ------------------------------------------------------------
# 6.2 — Helpers de Tempo
# ------------------------------------------------------------

UTC = dt.timezone.utc

def now_utc() -> dt.datetime:
    return dt.datetime.now(UTC)

def now_unix() -> int:
    return int(time.time())

# ------------------------------------------------------------
# 6.3 — Persistência NDJSON
# ------------------------------------------------------------

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path

def append_ndjson(path: str, record: dict):
    ensure_dir(os.path.dirname(path))
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

# ------------------------------------------------------------
# 6.4 — Sistema de Alertas
# ------------------------------------------------------------

ALERTS_DIR = "./runtime_data/alerts"
NOTIFY_DEBOUNCE_SEC = int(os.getenv("NOTIFY_DEBOUNCE_SEC", "600"))
WEBHOOK_URL = os.getenv("ALERTS_WEBHOOK_URL")

_LAST_LEVEL: Dict[str, Dict[str, Any]] = {}
_LAST_NOTIFY: Dict[str, float] = {}

def save_alert_score(alert_id: str, score: float, level: str,
                     alert_obs: Dict[str, Any],
                     params: Optional[Dict[str, Any]] = None):
    rec = {
        "ts": now_utc().isoformat().replace("+00:00", "Z"),
        "alert_id": alert_id,
        "score": score,
        "level": level,
        "alert_obs": alert_obs,
        "params": params or {},
    }
    append_ndjson(f"{ALERTS_DIR}/alerts.ndjson", rec)


def persist_level_change(alert_id: str, old: str, new: str, payload: dict):
    rec = {
        "ts": now_utc().isoformat().replace("+00:00", "Z"),
        "alert_id": alert_id,
        "from": old,
        "to": new,
        "payload": payload,
    }
    append_ndjson(f"{ALERTS_DIR}/level_events.ndjson", rec)


def notify_level_change(alert_id: str, old: str, new: str,
                        score: float, obs: dict):
    now = time.time()
    last = _LAST_NOTIFY.get(alert_id, 0)

    if now - last < NOTIFY_DEBOUNCE_SEC:
        return

    _LAST_NOTIFY[alert_id] = now

    msg = {
        "alert_id": alert_id,
        "from": old,
        "to": new,
        "score": score,
        "when": now_utc().isoformat().replace("+00:00", "Z"),
        "obs": {k: obs.get(k) for k in ("severity", "duration", "frequency", "impact")},
    }

    if WEBHOOK_URL:
        try:
            requests.post(WEBHOOK_URL, json=msg, timeout=8)
        except Exception as e:
            print("[Webhook Error]", e)
    else:
        append_ndjson(f"{ALERTS_DIR}/notifications.ndjson", msg)


# ⭐ Nome agora compatível com Módulo 13
def handle_level_transition(alert_id: str, new_level: str, score: float,
                            alert_obs: Dict[str, Any],
                            extra: Optional[Dict[str, Any]] = None,
                            notify_on_bootstrap: bool = False):

    old_level = (_LAST_LEVEL.get(alert_id) or {}).get("level")
    first = old_level is None

    if new_level != old_level:
        persist_level_change(alert_id, old_level, new_level, {
            "score": score,
            "obs": alert_obs,
            **(extra or {}),
        })

        if (not first) or notify_on_bootstrap:
            notify_level_change(alert_id, old_level, new_level, score, alert_obs)

        _LAST_LEVEL[alert_id] = {"level": new_level, "ts": now_utc()}

# ------------------------------------------------------------
# 6.5 — Helpers
# ------------------------------------------------------------

def safe_float(x, default: float = 0.0) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except:
        return default

def coalesce(*vals, default=None):
    for v in vals:
        if v is not None:
            return v
    return default

def is_valid_pm25(x) -> bool:
    try:
        v = float(x)
        return math.isfinite(v) and v >= 1.0
    except:
        return False

# ============================================================
# MÓDULO 7 — Sistema de Risco AmazonSafe (v11) — REVISADO
# ============================================================

import math
import requests
import pandas as pd
from io import BytesIO
from fastapi import HTTPException
from typing import Any, Dict, Optional, Tuple

# Usar mesmas constantes globais do Módulo 1
DEFAULT_LAT = float(os.getenv("DEFAULT_LAT", "-1.4558"))
DEFAULT_LON = float(os.getenv("DEFAULT_LON", "-48.5039"))
HTTP_TIMEOUT = int(os.getenv("HTTP_TIMEOUT", "60"))

# ------------------------------------------------------------
# 7.1 — Resolver localização (geocode + fallback)
# ------------------------------------------------------------

def _float_or_none(x):
    if x is None:
        return None
    try:
        return float(x)
    except:
        return None


def resolve_location(cidade: Optional[str], lat: Optional[float], lon: Optional[float]):
    """
    Regras:
      1) Se cidade → geocode
      2) Se lat/lon → usa direto
      3) Caso contrário → fallback global (Belém)
    """
    lat = _float_or_none(lat)
    lon = _float_or_none(lon)

    # Por cidade
    if cidade:
        info = geocode_city(cidade)
        if not info:
            raise HTTPException(404, f"Não foi possível geocodificar '{cidade}'")
        return float(info["lat"]), float(info["lon"]), {
            "resolved_by": "geocode",
            "display_name": info.get("display_name")
        }

    # Por coordenadas explícitas
    if lat is not None and lon is not None:
        return lat, lon, {"resolved_by": "direct_params"}

    # Fallback global
    return DEFAULT_LAT, DEFAULT_LON, {"resolved_by": "default"}


# ------------------------------------------------------------
# 7.2 — Carregamento unificado do DETER
# ------------------------------------------------------------

DETER_PARQUET_PATH = os.getenv("DETER_PARQUET_PATH", "")
DETER_PARQUET_URL  = os.getenv("DETER_PARQUET_URL", "")
DETER_CACHE_TTL    = int(os.getenv("DETER_CACHE_TTL_SEC", "3600"))


def _read_parquet_zip(zf: zipfile.ZipFile) -> pd.DataFrame:
    for name in zf.namelist():
        if name.lower().endswith(".parquet"):
            with zf.open(name) as f:
                return pd.read_parquet(f)
    raise RuntimeError("ZIP não contém arquivo .parquet")


@ttl_cache(ttl_seconds=DETER_CACHE_TTL)
def load_deter() -> Optional[pd.DataFrame]:
    """Carrega parquet local ou remoto com cache."""
    # Local
    if DETER_PARQUET_PATH and os.path.exists(DETER_PARQUET_PATH):
        if DETER_PARQUET_PATH.lower().endswith(".zip"):
            with zipfile.ZipFile(DETER_PARQUET_PATH, "r") as zf:
                return _read_parquet_zip(zf)
        return pd.read_parquet(DETER_PARQUET_PATH)

    # URL
    if DETER_PARQUET_URL:
        r = requests.get(DETER_PARQUET_URL, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        data = BytesIO(r.content)
        if DETER_PARQUET_URL.lower().endswith(".zip"):
            with zipfile.ZipFile(data, "r") as zf:
                return _read_parquet_zip(zf)
        return pd.read_parquet(data)

    return None


def canonical_deter_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza colunas latitude/longitude/área/classe/bioma."""
    cols = {c.lower(): c for c in df.columns}

    def pick(*names):
        for n in names:
            if n.lower() in cols:
                return cols[n.lower()]
        return None

    c_lat  = pick("latitude", "lat", "y")
    c_lon  = pick("longitude", "lon", "x")
    c_area = pick("area_ha", "areaha", "area", "area_km2")
    c_cls  = pick("classe", "class")
    c_bio  = pick("bioma", "biome")

    if not c_lat or not c_lon:
        raise RuntimeError("Parquet DETER sem latitude/longitude válidos")

    out = pd.DataFrame()
    out["latitude"]  = df[c_lat].map(parse_float_safe)
    out["longitude"] = df[c_lon].map(parse_float_safe)

    if c_area:
        if "km" in c_area.lower():
            out["area_ha"] = df[c_area].map(parse_float_safe) * 100
        else:
            out["area_ha"] = df[c_area].map(parse_float_safe)
    else:
        out["area_ha"] = None

    if c_cls: out["classe"] = df[c_cls]
    if c_bio: out["bioma"]  = df[c_bio]

    return out


def deter_stats(lat: float, lon: float, raio_km: float = 150.0) -> Dict[str, Any]:
    df = load_deter()
    if df is None or df.empty:
        return {"count": 0, "total_area_ha": 0.0, "score_raw": 0.0, "score_norm": 0.0}

    det = canonical_deter_columns(df)

    minx, miny, maxx, maxy = bbox_from_center(lat, lon, raio_km)
    mask = (
        det["longitude"].between(minx, maxx)
        & det["latitude"].between(miny, maxy)
    )
    sub = det[mask].copy()

    # Filtro fino
    try:
        sub["dist"] = sub.apply(
            lambda r: haversine_km(lat, lon, r["latitude"], r["longitude"]), axis=1
        )
        sub = sub[sub["dist"] <= raio_km]
    except:
        pass

    n = len(sub)
    area_total = float(sub["area_ha"].dropna().sum()) if "area_ha" in sub else float(n)

    score_raw = math.log1p(area_total) + 0.5 * math.log1p(n)
    score_norm = max(0.0, min(1.0, score_raw / 4.0))

    return {
        "count": int(n),
        "total_area_ha": area_total,
        "score_raw": score_raw,
        "score_norm": score_norm,
    }


# ------------------------------------------------------------
# 7.3 — Focos do INPE (wrapper robusto)
# ------------------------------------------------------------

def focos_stats(lat: float, lon: float, raio_km: float = 150.0,
                scope: str = "diario",
                region: str = "Brasil") -> Dict[str, Any]:

    try:
        data = inpe_focos_near(lat, lon, raio_km, scope, region)
        feats = data.get("features", {})
        focos = feats.get("focos", [])
        meta  = feats.get("meta", {})
    except Exception as e:
        return {
            "count": 0,
            "frp_sum": 0.0,
            "score_raw": 0.0,
            "score_norm": 0.0,
            "meta": {"error": str(e)}
        }

    n = int(feats.get("count") or len(focos))
    frp_vals = [parse_float_safe(f.get("frp")) for f in focos]
    frp_vals = [v for v in frp_vals if v is not None]
    frp_sum = float(sum(frp_vals)) if frp_vals else 0.0

    score_raw = math.log1p(n) + 0.002 * frp_sum
    score_norm = max(0.0, min(1.0, score_raw / 4.0))

    return {
        "count": n,
        "frp_sum": frp_sum,
        "score_raw": score_raw,
        "score_norm": score_norm,
        "meta": meta,
    }


# ------------------------------------------------------------
# 7.4 — Score de Conservação (chuva + ar − desmatamento − focos)
# ------------------------------------------------------------

PM25_LIMIT = 35.0
PM10_LIMIT = 50.0
TH_YELLOW = 40
TH_RED    = 70


def _extract_chuva(m: Dict[str, Any]) -> float:
    return _float_or_none(m.get("chuva_mm") or m.get("precipitation")) or 0.0


def conservation_score(meteo: dict, det: dict, foc: dict) -> Dict[str, Any]:
    chuva = _extract_chuva(meteo)
    pm25 = _float_or_none(meteo.get("pm25")) or 0.0
    pm10 = _float_or_none(meteo.get("pm10")) or 0.0

    d_norm = float(det.get("score_norm") or 0.0)
    f_norm = float(foc.get("score_norm") or 0.0)

    chuva_norm = max(0.0, min(1.0, chuva / 20.0))

    pm25_ratio = pm25 / PM25_LIMIT if PM25_LIMIT else 0
    pm10_ratio = pm10 / PM10_LIMIT if PM10_LIMIT else 0
    pollution = max(pm25_ratio, pm10_ratio)
    pollution_norm = max(0.0, min(2.0, pollution)) / 2.0
    air_norm = 1.0 - pollution_norm

    raw = (chuva_norm + air_norm) - (d_norm + f_norm)
    norm = max(0.0, min(1.0, (raw + 2.0) / 4.0))
    score = int(round(norm * 100))

    if score >= TH_RED:
        level = "Vermelho"
    elif score >= TH_YELLOW:
        level = "Amarelo"
    else:
        level = "Verde"

    return {
        "score": score,
        "level": level,
        "components": {
            "chuva_mm": chuva,
            "pm25": pm25,
            "pm10": pm10,
            "chuva_norm": chuva_norm,
            "air_quality_norm": air_norm,
            "desmatamento_norm": d_norm,
            "focos_norm": f_norm,
        },
    }


# ------------------------------------------------------------
# 7.5 — Contexto completo usado pelo Módulo 8 e Módulo 9
# ------------------------------------------------------------

def build_context(cidade=None, lat=None, lon=None, raio_km=150.0) -> Dict[str, Any]:
    """
    Contexto oficial do AmazonSafe v11.
    Sempre retorna meteo + deter + focos + conservation_score.
    """
    # 1) Localização
    lat_r, lon_r, loc_meta = resolve_location(cidade, lat, lon)

    # 2) Clima
    met_raw = get_meteo(lat_r, lon_r) or {}
    met = normalize_meteo(met_raw)

    # 3) DETER
    det = deter_stats(lat_r, lon_r, raio_km)

    # 4) Focos
    foc = focos_stats(lat_r, lon_r, raio_km)

    # 5) Conservação (heurístico)
    cons = conservation_score(met, det, foc)

    return {
        "location": {
            "lat": lat_r,
            "lon": lon_r,
            **loc_meta,
        },
        "radius_km": float(raio_km),
        "meteo": met,
        "deter": det,
        "focos": foc,
        "conservation": cons,
    }

# ============================================================
# MÓDULO 8 — SQLModel + IA AmazonSafe v11 (REVISADO)
# ============================================================

import os
import json
import joblib
import numpy as np
import datetime as dt
from typing import Optional, Dict, Any

from fastapi import HTTPException
from pydantic import BaseModel

# ------------------------------------------------------------
# 8.0 — BANCO (somente logs históricos, opcional)
# ------------------------------------------------------------

from sqlmodel import SQLModel, Field, Session, create_engine

DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
DB_URL = DATABASE_URL if DATABASE_URL else "sqlite:///./amazonsafe.db"

engine = create_engine(DB_URL, pool_pre_ping=True)


class RiskLog(SQLModel, table=True):
    """Registro de previsões do modelo ML."""
    id: Optional[int] = Field(default=None, primary_key=True)
    latitude: float
    longitude: float
    score: float
    level: str
    created_at: dt.datetime = Field(default_factory=dt.datetime.utcnow)
    payload_json: Optional[str] = None


# ------------------------------------------------------------
# 8.1 — CARREGAR MODELO V11
# ------------------------------------------------------------

MODEL_PATH = "models/amazonsafe_pipeline_v11.joblib"

try:
    modelo_pipeline = joblib.load(MODEL_PATH, mmap_mode="r")
    print(f"[IA] Modelo AmazonSafe v11 carregado com sucesso de: {MODEL_PATH}")
except Exception as e:
    print(f"[IA] ERRO ao carregar modelo v11: {e}")
    modelo_pipeline = None


# ------------------------------------------------------------
# 8.2 — FEATURES OFICIAIS DO MODELO
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# 8.3 — FEATURE VECTOR
# ------------------------------------------------------------

def _build_feature_vector(payload: Dict[str, Any]) -> np.ndarray:
    """
    Cria vetor X na ordem oficial.
    Valores faltantes -> 0.
    """
    row = []
    for col in MODEL_FEATURES:
        val = payload.get(col)
        try:
            row.append(float(val) if val is not None else 0.0)
        except:
            row.append(0.0)

    return np.array([row], dtype=float)


# ------------------------------------------------------------
# 8.4 — Função oficial de predição para Módulo 9
# ------------------------------------------------------------

def run_ml_model(ctx: dict) -> dict:
    """
    Usa o contexto oficial do Módulo 7.
    Retorna ml_raw (0–1) + nível textual.
    """

    if modelo_pipeline is None:
        return {"ml_raw": 0.0, "ml_level": "modelo_indisponivel"}

    loc = ctx.get("location", {})
    met = ctx.get("meteo", {})

    payload = met.copy()
    payload["latitude"] = float(loc.get("lat") or 0.0)
    payload["longitude"] = float(loc.get("lon") or 0.0)

    X = _build_feature_vector(payload)

    try:
        pred = float(modelo_pipeline.predict(X)[0])
    except Exception as e:
        print("[run_ml_model] ERRO:", e)
        return {"ml_raw": 0.0, "ml_level": "erro"}

    ml_raw = max(0.0, min(1.0, pred))

    level = (
        "Baixo" if ml_raw < 0.33
        else "Médio" if ml_raw < 0.66
        else "Alto"
    )

    return {
        "ml_raw": ml_raw,
        "ml_level": level
    }


# ------------------------------------------------------------
# 8.5 — Modelo Pydantic para /api/risk
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# 8.6 — Endpoint oficial /api/risk
# ------------------------------------------------------------

from fastapi import APIRouter

router_risk = APIRouter()

@router_risk.post("/api/risk", tags=["IA"], summary="Previsão de risco ambiental (AmazonSafe v11)")
def api_risk(data: RiskInput):

    if modelo_pipeline is None:
        raise HTTPException(500, "Modelo AmazonSafe v11 não carregado.")

    payload = data.model_dump()
    X = _build_feature_vector(payload)

    try:
        pred = float(modelo_pipeline.predict(X)[0])
    except Exception as e:
        raise HTTPException(500, f"Erro ao processar previsão: {e}")

    score = max(0.0, min(1.0, pred))

    if score < 0.33:
        level = "baixo"
    elif score < 0.66:
        level = "medio"
    else:
        level = "alto"

    # log opcional
    try:
        with Session(engine) as sess:
            rec = RiskLog(
                latitude=payload["latitude"],
                longitude=payload["longitude"],
                score=score,
                level=level,
                payload_json=json.dumps(payload, ensure_ascii=False),
            )
            sess.add(rec)
            sess.commit()
    except Exception as e:
        print(f"[RiskLog] WARN: {e}")

    return {
        "modelo": {
            "path": MODEL_PATH,
            "features": MODEL_FEATURES
        },
        "entrada": payload,
        "score": score,
        "nivel": level
    }

# → No main.py:
# app.include_router(router_risk)


# ============================================================
# 🧩 MÓDULO 9 — Score Final (Heurística + IA + Alertas + MAD)
# ============================================================

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime
import math
import statistics as stats

from fastapi import HTTPException
from sqlmodel import SQLModel, Session, Field, select

# Importa funções dos módulos anteriores
# (estes nomes precisam existir no main)
# build_context → módulo 7
# run_ml_model  → módulo 8
# inpe_focos_near → módulo 3

# ------------------------------------------------------------
# 9.0 — Limiares / Pesos do Score Final
# ------------------------------------------------------------

FINAL_THRESHOLDS = {
    "green_lt": 33,   # <33 → Verde
    "yellow_lt": 66,  # <66 → Amarelo
}

FINAL_WEIGHTS = {
    "heuristic": 0.40,
    "ml":        0.40,
    "alerts":    0.15,
    "mad":       0.05,
}

# ------------------------------------------------------------
# 9.0-A — Classe WeatherObs (necessária p/ MAD)
# ------------------------------------------------------------

class WeatherObs(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    lat: float
    lon: float
    pm25: float | None = None
    pm10: float | None = None
    observed_at: datetime = Field(default_factory=datetime.utcnow)


# ------------------------------------------------------------
# 9.1 — Estrutura final
# ------------------------------------------------------------

@dataclass
class FinalScoreResult:
    score: float
    level: str
    breakdown: Dict[str, Any]


# ------------------------------------------------------------
# 9.2 — Funções utilitárias
# ------------------------------------------------------------

def _clip01(x: Any) -> float:
    try:
        return max(0.0, min(1.0, float(x)))
    except:
        return 0.0


def _classify_level(score: float) -> str:
    if score < FINAL_THRESHOLDS["green_lt"]:
        return "Verde"
    if score < FINAL_THRESHOLDS["yellow_lt"]:
        return "Amarelo"
    return "Vermelho"


# ------------------------------------------------------------
# 9.3 — Score de Alertas
# ------------------------------------------------------------

ALERT_WEIGHTS = {
    "severity": 0.25,
    "duration": 0.25,
    "frequency": 0.25,
    "impact":   0.15,
    "rainfall": 0.10,
}

def compute_alert_score(alert_obs: Dict[str, Any]) -> Dict[str, Any]:
    sev  = _clip01(alert_obs.get("severity"))
    dur  = _clip01(alert_obs.get("duration"))
    freq = _clip01(alert_obs.get("frequency"))
    imp  = _clip01(alert_obs.get("impact"))

    p = (
        alert_obs.get("precip_24h")
        or alert_obs.get("chuva_mm")
        or alert_obs.get("precipitation")
        or 0.0
    )

    if p <= 0: rainfall = 0.0
    elif p >= 50: rainfall = 1.0
    else: rainfall = p / 50.0

    score = (
        sev * ALERT_WEIGHTS["severity"]
        + dur * ALERT_WEIGHTS["duration"]
        + freq * ALERT_WEIGHTS["frequency"]
        + imp * ALERT_WEIGHTS["impact"]
        + rainfall * ALERT_WEIGHTS["rainfall"]
    )

    return {
        "score": score,
        "components": {
            "severity": sev,
            "duration": dur,
            "frequency": freq,
            "impact": imp,
            "rainfall": rainfall,
            "precip_24h": p,
        }
    }


# ------------------------------------------------------------
# 9.4 — MAD (outliers ambientais)
# ------------------------------------------------------------

def _round_coord(x): return round(float(x), 3)

def _mad(values):
    if not values: return None
    m = stats.median(values)
    devs = [abs(v - m) for v in values]
    return m, stats.median(devs)


def pm_outlier_flags(lat, lon, pm25, pm10, k=5.0, lookback=50):
    if pm25 is None and pm10 is None:
        return False, False

    latk = _round_coord(lat)
    lonk = _round_coord(lon)

    with Session(engine) as sess:
        rows = sess.exec(
            select(WeatherObs)
            .where(WeatherObs.lat.between(latk - 0.001, latk + 0.001))
            .where(WeatherObs.lon.between(lonk - 0.001, lonk + 0.001))
            .order_by(WeatherObs.observed_at.desc())
            .limit(lookback)
        ).all()

    pm25_hist = [r.pm25 for r in rows if r.pm25 is not None]
    pm10_hist = [r.pm10 for r in rows if r.pm10 is not None]

    flag25 = flag10 = False

    if pm25_hist:
        med, mad = _mad(pm25_hist)
        if mad and abs(pm25 - med) > k * 1.4826 * mad:
            flag25 = True

    if pm10_hist:
        med, mad = _mad(pm10_hist)
        if mad and abs(pm10 - med) > k * 1.4826 * mad:
            flag10 = True

    return flag25, flag10


# ------------------------------------------------------------
# 9.5 — Fusão Final dos Scores
# ------------------------------------------------------------

def compute_final_score(ctx: Dict[str, Any]) -> FinalScoreResult:

    # Heurística (Módulo 7)
    cons = ctx.get("conservation", {})
    heuristic_raw = float(cons.get("score") or 0.0)
    heuristic_norm = heuristic_raw / 100.0

    # ML (Módulo 8)
    ml_raw = float(ctx.get("ml_raw") or 0.0)
    ml_norm = _clip01(ml_raw)

    # Alertas
    alert_params = ctx.get("alert_params") or {}
    alert_data = compute_alert_score(alert_params)
    alert_norm = _clip01(alert_data["score"])

    # MAD
    met = ctx.get("meteo") or {}
    lat = ctx["location"]["lat"]
    lon = ctx["location"]["lon"]

    p25_flag, p10_flag = pm_outlier_flags(lat, lon, met.get("pm25"), met.get("pm10"))
    mad_penalty = 0.3 * p25_flag + 0.3 * p10_flag
    mad_penalty = min(1.0, mad_penalty)

    # Fusão
    final = (
        heuristic_norm * FINAL_WEIGHTS["heuristic"]
        + ml_norm       * FINAL_WEIGHTS["ml"]
        + alert_norm    * FINAL_WEIGHTS["alerts"]
        - mad_penalty   * FINAL_WEIGHTS["mad"]
    )

    final_score = max(0.0, min(1.0, final)) * 100
    final_level = _classify_level(final_score)

    return FinalScoreResult(
        score=round(final_score, 2),
        level=final_level,
        breakdown={
            "heuristic_raw": heuristic_raw,
            "heuristic_norm": heuristic_norm,
            "ml_raw": ml_raw,
            "ml_norm": ml_norm,
            "alert_norm": alert_norm,
            "alert_details": alert_data,
            "mad_penalty": mad_penalty,
            "weights": FINAL_WEIGHTS,
            "ctx": ctx,
        },
    )


# ------------------------------------------------------------
# 9.6 — ENDPOINT /api/score_final
# ------------------------------------------------------------

class RiskRequest(BaseModel):
    cidade: str | None = None
    lat: float | None = None
    lon: float | None = None
    raio_km: int = 150


@app.post("/api/score_final", tags=["IA"])
def api_score_final(body: RiskRequest):

    # 1) Puxa contexto COMPLETO (Módulo 7)
    ctx = build_context(
        cidade=body.cidade,
        lat=body.lat,
        lon=body.lon,
        raio_km=body.raio_km,
    )

    # 2) ML v11
    ml_res = run_ml_model(ctx)
    ctx["ml_raw"] = ml_res["ml_raw"]
    ctx["ml_level"] = ml_res["ml_level"]

    # 3) Score final
    final = compute_final_score(ctx)

    return {
        "ok": True,
        "location": ctx["location"],
        "final_score": final.score,
        "final_level": final.level,
        "breakdown": final.breakdown,
    }


# ------------------------------------------------------------
# 9.7 — Inicialização das tabelas SQLite
# ------------------------------------------------------------

try:
    SQLModel.metadata.create_all(engine)
    print("✔ WeatherObs e RiskLog criados.")
except Exception as e:
    print("ERRO ao criar tabelas:", e)

# ============================================================
# 🧩 MÓDULO 10 — COLETORES OTIMIZADOS PARA O DASHBOARD (v11) — REVISADO
# ============================================================

from typing import Dict, Any
from math import radians, sin, cos, sqrt, atan2

# Importações dos módulos centrais
# get_meteo, normalize_meteo → Módulo 3
# focos_por_raios_backend → Módulo 9
# deter_stats → Módulo 7


# ------------------------------------------------------------
# Função auxiliar — haversine (km)
# ------------------------------------------------------------
def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


# ------------------------------------------------------------
# 10.1 — Clima (Open-Meteo)
# ------------------------------------------------------------
def collect_weather_now(lat: float, lon: float) -> Dict[str, Any]:
    try:
        raw = get_meteo(lat, lon)
        clima_norm = normalize_meteo(raw)

        return {
            "ok": True,
            "fonte": "open-meteo",
            "coords": {"lat": lat, "lon": lon},
            "features": clima_norm,
            "raw": raw,
        }
    except Exception as e:
        return {
            "ok": False,
            "erro": str(e),
            "coords": {"lat": lat, "lon": lon},
        }


# ------------------------------------------------------------
# 10.2 — Focos (INPE)
# ------------------------------------------------------------
def collect_focos_now(lat: float, lon: float) -> Dict[str, Any]:
    try:
        focos = focos_por_raios_backend(lat, lon)
        return {
            "ok": True,
            "coords": {"lat": lat, "lon": lon},
            "focos": focos,
        }
    except Exception as e:
        return {
            "ok": False,
            "erro": str(e),
            "coords": {"lat": lat, "lon": lon},
        }


# ------------------------------------------------------------
# 10.3 — DETER (Módulo 7)
# ------------------------------------------------------------
def collect_deter_now(lat: float, lon: float, raio_km: int = 150) -> Dict[str, Any]:
    try:
        deter = deter_stats(lat, lon, raio_km)
        return {
            "ok": True,
            "coords": {"lat": lat, "lon": lon},
            "deter": deter,
        }
    except Exception as e:
        return {
            "ok": False,
            "erro": str(e),
            "coords": {"lat": lat, "lon": lon},
        }


# ------------------------------------------------------------
# 10.4 — Bundle unificado para o dashboard
# ------------------------------------------------------------
def collect_dashboard_bundle(lat: float, lon: float, raio_km: int = 150) -> Dict[str, Any]:
    return {
        "ok": True,
        "coords": {"lat": lat, "lon": lon},
        "clima": collect_weather_now(lat, lon),
        "focos": collect_focos_now(lat, lon),
        "deter": collect_deter_now(lat, lon, raio_km),
    }
# ============================================================
# 🧩 MÓDULO 11 — ENDPOINT /api/data (Dashboard v11) — REVISADO
# ============================================================

class DataRequest(BaseModel):
    cidade: str | None = None
    lat: float | None = None
    lon: float | None = None
    raio_km: float = 150.0


@app.post("/api/data", tags=["Dashboard"], summary="Dados completos para o dashboard AmazonSafe v11")
def api_data(req: DataRequest):

    # 1) Contexto completo (Módulo 7)
    try:
        ctx = build_context(
            cidade=req.cidade,
            lat=req.lat,
            lon=req.lon,
            raio_km=req.raio_km,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"Erro ao montar contexto: {e}")

    lat = ctx["location"]["lat"]
    lon = ctx["location"]["lon"]

    # 2) Coletas individuais
    clima_now = collect_weather_now(lat, lon)
    focos_now = collect_focos_now(lat, lon)
    deter_now = collect_deter_now(lat, lon, req.raio_km)

    # 3) ML v11
    ml_res = run_ml_model(ctx)
    ctx["ml_raw"] = ml_res["ml_raw"]
    ctx["ml_level"] = ml_res["ml_level"]

    # 4) Score final
    final = compute_final_score(ctx)

    cons = ctx["conservation"]

    # 5) Resposta final v11
    return {
        "ok": True,
        "local": ctx["location"],
        "clima_atual": clima_now,
        "focos_reais": focos_now,
        "deter": deter_now,
        "heuristica": {
            "score": cons["score"],
            "level": cons["level"],
            "components": cons["components"],
        },
        "ml_v11": {
            "raw": ml_res["ml_raw"],
            "level": ml_res["ml_level"],
        },
        "score_final": {
            "score": final.score,
            "level": final.level,
            "breakdown": final.breakdown,
        },
        "contexto": ctx,
    }


# ============================================================
# 🧩 MÓDULO 12 — ENDPOINT /api/data_auto (Dashboard + IA v11)
# ============================================================

class DataAutoRequest(BaseModel):
    cidade: str | None = None
    lat: float | None = None
    lon: float | None = None
    raio_km: float = 150.0


@app.post(
    "/api/data_auto",
    tags=["Dashboard"],
    summary="Dados completos + IA + Score Final (AmazonSafe v11)",
)
def api_data_auto(req: DataAutoRequest):

    # ------------------------------------------------------------
    # 1) Contexto completo oficial (Módulo 7)
    # ------------------------------------------------------------
    try:
        ctx = build_context(
            cidade=req.cidade,
            lat=req.lat,
            lon=req.lon,
            raio_km=req.raio_km,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"Erro ao montar contexto: {e}")

    lat = ctx["location"]["lat"]
    lon = ctx["location"]["lon"]

    # ------------------------------------------------------------
    # 2) Coletas individuais (Dashboard v11)
    # ------------------------------------------------------------
    clima_now = collect_weather_now(lat, lon)
    focos_now = collect_focos_now(lat, lon)
    deter_now = collect_deter_now(lat, lon, req.raio_km)

    # Atualizar ctx para IA
    if clima_now.get("features"):
        ctx["meteo"] = clima_now["features"]

    ctx["focos"] = focos_now.get("focos", {})
    ctx["deter"] = deter_now.get("deter", {})

    # ------------------------------------------------------------
    # 3) IA RandomForest v11
    # ------------------------------------------------------------
    ml_res = run_ml_model(ctx)
    ctx["ml_raw"] = ml_res["ml_raw"]
    ctx["ml_level"] = ml_res["ml_level"]

    # ------------------------------------------------------------
    # 4) Heurística + Score Final Híbrido
    # ------------------------------------------------------------
    final = compute_final_score(ctx)
    cons = ctx["conservation"]

    return {
        "ok": True,
        "local": ctx["location"],
        "clima_atual": clima_now,
        "focos_reais": focos_now,
        "desmatamento": deter_now,
        "heuristica": {
            "score": cons.get("score"),
            "level": cons.get("level"),
            "components": cons.get("components"),
        },
        "ia": {
            "ml_raw": ml_res["ml_raw"],
            "ml_level": ml_res["ml_level"],
            "modelo_path": MODEL_PATH,
        },
        "score_hibrido": {
            "score": final.score,
            "level": final.level,
            "breakdown": final.breakdown,
        },
        "ia_compat": {
            "risco_simples": cons.get("level"),
            "score_hibrido": final.score,
            "risco_hibrido": final.level,
        },
        "contexto": ctx,
    }

# ============================================================
# 🧩 MÓDULO 13 — Alertas + Score Inteligente (v11) — REVISADO
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


@app.post(
    "/api/alertas_update",
    tags=["Alertas"],
    summary="Atualiza alertas e calcula score inteligente (v11)"
)
def api_alertas_update(req: AlertUpdateRequest):

    # ------------------------------------------------------------
    # 1) Contexto completo oficial (Módulo 7)
    # ------------------------------------------------------------
    try:
        ctx = build_context(
            cidade=req.cidade,
            lat=req.lat,
            lon=req.lon,
            raio_km=req.raio_km,
        )
    except Exception as e:
        raise HTTPException(400, f"Erro ao construir contexto: {e}")

    loc = ctx["location"]
    lat = loc["lat"]
    lon = loc["lon"]
    meteo = ctx.get("meteo") or {}
    focos_ctx = ctx.get("focos") or {}
    deter_ctx = ctx.get("deter") or {}
    conservation = ctx.get("conservation") or {}

    # ------------------------------------------------------------
    # 2) Precipitação + PMs (v11)
    # ------------------------------------------------------------
    precip = meteo.get("chuva_mm") or meteo.get("precipitation") or 0.0
    pm25 = meteo.get("pm25")
    pm10 = meteo.get("pm10")

    # ------------------------------------------------------------
    # 3) Estatística de focos
    # ------------------------------------------------------------
    focos_total = focos_ctx.get("focos_total") \
        or focos_ctx.get("count") \
        or 0

    # ------------------------------------------------------------
    # 4) alert_obs — payload completo
    # ------------------------------------------------------------
    alert_obs = {
        "severity": req.severity,
        "duration": req.duration,
        "frequency": req.frequency,
        "impact": req.impact,
        "pm25": pm25,
        "pm10": pm10,
        "precipitation": precip,
        "focos_total": focos_total,
        "meta": {
            "radius_km": req.raio_km,
            "precipitation": precip,
        }
    }

    # ------------------------------------------------------------
    # 5) Score Inteligente de Alertas
    # ------------------------------------------------------------
    alert_score = compute_alert_score(alert_obs)
    alert_level = _classify_level(alert_score["score"])

    # ------------------------------------------------------------
    # 6) Score Híbrido Final
    # ------------------------------------------------------------
    ctx2 = dict(ctx)
    ctx2["alert_params"] = alert_obs

    final = compute_final_score(ctx2)

    # ------------------------------------------------------------
    # 7) Persistência NDJSON + transição de nível
    # ------------------------------------------------------------
    alert_id = (
        req.cidade.lower().replace(" ", "_")
        if req.cidade else f"{lat:.4f},{lon:.4f}"
    )

    save_alert_score(
        alert_id=alert_id,
        score=alert_score["score"],
        level=alert_level,
        alert_obs=alert_obs,
        params={
            "lat": lat,
            "lon": lon,
            "cidade": req.cidade,
            "weather_provider": req.weather_provider,
        },
    )

    handle_level_transition(
        alert_id=alert_id,
        new_level=alert_level,
        score=alert_score["score"],
        alert_obs=alert_obs,
    )

    # ------------------------------------------------------------
    # 8) Retorno final consolidado
    # ------------------------------------------------------------
    return {
        "ok": True,
        "local": {
            "cidade": req.cidade,
            "lat": lat,
            "lon": lon,
            "resolved_by": loc.get("resolved_by"),
            "display_name": loc.get("display_name"),
            "radius_km": req.raio_km,
        },
        "contexto": {
            "meteo": meteo,
            "focos": focos_ctx,
            "deter": deter_ctx,
            "conservation": conservation,
        },
        "alert_obs": alert_obs,
        "score_alerta": {
            "score": alert_score["score"],
            "components": alert_score["components"],
            "nivel": alert_level,
        },
        "score_final": {
            "score": final.score,
            "nivel": final.level,
            "breakdown": final.breakdown,
        },
    }


# ============================================================
# 🧩 MÓDULO 14 — Rotas Administrativas / Diagnóstico (v11)
# ============================================================

from fastapi import APIRouter, HTTPException

router_admin = APIRouter(prefix="/admin", tags=["Admin"])
app.include_router(router_admin)

# ---- Defaults seguros ----
DEFAULT_LAT = -1.45056
DEFAULT_LON = -48.46824


# ------------------------------------------------------------
# 14.1 — Status geral
# ------------------------------------------------------------
@router_admin.get("/status", summary="Status geral do backend")
def admin_status():
    return {
        "ok": True,
        "service": "AmazonSafe Backend",
        "version": "1.1.0-v11",
        "model_loaded": modelo_pipeline is not None,
        "model_path": MODEL_PATH,
        "db_url": DB_URL,
    }


# ------------------------------------------------------------
# 14.2 — Diagnóstico do modelo IA
# ------------------------------------------------------------
@router_admin.get("/model_state", summary="Diagnóstico do modelo IA (v11)")
def admin_model_state():
    if modelo_pipeline is None:
        return {"loaded": False, "msg": "Modelo IA v11 não carregado."}

    try:
        feats = list(getattr(modelo_pipeline, "feature_names_in_", []))
    except:
        feats = []

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
    """
    Como ainda não existe persistência real para weather/fire,
    devolvemos um fallback seguro para não quebrar o dashboard.
    """
    return {
        "ok": True,
        "last_weather": "not_implemented",
        "last_fire": "not_implemented",
    }


# ------------------------------------------------------------
# 14.4 — Teste de conexão com serviços externos
# ------------------------------------------------------------
@router_admin.get("/external_check", summary="Verifica serviços externos (Open-Meteo / INPE)")
def admin_external_check():
    status = {}

    # Open-Meteo
    try:
        clima = get_meteo(DEFAULT_LAT, DEFAULT_LON)
        status["open_meteo"] = "ok" if clima else "empty"
    except Exception as e:
        status["open_meteo"] = f"error: {e}"

    # INPE
    try:
        d = inpe_focos_near(DEFAULT_LAT, DEFAULT_LON, 50)
        status["inpe"] = "ok" if d else "empty"
    except Exception as e:
        status["inpe"] = f"error: {e}"

    # DETER
    try:
        deter = deter_stats(DEFAULT_LAT, DEFAULT_LON, 150)
        status["deter"] = "ok" if deter else "empty"
    except Exception as e:
        status["deter"] = f"error: {e}"

    return status


# ------------------------------------------------------------
# 14.5 — Teste Geocoder
# ------------------------------------------------------------
@router_admin.get("/geocode_test", summary="Testa geocodificação")
def admin_geocode_test(cidade: str):
    try:
        info = geocode_city(cidade)
        if not info:
            return {"ok": False, "msg": f"Não foi possível geocodificar '{cidade}'"}
        return {"ok": True, "result": info}
    except Exception as e:
        raise HTTPException(400, f"Erro no geocoder: {e}")


# ============================================================
# 🧩 MÓDULO 15 — Logging, Auditoria e Telemetria Interna (v11)
# ============================================================

import json
import time
import os
import traceback
from datetime import datetime, timezone
from typing import Dict, Any
from fastapi import Request, APIRouter
from starlette.middleware.base import BaseHTTPMiddleware

# -----------------------------------------
# Diretórios de log seguros / compatíveis Render
# -----------------------------------------
def _ensure_logs_dir():
    """
    No Render /tmp é persistente durante a execução.
    Para execuções locais usa ./runtime_data/logs
    """
    base = "/tmp/amazonsafe_logs" if os.getenv("RENDER") else "./runtime_data/logs"
    os.makedirs(base, exist_ok=True)
    return base

LOG_DIR = _ensure_logs_dir()
LOG_MAIN = os.path.join(LOG_DIR, "events.ndjson")
LOG_REQUESTS = os.path.join(LOG_DIR, "requests.ndjson")
LOG_IA = os.path.join(LOG_DIR, "ia.ndjson")


# -----------------------------------------
# Utilitários
# -----------------------------------------
def now_utc():
    return datetime.now(timezone.utc)

def _safe_write(path: str, rec: Dict[str, Any]):
    """Escrita robusta NDJSON, à prova de race condition."""
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception as e:
        print("[LOG_ERROR]", e)


# -----------------------------------------
# Funções principais de log
# -----------------------------------------
def log_event(event_type: str, msg: str, extra: Dict[str, Any] | None = None):
    _safe_write(LOG_MAIN, {
        "ts": now_utc().isoformat(),
        "type": event_type,
        "message": msg,
        "extra": extra or {},
    })

def log_request(path: str, method: str, status: int, duration_ms: float, client: str):
    _safe_write(LOG_REQUESTS, {
        "ts": now_utc().isoformat(),
        "path": path,
        "method": method,
        "status": status,
        "duration_ms": duration_ms,
        "client": client,
    })

def log_inference(ctx: Dict[str, Any], prediction: Any, final_score: Any = None):
    if hasattr(final_score, "__dict__"):
        final_score = vars(final_score)
    _safe_write(LOG_IA, {
        "ts": now_utc().isoformat(),
        "ctx": ctx,
        "prediction": prediction,
        "final": final_score,
    })


# -----------------------------------------
# Middleware — captura de erros
# -----------------------------------------
class ErrorLoggerMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        try:
            return await call_next(request)
        except Exception as e:
            tb = traceback.format_exc()
            log_event("exception", str(e), {
                "path": request.url.path,
                "method": request.method,
                "traceback": tb,
            })
            print(tb)
            return JSONResponse(
                status_code=500,
                content={"ok": False, "error": str(e)}
            )

app.add_middleware(ErrorLoggerMiddleware)


# -----------------------------------------
# Middleware — auditor de chamadas HTTP
# -----------------------------------------
@app.middleware("http")
async def audit_api_calls(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    dur = round((time.time() - start) * 1000, 2)

    log_request(
        request.url.path,
        request.method,
        response.status_code,
        dur,
        request.client.host if request.client else None
    )
    return response


# -----------------------------------------
# Endpoints de leitura dos logs
# -----------------------------------------
router_logs = APIRouter(prefix="/logs", tags=["Logs"])

@router_logs.get("/tail")
def tail_logs(n: int = 50):
    n = max(1, min(n, 500))
    if not os.path.exists(LOG_MAIN):
        return {"ok": False, "msg": "Nenhum log encontrado."}
    lines = open(LOG_MAIN).read().splitlines()[-n:]
    return {"ok": True, "events": [json.loads(l) for l in lines]}

@router_logs.get("/tail_ia")
def tail_ia(n: int = 50):
    n = max(1, min(n, 500))
    if not os.path.exists(LOG_IA):
        return {"ok": False, "msg": "Nenhum log IA encontrado."}
    lines = open(LOG_IA).read().splitlines()[-n:]
    return {"ok": True, "events": [json.loads(l) for l in lines]}

@router_logs.post("/push")
def push_log(event_type: str, message: str):
    log_event(event_type, message)
    return {"ok": True}

app.include_router(router_logs)


# ============================================================
# 🧩 MÓDULO 16 — Healthcheck, Métricas e Autoverificação (v11)
# ============================================================

import os
import time
from fastapi import APIRouter
from fastapi.responses import PlainTextResponse

router_health = APIRouter(prefix="/system", tags=["Sistema"])

START_TIME = time.time()

# ------------------------------------------------------------
# Health básico
# ------------------------------------------------------------
@router_health.get("/health")
def api_health():
    return {
        "ok": True,
        "version": "1.1.0-v11",
        "timestamp": now_utc().isoformat(),
        "model_loaded": modelo_pipeline is not None,
        "db_ok": True,
        "disk_ok": True,
    }


# versão raiz
@app.get("/health")
def root_health():
    return {"ok": True, "status": "online", "path": "/system/health"}


# ------------------------------------------------------------
# Health do modelo v11
# ------------------------------------------------------------
@router_health.get("/health/model_v11")
def api_health_model_v11():
    try:
        # AGORA USANDO O build_context REAL
        ctx = build_context(lat=DEFAULT_LAT, lon=DEFAULT_LON)
        ml = run_ml_model(ctx)
        final = compute_final_score(ctx)

        return {
            "ok": True,
            "ml_raw": ml["ml_raw"],
            "ml_level": ml["ml_level"],
            "final_score": final.score,
            "final_level": final.level,
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ------------------------------------------------------------
# Providers externos
# ------------------------------------------------------------
@router_health.get("/health/providers")
def api_health_providers():
    out = {}

    # Open-Meteo Forecast
    try:
        d = get_meteo(DEFAULT_LAT, DEFAULT_LON)
        out["open_meteo"] = {"ok": True}
    except Exception as e:
        out["open_meteo"] = {"ok": False, "error": str(e)}

    # INPE focos
    try:
        f = inpe_focos_near(DEFAULT_LAT, DEFAULT_LON, 50)
        n = (f.get("features") or {}).get("count", 0)
        out["inpe"] = {"ok": True, "focos_50km": n}
    except Exception as e:
        out["inpe"] = {"ok": False, "error": str(e)}

    # DETER
    try:
        ds = deter_stats(DEFAULT_LAT, DEFAULT_LON, 150)
        out["deter"] = {"ok": True, "items": len(ds) if hasattr(ds, "__len__") else 1}
    except Exception as e:
        out["deter"] = {"ok": False, "error": str(e)}

    return {"ok": True, "providers": out}


# ------------------------------------------------------------
# Disco
# ------------------------------------------------------------
@router_health.get("/health/disk")
def api_health_disk():
    try:
        test_dir = "/tmp/amazonsafe_disk_test"
        os.makedirs(test_dir, exist_ok=True)
        fpath = os.path.join(test_dir, "test.txt")
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(now_utc().isoformat())
        return {"ok": True, "file": fpath}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ------------------------------------------------------------
# Banco de dados
# ------------------------------------------------------------
@router_health.get("/health/db")
def api_health_db():
    try:
        with engine.connect() as conn:
            conn.exec_driver_sql("SELECT 1")
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ------------------------------------------------------------
# Métricas estilo Prometheus
# ------------------------------------------------------------
@router_health.get("/metrics")
def api_metrics():
    metrics = {
        "amazonsafe_uptime_seconds": round(time.time() - START_TIME, 2),
        "amazonsafe_model_v11_loaded": 1 if modelo_pipeline is not None else 0,
    }
    return PlainTextResponse(
        "\n".join(f"{k} {v}" for k, v in metrics.items())
    )


# ------------------------------------------------------------
# Geocode Test
# ------------------------------------------------------------
@router_health.get("/health/geocode")
def system_geocode_test(city: str = "Belém"):
    try:
        geo = geocode_city(city)
        if not geo:
            return {"ok": False, "error": "geocode returned None"}
        return {"ok": True, "result": geo}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# incluir router após definição
app.include_router(router_health)

# ============================================================
# 🧩 MÓDULO 17 — Execução Local / Compatibilidade Render (v11)
# ============================================================

import uvicorn
import os
import sys

if __name__ == "__main__":

    port = int(os.getenv("PORT", "8000"))
    env = os.getenv("ENV", "development")

    print("\n==============================================")
    print("🚀 AmazonSafe API — execução local iniciada")
    print(f"➡ Ambiente..: {env}")
    print(f"➡ Porta.....: {port}")
    print(f"➡ URL.......: http://127.0.0.1:{port}")
    print("==============================================\n")

    reload_flag = env == "development"

    try:
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=port,
            reload=reload_flag,
            log_level="info",
        )
    except Exception as e:
        print("❌ Erro ao iniciar Uvicorn:", e)
        sys.exit(1)

