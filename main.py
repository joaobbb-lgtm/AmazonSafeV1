
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
# Banco de Dados — Mantido por compatibilidade
# ============================================================

DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
DB_URL = DATABASE_URL if DATABASE_URL else "sqlite:///./amazonsafe.db"
engine = create_engine(DB_URL, pool_pre_ping=True)

# ============================================================
# Inicialização ÚNICA das tabelas
# ============================================================

def init_db():
    print("🔧 Criando tabelas SQLite...")
    SQLModel.metadata.create_all(engine)
    print("✔ Tabelas prontas.")

init_db()

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
# 3.1 — GEOCODING (Nominatim + Open-Meteo + Fallback Belém)
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

GEOCODE_UA = "AmazonSafe/3 (contato: joaobbb@gmail.com; https://amazonsafe-api.onrender.com)"
GEO_TIMEOUT = 8  # timeouts mais curtos para não travar o backend


def _split_city_state(q: str):
    s = q.strip()
    m = re.split(r"\s*[,;-]\s*|\s{2,}", s, maxsplit=1)
    return (m[0].strip(), m[1].strip()) if len(m) == 2 else (s, None)


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


def _geocode_nominatim(q: str, state_name: str | None):
    """
    Tenta geocodificar via Nominatim (OpenStreetMap).
    Retorna dict com lat, lon, display_name ou None.
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
    Geocodificador robusto com 3 estágios:
      1) Nominatim (OpenStreetMap)
      2) Open-Meteo Geocoding
      3) Fallback: Belém (DEFAULT_LAT/DEFAULT_LON)

    Nunca levanta erro para o chamador — sempre retorna dict.
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

    # 2) Open-Meteo Geocoding
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

    # 3) Fallback seguro: Belém/PA
    try:
        lat = float(DEFAULT_LAT)
        lon = float(DEFAULT_LON)
    except Exception:
        lat, lon = -1.45056, -48.4682453  # hardcoded caso env quebre

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
    except Exception:
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
    Normaliza as colunas principais do CSV do INPE.
    Tenta vários nomes possíveis para latitude/longitude, data, etc.
    """
    cols = {c.lower(): c for c in df.columns}

    def pick(*names):
        for n in names:
            if n in cols:
                return cols[n]
        return None

    c_lat = pick("latitude","lat","y")
    c_lon = pick("longitude","lon","x")
    c_dt  = pick("datahora","data_hora","data_hora_gmt")
    c_sat = pick("satelite","satellite")
    c_frp = pick("frp","radiative_power")
    c_uf  = pick("uf","estado","state")
    c_mun = pick("municipio","município","city","nome_munic","municipality")

    out = pd.DataFrame()

    out["latitude"] = df[c_lat] if c_lat else None
    out["longitude"] = df[c_lon] if c_lon else None
    out["datahora"] = df[c_dt] if c_dt else None
    out["satelite"] = df[c_sat] if c_sat else None
    out["frp"] = df[c_frp] if c_frp else None

    # campos opcionais
    if c_uf:
        out["uf"] = df[c_uf]
    else:
        out["uf"] = None

    if c_mun:
        out["municipio"] = df[c_mun]
    else:
        out["municipio"] = None

    return out

def _json_safe(v):
    try:
        if pd.isna(v):
            return None
    except Exception:
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
    """
    Retorna focos do INPE próximos ao ponto (lat, lon),
    dentro de um raio em km.
    """

    payload = inpe_fetch_csv(scope=scope, region=region, timeout=timeout)
    df = payload["df"]
    norm = _canonical_inpe_columns(df)

    # Converte para float seguro
    norm["latitude"] = norm["latitude"].map(parse_float_safe)
    norm["longitude"] = norm["longitude"].map(parse_float_safe)

    # Filtra por bounding box aproximado
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

    # Distância exata em km (se a função existir)
    try:
        sub["dist_km"] = sub.apply(
            lambda r: haversine_km(lat, lon, r["latitude"], r["longitude"]),
            axis=1
        )
        sub = sub[sub["dist_km"] <= float(raio_km)]
    except Exception:
        # Se der erro em haversine_km, segue sem a filtragem fina
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
            "uf": _json_safe(r.get("uf")) if "uf" in r else None,
            "municipio": _json_safe(r.get("municipio")) if "municipio" in r else None,
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
# 3.6 — DETER PARQUET LOCAL (Cloudflare R2)
# ============================================================

import zipfile

DETER_R2_URL = os.getenv(
    "DETER_R2_URL",
    "https://0be47677b22a6dd946b4ff62d6dce778.r2.cloudflarestorage.com/deter-storage/deter_parquet.zip"
)

DETER_LOCAL_ZIP = "/tmp/deter_parquet.zip"
DETER_LOCAL_PARQUET = "/tmp/deter.parquet"

@ttl_cache(ttl_seconds=3600)   # revalida a cada 1h
def load_deter_parquet():
    """
    Baixa o arquivo deter_parquet.zip do Cloudflare R2,
    extrai o .parquet para /tmp e devolve um DataFrame.
    """

    # 1) baixar zip
    try:
        r = http_get(DETER_R2_URL, timeout=HTTP_TIMEOUT)
        with open(DETER_LOCAL_ZIP, "wb") as f:
            f.write(r.content)
    except Exception as e:
        print("[DETER] Erro ao baixar:", e)
        return None

    # 2) extrair parquet
    try:
        with zipfile.ZipFile(DETER_LOCAL_ZIP, "r") as z:
            fname = z.namelist()[0]
            z.extract(fname, "/tmp/")
            os.rename(f"/tmp/{fname}", DETER_LOCAL_PARQUET)
    except Exception as e:
        print("[DETER] Erro ao extrair:", e)
        return None

    # 3) carregar parquet
    try:
        df = pd.read_parquet(DETER_LOCAL_PARQUET)
        return df
    except Exception as e:
        print("[DETER] Erro ao ler parquet:", e)
        return None

# ============================================================
# 3.7 — DESMATAMENTO POR RAIO USANDO DETER PARQUET
# ============================================================

def get_desmatamento(lat: float, lon: float, raio_km: float = 50):
    df = load_deter_parquet()

    if df is None or df.empty:
        return {"count": 0, "area_total": 0}

    # campos esperados:
    # latitude, longitude, area_ha, dataalerta

    df = df.copy()
    df["lat"] = df["latitude"].astype(float)
    df["lon"] = df["longitude"].astype(float)

    # bounding box rápido
    minx, miny, maxx, maxy = bbox_from_center(lat, lon, raio_km)
    sub = df[
        (df["lon"] >= minx) & (df["lon"] <= maxx) &
        (df["lat"] >= miny) & (df["lat"] <= maxy)
    ].copy()

    if sub.empty:
        return {"count": 0, "area_total": 0}

    # filtro exato pelo haversine
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
# MÓDULO 4 — NORMALIZADORES E SANITIZAÇÃO DE DADOS
# ============================================================

def normalize_value(x):
    """Converte para float seguro e descarta valores impossíveis."""
    try:
        if x is None:
            return None
        v = float(x)

        # Remover absurdos
        if math.isinf(v) or math.isnan(v):
            return None

        return v
    except:
        return None


def normalize_meteo(data: dict) -> dict:
    """
    Normaliza o dicionário retornado por Open-Meteo:
    - converte tudo para float
    - corrige limites impossíveis
    - mantém apenas variáveis usadas pelo ML e pelo ConservationScore
    """
    if not isinstance(data, dict):
        return {}

    out = {}

    for k, v in data.items():
        out[k] = normalize_value(v)

    # Segurança extra — remover lixo inesperado
    allowed_keys = {
        "temperature_2m", "relativehumidity_2m", "dewpoint_2m",
        "surface_pressure", "windspeed_10m", "winddirection_10m",
        "windgusts_10m", "precipitation",
        "shortwave_radiation", "direct_normal_irradiance",
        "soil_temperature_0cm", "soil_moisture_0_to_1cm",
        "evapotranspiration",
        "pm10", "pm25", "o3", "no2", "so2", "co", "uv"
    }

    # Mantém apenas variáveis conhecidas
    out = {k: out.get(k) for k in allowed_keys}

    return out


def normalize_focos_result(focos_data: dict) -> dict:
    """
    Normaliza o resultado da busca INPE focos_near().
    """
    if not isinstance(focos_data, dict):
        return {"count": 0}

    try:
        return {
            "count": int(focos_data.get("count", 0)),
        }
    except:
        return {"count": 0}


def normalize_desmatamento(data: dict) -> dict:
    """
    Normaliza o retorno do DETER via parquet.
    """
    if not isinstance(data, dict):
        return {"count": 0, "area_total": 0.0}

    try:
        return {
            "count": int(data.get("count", 0)),
            "area_total": float(data.get("area_total", 0.0))
        }
    except:
        return {"count": 0, "area_total": 0.0}




# ============================================================
# MÓDULO 5 — Inicialização da API FastAPI
# ============================================================

app = FastAPI(
    title="AmazonSafe API",
    version="2.0",
    description="API para risco ambiental, previsão de incêndios e monitoramento de conservação"
)

# CORS — liberar geral (MVP)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Configuração do AmazonSafe API carregada com sucesso.")


# ------------------------------------------------------------
# HEALTHCHECK / ROOT
# ------------------------------------------------------------

@app.get("/", summary="Health Check", tags=["Infra"])
def root():
    return {
        "ok": True,
        "message": "AmazonSafe API funcionando",
        "version": "2.0"
    }


# ============================================================
# MÓDULO 6 — Persistência & Helpers Gerais (v11)
# ============================================================

import os
import json
import time
import math
import datetime as dt
from typing import Any, Dict, Optional

from sqlmodel import create_engine, Session

# ------------------------------------------------------------
# 6.1 — Banco de Dados: Engine único
# ------------------------------------------------------------

DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
DB_URL = DATABASE_URL if DATABASE_URL else "sqlite:///./amazonsafe.db"

engine = create_engine(DB_URL, pool_pre_ping=True)

print(f"[DB] Engine inicializado: {DB_URL}")


# ------------------------------------------------------------
# 6.2 — Helpers de Tempo
# ------------------------------------------------------------

UTC = dt.timezone.utc

def now_utc() -> dt.datetime:
    return dt.datetime.now(UTC)


def now_unix() -> int:
    return int(time.time())


# ------------------------------------------------------------
# 6.3 — Helpers de Persistência Genérica (NDJSON)
# ------------------------------------------------------------

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path


def append_ndjson(path: str, record: dict):
    """Salva uma linha NDJSON em um arquivo."""
    ensure_dir(os.path.dirname(path))
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ------------------------------------------------------------
# 6.4 — Sistema de Alertas (eventos e notificações)
# ------------------------------------------------------------

ALERTS_DIR = "./runtime_data/alerts"
NOTIFY_DEBOUNCE_SEC = int(os.getenv("NOTIFY_DEBOUNCE_SEC", "600"))
WEBHOOK_URL = os.getenv("ALERTS_WEBHOOK_URL")

_LAST_LEVEL: Dict[str, Dict[str, Any]] = {}
_LAST_NOTIFY: Dict[str, float] = {}


def save_alert_score(alert_id: str, score: float, level: str,
                      alert_obs: Dict[str, Any],
                      params: Optional[Dict[str, Any]] = None):
    """Armazena uma linha em alerts.ndjson."""
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
    """Salva mudança de nível em level_events.ndjson."""
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
    """Envia webhook ou salva no arquivo local."""
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
        "obs": {
            k: obs.get(k)
            for k in ("severity", "duration", "frequency", "impact")
        },
    }

    if WEBHOOK_URL:
        try:
            import requests
            requests.post(WEBHOOK_URL, json=msg, timeout=8)
        except Exception as e:
            print("[notify error]", e)
    else:
        append_ndjson(f"{ALERTS_DIR}/notifications.ndjson", msg)


def handle_alert_transition(alert_id: str, new_level: str, score: float,
                            alert_obs: Dict[str, Any],
                            extra: Optional[Dict[str, Any]] = None,
                            notify_on_bootstrap: bool = False):
    """Controla transições de níveis com debounce e persistência."""

    old_level = (_LAST_LEVEL.get(alert_id) or {}).get("level")
    first_time = old_level is None

    if new_level != old_level:
        persist_level_change(alert_id, old_level, new_level, {
            "score": score,
            "obs": alert_obs,
            **(extra or {}),
        })

        if (not first_time) or notify_on_bootstrap:
            notify_level_change(alert_id, old_level, new_level, score, alert_obs)

        _LAST_LEVEL[alert_id] = {"level": new_level, "ts": now_utc()}


# ------------------------------------------------------------
# 6.5 — Helpers Genéricos
# ------------------------------------------------------------

def safe_float(x, default: float = 0.0) -> float:
    try:
        f = float(x)
        return f if math.isfinite(f) else default
    except Exception:
        return default


def coalesce(*vals, default=None):
    for v in vals:
        if v is not None:
            return v
    return default


def is_valid_pm25(x) -> bool:
    """Usado no ML híbrido."""
    try:
        xf = float(x)
        return math.isfinite(xf) and xf >= 1.0
    except:
        return False

# ============================================================
# MÓDULO 7 — Sistema de Risco AmazonSafe (v11)
# ============================================================

import os
import math
import zipfile
from io import BytesIO
from typing import Any, Dict, Optional

import pandas as pd
import requests
from fastapi import HTTPException

# Importa helpers essenciais (sem SQLModel)
from main import (
    safe_float,
    coalesce,
    now_utc,
)

# Importa funções de meteo do módulo responsável
from main import get_meteo, normalize_meteo

# Importa helpers geográficos existentes no projeto
from main import (
    bbox_from_center,
    haversine_km,
    parse_float_safe,
    ttl_cache,
)

# Defaults gerais
DEFAULT_LAT = -1.45056
DEFAULT_LON = -48.4682

HTTP_TIMEOUT = 15

# ------------------------------------------------------------
# 7.1 — Resolver localização
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
    - Se cidade → geocodifica
    - Se lat/lon enviados → usa direto
    - Caso contrário → fallback (Belém)
    """
    lat = _float_or_none(lat)
    lon = _float_or_none(lon)

    if cidade:
        info = geocode_city(cidade)
        if not info:
            raise HTTPException(404, f"Não foi possível geocodificar '{cidade}'")
        return float(info["lat"]), float(info["lon"]), {
            "resolved_by": "geocode",
            "display_name": info.get("display_name")
        }

    if lat is not None and lon is not None:
        return lat, lon, {"resolved_by": "direct_params"}

    return DEFAULT_LAT, DEFAULT_LON, {"resolved_by": "default"}


# ------------------------------------------------------------
# 7.2 — Carregamento do DETER unificado (.parquet ou ZIP)
# ------------------------------------------------------------

DETER_PARQUET_PATH = os.getenv("DETER_PARQUET_PATH", "")
DETER_PARQUET_URL  = os.getenv("DETER_PARQUET_URL", "")
DETER_CACHE_TTL     = int(os.getenv("DETER_CACHE_TTL_SEC", "3600"))


def _read_parquet_zip(zf: zipfile.ZipFile) -> pd.DataFrame:
    for name in zf.namelist():
        if name.lower().endswith(".parquet"):
            with zf.open(name) as f:
                return pd.read_parquet(f)
    raise RuntimeError("ZIP não contém arquivo .parquet")


@ttl_cache(ttl_seconds=DETER_CACHE_TTL)
def load_deter() -> Optional[pd.DataFrame]:
    """Carrega o parquet local, ou baixa da URL configurada."""
    # Fonte local
    if DETER_PARQUET_PATH and os.path.exists(DETER_PARQUET_PATH):
        if DETER_PARQUET_PATH.lower().endswith(".zip"):
            with zipfile.ZipFile(DETER_PARQUET_PATH, "r") as zf:
                return _read_parquet_zip(zf)
        return pd.read_parquet(DETER_PARQUET_PATH)

    # URL remota
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
    """Normaliza latitude/longitude/área/classes."""
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

    # filtro fino
    try:
        sub["dist"] = sub.apply(lambda r: haversine_km(lat, lon, r["latitude"], r["longitude"]), axis=1)
        sub = sub[sub["dist"] <= raio_km]
    except:
        pass

    n = len(sub)
    if "area_ha" in sub.columns:
        area_total = float(sub["area_ha"].dropna().sum())
    else:
        area_total = float(n)

    score_raw = math.log1p(area_total) + 0.5 * math.log1p(n)
    score_norm = max(0.0, min(1.0, score_raw / 4.0))

    return {
        "count": int(n),
        "total_area_ha": float(area_total),
        "score_raw": score_raw,
        "score_norm": score_norm,
    }


# ------------------------------------------------------------
# 7.3 — Focos INPE
# ------------------------------------------------------------

INPE_DEFAULT_SCOPE  = "diario"
INPE_DEFAULT_REGION = "Brasil"

from main import inpe_focos_near  # mesma função já existente


def focos_stats(lat: float, lon: float, raio_km: float = 150.0,
                scope: str = INPE_DEFAULT_SCOPE,
                region: str = INPE_DEFAULT_REGION) -> Dict[str, Any]:

    data = inpe_focos_near(lat, lon, raio_km, scope, region)
    feats = data.get("features", {})
    focos = feats.get("focos", [])
    meta  = feats.get("meta", {})

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
# 7.4 — Score Heurístico de Conservação
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
# 7.5 — Construção do CONTEXTO (para módulos 8 e 9)
# ------------------------------------------------------------

def build_context(cidade=None, lat=None, lon=None, raio_km=150.0) -> Dict[str, Any]:

    lat_r, lon_r, loc_meta = resolve_location(cidade, lat, lon)

    met_raw = get_meteo(lat_r, lon_r) or {}
    met = normalize_meteo(met_raw)

    det = deter_stats(lat_r, lon_r, raio_km)
    foc = focos_stats(lat_r, lon_r, raio_km)
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
# NOVO MÓDULO 8 — SQLModel + IA AmazonSafe v11
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
# 8.0 — BANCO (opcional — apenas logs e histórico)
# ------------------------------------------------------------

from sqlmodel import SQLModel, Field, Session, create_engine

DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
DB_URL = DATABASE_URL if DATABASE_URL else "sqlite:///./amazonsafe.db"

engine = create_engine(DB_URL, pool_pre_ping=True)


class RiskLog(SQLModel, table=True):
    """Registro histórico de previsões do modelo ML."""
    id: Optional[int] = Field(default=None, primary_key=True)
    latitude: float
    longitude: float
    score: float
    level: str
    created_at: dt.datetime = Field(default_factory=dt.datetime.utcnow)
    payload_json: Optional[str] = None


def init_db_ai():
    SQLModel.metadata.create_all(engine)
    print("✔ [M8] Tabelas de IA criadas (RiskLog).")


init_db_ai()


# ------------------------------------------------------------
# 8.1 — CARREGAR MODELO V11
# ------------------------------------------------------------

MODEL_PATH = "models/amazonsafe_pipeline_v11.joblib"

try:
    modelo_pipeline = joblib.load(MODEL_PATH)
    print(f"[IA] Modelo AmazonSafe v11 carregado de: {MODEL_PATH}")
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
# 8.3 — Função auxiliar: montar vetor X
# ------------------------------------------------------------

def _build_feature_vector(payload: Dict[str, Any]) -> np.ndarray:
    """
    Gera vetor ordenado nas FEATURES do modelo.
    Valores inexistentes -> 0.
    """
    row = []
    for col in MODEL_FEATURES:
        v = payload.get(col)
        try:
            row.append(float(v) if v is not None else 0.0)
        except:
            row.append(0.0)

    return np.array([row], dtype=float)


# ------------------------------------------------------------
# 8.4 — Função de previsão (usada pelo Módulo 9)
# ------------------------------------------------------------

def run_ml_model(ctx: dict) -> dict:
    """
    Recebe contexto do Módulo 7:
      { location{}, meteo{}, ... }

    Retorna:
      { ml_raw: float, ml_level: str }
    """

    if modelo_pipeline is None:
        return {"ml_raw": 0.0, "ml_level": "desconhecido"}

    loc = ctx.get("location", {})
    met = ctx.get("meteo", {})

    # Monta payload unificado
    payload = met.copy()
    payload["latitude"] = loc.get("lat")
    payload["longitude"] = loc.get("lon")

    X = _build_feature_vector(payload)

    try:
        pred = float(modelo_pipeline.predict(X)[0])
    except Exception as e:
        print("[run_ml_model] ERRO:", e)
        return {"ml_raw": 0.0, "ml_level": "erro"}

    ml_raw = max(0.0, min(1.0, pred))

    if ml_raw < 0.33:
        level = "Baixo"
    elif ml_raw < 0.66:
        level = "Médio"
    else:
        level = "Alto"

    return {
        "ml_raw": ml_raw,
        "ml_level": level,
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

    # -> salva log opcional
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


# Nota: o router precisa ser incluído no main:
# app.include_router(router_risk)


# ============================================================
# 🧩 MÓDULO 9 — IA Leve + Scoring Inteligente (v11) — OTIMIZADO
# ============================================================

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime, timezone
import math
import statistics as stats

from fastapi import HTTPException
from sqlmodel import Session, select

# Importações diretas (evita import circular do main)
from main import engine
from main import WeatherObs
from main import build_observation_context
from main import run_ml_model


# ------------------------------------------------------------
# 9.0 — Limiares / Pesos do Score Final
# ------------------------------------------------------------

FINAL_THRESHOLDS = {
    "green_lt": 33,   # score < 33  → Verde
    "yellow_lt": 66,  # score < 66  → Amarelo
}

FINAL_WEIGHTS = {
    "heuristic": 0.40,   # ConservationScore (Módulo 7)
    "ml":        0.40,   # RandomForest v11 (Módulo 8)
    "alerts":    0.15,   # severity/duration/frequency/impact
    "mad":       0.05,   # penalidades por outliers PM
}


# ------------------------------------------------------------
# 9.1 — Estrutura final de retorno
# ------------------------------------------------------------

@dataclass
class FinalScoreResult:
    score: float
    level: str
    breakdown: Dict[str, Any]


# ============================================================
# 9.2 — Funções utilitárias otimizadas
# ============================================================

def _clip01(x: Any) -> float:
    """Garante que x ∈ [0,1]."""
    try:
        return max(0.0, min(1.0, float(x)))
    except:
        return 0.0


def _classify_level(score: float) -> str:
    """Retorna Verde / Amarelo / Vermelho."""
    if score < FINAL_THRESHOLDS["green_lt"]:
        return "Verde"
    if score < FINAL_THRESHOLDS["yellow_lt"]:
        return "Amarelo"
    return "Vermelho"


def _rainfall_index(mm: Optional[float]) -> float:
    """Normaliza chuva em 0–1 usando escala fixa (50 mm → 1.0)."""
    try:
        v = float(mm or 0.0)
    except:
        return 0.0

    if v <= 0:
        return 0.0
    if v >= 50:
        return 1.0
    return v / 50.0


# ============================================================
# 9.3 — Score de Alerta (severity/duration/freq/impact + chuva)
# ============================================================

ALERT_WEIGHTS = {
    "severity": 0.25,
    "duration": 0.25,
    "frequency": 0.25,
    "impact":   0.15,
    "rainfall": 0.10,
}

def compute_alert_score(alert_obs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calcula score leve baseado em:
      - severidade
      - duração
      - frequência
      - impacto
      - chuva nas últimas 24h
    """
    sev  = _clip01(alert_obs.get("severity"))
    dur  = _clip01(alert_obs.get("duration"))
    freq = _clip01(alert_obs.get("frequency"))
    imp  = _clip01(alert_obs.get("impact"))

    precip_24h = (
        alert_obs.get("precip_24h")
        or alert_obs.get("chuva_mm")
        or alert_obs.get("precipitation")
        or (alert_obs.get("meta", {}).get("precipitation"))
        or 0.0
    )

    rainfall = _rainfall_index(precip_24h)

    score = (
        sev  * ALERT_WEIGHTS["severity"]
        + dur  * ALERT_WEIGHTS["duration"]
        + freq * ALERT_WEIGHTS["frequency"]
        + imp  * ALERT_WEIGHTS["impact"]
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
            "precip_24h_mm": float(precip_24h),
        }
    }


# ============================================================
# 9.4 — MAD (Detecção de Outliers Ambientais)
# ============================================================

def _round_coord(v: float, ndigits: int = 3):
    """Arredonda coordenadas para indexação local."""
    return round(float(v), ndigits)


def _mad(values: List[float]):
    """Retorna mediana e MAD (Median Absolute Deviation)."""
    if not values:
        return None
    med = stats.median(values)
    devs = [abs(x - med) for x in values]
    mad = stats.median(devs)
    return med, mad


def pm_outlier_flags(lat: float, lon: float, pm25, pm10, k=5.0, lookback=50):
    """
    Verifica se PM2.5/PM10 são outliers em relação ao histórico local.
    Usa WeatherObs do módulo 6.
    """
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

    # Outlier PM2.5
    if pm25 is not None and len(pm25_hist) >= 10:
        mad_res = _mad(pm25_hist)
        if mad_res:
            med, mad = mad_res
            if mad and abs(pm25 - med) > k * 1.4826 * mad:
                flag25 = True

    # Outlier PM10
    if pm10 is not None and len(pm10_hist) >= 10:
        mad_res = _mad(pm10_hist)
        if mad_res:
            med, mad = mad_res
            if mad and abs(pm10 - med) > k * 1.4826 * mad:
                flag10 = True

    return flag25, flag10


# ============================================================
# 9.5 — Fusão Final de Scores (Heurística + IA + Alertas + MAD)
# ============================================================

def _extract_heuristic_from_ctx(ctx: Dict[str, Any]) -> Tuple[float, Optional[str]]:
    """
    Extrai o heuristic_score do Módulo 7.
    Prioridade:
      1) ctx["conservation"]["score"]
      2) ctx["heuristic_score"] (compatibilidade)
    """
    cons = ctx.get("conservation") or {}
    score = cons.get("score")
    level = cons.get("level")

    if score is None:
        score = ctx.get("heuristic_score")
    if level is None:
        level = ctx.get("heuristic_level")

    try:
        return float(score or 0.0), level
    except:
        return 0.0, level


def compute_final_score(ctx: Dict[str, Any]) -> FinalScoreResult:
    """
    Combinação final:
        - heurística → peso 0.40
        - ML v11    → peso 0.40
        - alertas   → peso 0.15
        - MAD       → peso 0.05
    """

    # --- Heurística ---
    heuristic_raw, heuristic_level = _extract_heuristic_from_ctx(ctx)
    heuristic_norm = heuristic_raw / 100.0

    # --- ML ---
    ml_raw = float(ctx.get("ml_raw", 0.0))
    ml_norm = _clip01(ml_raw)

    # --- Alertas ---
    alert_params = ctx.get("alert_params") or {}
    alert_data = compute_alert_score(alert_params) if alert_params else {
        "score": 0.0,
        "components": {}
    }
    alert_norm = _clip01(alert_data["score"])

    # --- Outliers MAD ---
    met = ctx.get("meteo") or {}
    loc = ctx.get("location") or {}

    lat = loc.get("lat")
    lon = loc.get("lon")

    mad_penalty = 0.0
    if lat and lon:
        p25_flag, p10_flag = pm_outlier_flags(lat, lon, met.get("pm25"), met.get("pm10"))
        if p25_flag:
            mad_penalty += 0.3
        if p10_flag:
            mad_penalty += 0.3
        mad_penalty = min(1.0, mad_penalty)

    # --- Fusão final ---
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
            "heuristic_score_norm": heuristic_norm,
            "heuristic_raw": heuristic_raw,
            "heuristic_level": heuristic_level,
            "ml_norm": ml_norm,
            "ml_raw": ml_raw,
            "alert_norm": alert_norm,
            "alert_details": alert_data,
            "mad_penalty": mad_penalty,
            "weights": FINAL_WEIGHTS,
            "ctx": ctx,
        },
    )


# ============================================================
# 9.6 — ENDPOINT OFICIAL /api/score_final
# ============================================================

class RiskRequest(BaseModel):
    cidade: str | None = None
    lat: float | None = None
    lon: float | None = None
    raio_km: int = 150   # padrão


@app.post("/api/score_final", tags=["IA"], summary="Score híbrido (ML + heurística + MAD + alertas)")
def api_score_final(body: RiskRequest):

    # 1) Monta contexto completo (Módulo 7)
    ctx = build_observation_context(
        cidade=body.cidade,
        lat=body.lat,
        lon=body.lon,
        raio_km=body.raio_km,
    )

    # 2) ML v11 (Módulo 8)
    ml_res = run_ml_model(ctx)
    ctx["ml_raw"] = ml_res.get("ml_raw")
    ctx["ml_level"] = ml_res.get("ml_level")

    # 3) Score final (este módulo)
    final = compute_final_score(ctx)

    return {
        "ok": True,
        "location": ctx["location"],
        "final_score": final.score,
        "final_level": final.level,
        "breakdown": final.breakdown,
    }


# ============================================================
# 🧩 MÓDULO 10 — COLETORES OTIMIZADOS PARA O DASHBOARD (v11)
# ============================================================

from typing import Dict, Any

# Importa utilidades do Módulo 7 (fonte oficial)
from main import (
    get_meteo,
    normalize_meteo,
    focos_por_raios_backend,
    get_deter_data,
)


# ------------------------------------------------------------
# 10.1 — Coletor de clima atual (Open-Meteo + Normalização)
# ------------------------------------------------------------

def collect_weather_now(lat: float, lon: float) -> Dict[str, Any]:
    """Coleta clima atual e padroniza para o dashboard v11."""
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
            "features": {},
        }


# ------------------------------------------------------------
# 10.2 — Coletor de focos
# ------------------------------------------------------------

def collect_focos_now(lat: float, lon: float) -> Dict[str, Any]:
    """Coleta número de focos reais em 50/150/300 km."""
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
# 10.3 — Coletor de DETER (últimas detecções)
# ------------------------------------------------------------

def collect_deter_now(lat: float, lon: float, raio_km: int = 150) -> Dict[str, Any]:
    """
    Coleta últimas detecções DETER (backend módulo 7).
    Usa o alvo espacial coerente com focos e IA.
    """
    try:
        deter = get_deter_data(lat, lon, raio_km)
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
# 10.4 — Bundle unificado para dashboards (v11)
# ------------------------------------------------------------

def collect_dashboard_bundle(lat: float, lon: float, raio_km: int = 150) -> Dict[str, Any]:
    """
    Pacote unificado (clima + ar + solo + focos + DETER) para dashboards.
    Tudo alinhado com Módulo 7 e modelos v11.
    """
    clima = collect_weather_now(lat, lon)
    focos = collect_focos_now(lat, lon)
    deter = collect_deter_now(lat, lon, raio_km)

    return {
        "ok": True,
        "coords": {"lat": lat, "lon": lon},
        "clima": clima,
        "focos": focos,
        "deter": deter,
    }


# ============================================================
# 🧩 MÓDULO 11 — ENDPOINT /api/data (Dashboard v11)
# ============================================================

class DataRequest(BaseModel):
    cidade: str | None = None
    lat: float | None = None
    lon: float | None = None
    raio_km: float = 150.0


@app.post("/api/data", tags=["Dashboard"], summary="Dados completos para o dashboard AmazonSafe v11")
def api_data(req: DataRequest):

    # ------------------------------------------------------------
    # 1) Contexto completo (v11)
    # ------------------------------------------------------------
    try:
        ctx = build_observation_context(
            cidade=req.cidade,
            lat=req.lat,
            lon=req.lon,
            raio_km=req.raio_km,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            400,
            detail=f"Erro ao obter contexto: {e}"
        )

    lat = ctx["location"]["lat"]
    lon = ctx["location"]["lon"]

    # ------------------------------------------------------------
    # 2) Clima + Focos + DETER
    # ------------------------------------------------------------
    clima_now = collect_weather_now(lat, lon)
    focos_now = collect_focos_now(lat, lon)
    deter = ctx.get("deter")

    # ------------------------------------------------------------
    # 3) IA v11 (ML)
    # ------------------------------------------------------------
    ml_res = run_ml_model(ctx)
    ctx["ml_raw"] = ml_res.get("ml_raw")
    ctx["ml_level"] = ml_res.get("ml_level")

    # ------------------------------------------------------------
    # 4) Score Final Híbrido
    # ------------------------------------------------------------
    final = compute_final_score(ctx)

    # ------------------------------------------------------------
    # 5) ConservationScore (heurístico)
    # ------------------------------------------------------------
    cons = ctx.get("conservation") or {}
    heuristic_score = cons.get("score")
    heuristic_level = cons.get("level")

    # ------------------------------------------------------------
    # 6) Resposta oficial (Dashboard v11)
    # ------------------------------------------------------------
    return {
        "ok": True,
        "local": ctx["location"],
        "clima_atual": clima_now,
        "focos_reais": focos_now,
        "deter": deter,
        "heuristica": {
            "score": heuristic_score,
            "level": heuristic_level,
            "components": cons.get("components"),
        },
        "ml_v11": {
            "raw": ml_res.get("ml_raw"),
            "level": ml_res.get("ml_level"),
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
    # 1) Contexto completo
    # ------------------------------------------------------------
    try:
        ctx = build_observation_context(
            cidade=req.cidade,
            lat=req.lat,
            lon=req.lon,
            raio_km=req.raio_km,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            400,
            detail=f"Erro ao construir contexto: {e}",
        )

    lat = ctx["location"]["lat"]
    lon = ctx["location"]["lon"]

    # ------------------------------------------------------------
    # 2) Clima + Focos + DETER
    # ------------------------------------------------------------
    clima_now = collect_weather_now(lat, lon)
    focos_now = collect_focos_now(lat, lon)
    deter = ctx.get("deter")

    # ------------------------------------------------------------
    # 3) IA RandomForest v11
    # ------------------------------------------------------------
    ml_res = run_ml_model(ctx)
    ctx["ml_raw"] = ml_res.get("ml_raw")
    ctx["ml_level"] = ml_res.get("ml_level")

    # ------------------------------------------------------------
    # 4) Heurística + Score Final Híbrido
    # ------------------------------------------------------------
    cons = ctx.get("conservation") or {}
    heuristic_score = cons.get("score")
    heuristic_level = cons.get("level")

    final = compute_final_score(ctx)

    return {
        "ok": True,
        "local": {
            "cidade_entrada": req.cidade,
            "lat": lat,
            "lon": lon,
            "resolved_by": ctx["location"].get("resolved_by"),
            "display_name": ctx["location"].get("display_name"),
            "radius_km": req.raio_km,
        },
        "clima_atual": clima_now,
        "focos_reais": focos_now,
        "desmatamento": deter,
        "heuristica": {
            "score": heuristic_score,
            "level": heuristic_level,
            "components": cons.get("components"),
        },
        "ia": {
            "ml_raw": ml_res.get("ml_raw"),
            "ml_level": ml_res.get("ml_level"),
            "modelo_path": MODEL_PATH,
        },
        "score_hibrido": {
            "score": final.score,
            "level": final.level,
            "breakdown": final.breakdown,
        },
        # Compatibilidade com o front antigo
        "ia_compat": {
            "risco_simples": heuristic_level,
            "score_hibrido": final.score,
            "risco_hibrido": final.level,
        },
        "contexto": ctx,
    }


# ============================================================
# 🧩 MÓDULO 13 — Alertas + Score Inteligente (v11)
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
    """
    Pipeline v11:
      1) build_observation_context → clima + solo + ar + DETER + focos
      2) monta alert_obs completo
      3) calcula score de alerta (compute_alert_score)
      4) adiciona alert_params ao ctx e calcula score final híbrido
      5) gera persistência NDJSON
      6) verifica transição de nível e envia notificações
      7) retorna resposta consolidada
    """

    # ----------------------------------------------------------------------
    # 1) Contexto ambiental completo
    # ----------------------------------------------------------------------
    try:
        ctx = build_observation_context(
            cidade=req.cidade,
            lat=req.lat,
            lon=req.lon,
            raio_km=req.raio_km
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"Erro ao construir contexto: {e}")

    loc = ctx["location"]
    lat = loc["lat"]
    lon = loc["lon"]

    meteo = ctx.get("meteo") or {}
    focos_ctx = ctx.get("focos") or {}
    deter_ctx = ctx.get("deter") or {}
    conservation = ctx.get("conservation") or {}

    # ----------------------------------------------------------------------
    # 2) Precipitação + PMs (v11)
    # ----------------------------------------------------------------------
    precip = (
        meteo.get("chuva_mm")
        or meteo.get("precipitation")
        or meteo.get("rain")
        or 0.0
    )

    pm25 = meteo.get("pm25")
    pm10 = meteo.get("pm10")

    # ----------------------------------------------------------------------
    # 3) Estatística simples de focos
    # ----------------------------------------------------------------------
    focos_total = (
        focos_ctx.get("count")
        or focos_ctx.get("total")
        or focos_ctx.get("focos_total")
    )

    if focos_total is None:
        lista = focos_ctx.get("focos") or focos_ctx.get("items") or []
        focos_total = len(lista)

    # ----------------------------------------------------------------------
    # 4) Montar alert_obs (payload completo para IA v11)
    # ----------------------------------------------------------------------
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

    # ----------------------------------------------------------------------
    # 5) Score Inteligente do alerta (v11)
    # ----------------------------------------------------------------------
    alert_score = compute_alert_score(alert_obs)
    alert_level = _classify_level(alert_score["score"])

    # ----------------------------------------------------------------------
    # 6) Score híbrido final (IA + heurística + alerta)
    # ----------------------------------------------------------------------
    ctx2 = dict(ctx)
    ctx2["alert_params"] = alert_obs

    final = compute_final_score(ctx2)

    # ----------------------------------------------------------------------
    # 7) Persistência em NDJSON + transição de nível
    # ----------------------------------------------------------------------
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

    # ----------------------------------------------------------------------
    # 8) Retorno final consolidado
    # ----------------------------------------------------------------------
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

from fastapi import APIRouter

router_admin = APIRouter(prefix="/admin", tags=["Admin"])

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
        "default_scope_inpe": INPE_DEFAULT_SCOPE,
        "default_region_inpe": INPE_DEFAULT_REGION,
    }


# ------------------------------------------------------------
# 14.2 — Diagnóstico do modelo IA
# ------------------------------------------------------------
@router_admin.get("/model_state", summary="Diagnóstico do modelo IA (v11)")
def admin_model_state():
    if modelo_pipeline is None:
        return {"loaded": False, "msg": "Modelo IA v11 não carregado."}

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
# 14.4 — Teste de conexão com serviços externos
# ------------------------------------------------------------
@router_admin.get("/external_check", summary="Verifica serviços externos (Open-Meteo / INPE)")
def admin_external_check():
    status = {}

    # Open-Meteo
    try:
        clima = get_meteo(DEFAULT_LAT, DEFAULT_LON)
        status["open_meteo"] = "ok" if clima else "fail"
    except Exception as e:
        status["open_meteo"] = f"error: {e}"

    # INPE
    try:
        d = inpe_focos_near(DEFAULT_LAT, DEFAULT_LON, 50)
        status["inpe"] = "ok" if d else "fail"
    except Exception as e:
        status["inpe"] = f"error: {e}"

    # DETER
    status["deter"] = "not_implemented"

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

# util
def now_utc():
    return datetime.now(timezone.utc)

# ------------------------------------------------------------
# Diretórios de log
# ------------------------------------------------------------
def _ensure_logs_dir():
    d = "./runtime_data/logs"
    os.makedirs(d, exist_ok=True)
    return d

LOG_DIR = _ensure_logs_dir()
LOG_MAIN = os.path.join(LOG_DIR, "events.ndjson")
LOG_REQUESTS = os.path.join(LOG_DIR, "requests.ndjson")
LOG_IA = os.path.join(LOG_DIR, "ia.ndjson")

# ------------------------------------------------------------
# Escrita em NDJSON
# ------------------------------------------------------------
def _append_jsonl(path: str, rec: Dict[str, Any]):
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception as e:
        print("[LOG_ERROR]", e)

# ------------------------------------------------------------
# Funções principais de log
# ------------------------------------------------------------
def log_event(event_type: str, msg: str, extra: Dict[str, Any] | None = None):
    _append_jsonl(LOG_MAIN, {
        "ts": now_utc().isoformat(),
        "type": event_type,
        "message": msg,
        "extra": extra or {},
    })

def log_request(path: str, method: str, status: int, duration_ms: float, client: str):
    _append_jsonl(LOG_REQUESTS, {
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
    _append_jsonl(LOG_IA, {
        "ts": now_utc().isoformat(),
        "ctx": ctx,
        "prediction": prediction,
        "final": final_score,
    })

# ------------------------------------------------------------
# Middleware de captura de erros
# ------------------------------------------------------------
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
            raise e

app.add_middleware(ErrorLoggerMiddleware)

# ------------------------------------------------------------
# Middleware auditor
# ------------------------------------------------------------
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

# ------------------------------------------------------------
# Endpoints de leitura
# ------------------------------------------------------------
router_logs = APIRouter(prefix="/logs", tags=["Logs"])

@router_logs.get("/tail")
def tail_logs(n: int = 50):
    if not os.path.exists(LOG_MAIN):
        return {"ok": False, "msg": "Nenhum log encontrado."}
    lines = open(LOG_MAIN).read().splitlines()[-n:]
    return {"ok": True, "events": [json.loads(l) for l in lines]}

@router_logs.get("/tail_ia")
def tail_ia(n: int = 50):
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
        "db_url": DB_URL,
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
        ctx = build_observation_context(lat=DEFAULT_LAT, lon=DEFAULT_LON)
        ml = run_ml_model(ctx)
        final = compute_final_score(ctx)
        return {
            "ok": True,
            "ml_raw": ml.get("ml_raw"),
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
        n = (f.get("features") or {}).get("count") or 0
        out["inpe"] = {"ok": True, "focos": n}
    except Exception as e:
        out["inpe"] = {"ok": False, "error": str(e)}

    # DETER (navegação simples)
    try:
        df = load_deter_df()
        out["deter"] = {"ok": True, "rows": len(df) if df is not None else 0}
    except Exception as e:
        out["deter"] = {"ok": False, "error": str(e)}

    return {"ok": True, "providers": out}

# ------------------------------------------------------------
# Disco
# ------------------------------------------------------------
@router_health.get("/health/disk")
def api_health_disk():
    try:
        d = "./runtime_data/healthcheck"
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, "test.txt")
        with open(p, "w", encoding="utf-8") as f:
            f.write(now_utc().isoformat())
        return {"ok": True, "file": p}
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
        "uptime_seconds": time.time() - START_TIME,
        "model_v11_loaded": 1 if modelo_pipeline is not None else 0,
    }
    return PlainTextResponse("\n".join(f"{k} {v}" for k, v in metrics.items()))

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

    try:
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=port,
            reload=(env == "development"),
            log_level="info",
        )
    except Exception as e:
        print("❌ Erro ao iniciar Uvicorn:", e)
        sys.exit(1)

