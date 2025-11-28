
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
# MÓDULO 7 — Sistema de Risco + DETER + Focos + Helpers
# ============================================================

import math
import os
import zipfile
from io import BytesIO
from typing import Optional, Dict, Any

import pandas as pd
import requests
from fastapi import HTTPException

# Reutiliza constantes e helpers já definidos nos módulos anteriores:
# - DEFAULT_LAT, DEFAULT_LON
# - PM25_LIMIT, PM10_LIMIT (podemos reaproveitar os do esboço anterior)
# - ttl_cache, bbox_from_center, haversine_km
# - parse_float_safe, inpe_focos_near
# - INPE_DEFAULT_SCOPE, INPE_DEFAULT_REGION
# - get_meteo, normalize_meteo
# - HTTP_TIMEOUT

# Se os limites de PM ainda não existirem, definimos aqui:
try:
    PM25_LIMIT
except NameError:
    PM25_LIMIT = 35.0  # µg/m³

try:
    PM10_LIMIT
except NameError:
    PM10_LIMIT = 50.0  # µg/m³

# Limiares para o índice final de conservação (0–100)
THRESHOLD_YELLOW = 40
THRESHOLD_RED    = 70

# ------------------------------------------------------------
# 7.1 — Helpers de localização (reaproveitados)
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
    except Exception:
        return None


def _resolve_location(cidade: Optional[str], lat: Optional[float], lon: Optional[float]):
    """
    Regras:
    1) Se cidade for passada → geocodifica
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


# ------------------------------------------------------------
# 7.2 — Carregamento do DETER (parquet em R2)
# ------------------------------------------------------------

DETER_PARQUET_PATH = os.getenv("DETER_PARQUET_PATH", "").strip()
DETER_PARQUET_URL  = os.getenv("DETER_PARQUET_URL", "").strip()
DETER_CACHE_TTL_SEC = int(os.getenv("DETER_CACHE_TTL_SEC", "3600"))

def _read_parquet_from_zipfile(zf: zipfile.ZipFile) -> pd.DataFrame:
    """
    Lê o primeiro arquivo .parquet dentro de um zip.
    """
    for name in zf.namelist():
        if name.lower().endswith(".parquet"):
            with zf.open(name) as f:
                return pd.read_parquet(f)
    raise RuntimeError("Arquivo .zip do DETER não contém nenhum .parquet.")


@ttl_cache(ttl_seconds=DETER_CACHE_TTL_SEC)
def load_deter_df() -> Optional[pd.DataFrame]:
    """
    Carrega o DataFrame unificado do DETER a partir de:
    - DETER_PARQUET_PATH (arquivo local .parquet ou .zip), OU
    - DETER_PARQUET_URL (URL HTTP/HTTPS .parquet ou .zip)

    Resultado fica em cache em memória (TTL configurável).
    """
    # 1) Caminho local (preferência)
    if DETER_PARQUET_PATH and os.path.exists(DETER_PARQUET_PATH):
        path = DETER_PARQUET_PATH
        if path.lower().endswith(".zip"):
            with zipfile.ZipFile(path, "r") as zf:
                return _read_parquet_from_zipfile(zf)
        return pd.read_parquet(path)

    # 2) URL remota (R2 object storage, por exemplo)
    if DETER_PARQUET_URL:
        resp = requests.get(DETER_PARQUET_URL, timeout=HTTP_TIMEOUT)
        resp.raise_for_status()
        data = BytesIO(resp.content)
        if DETER_PARQUET_URL.lower().endswith(".zip"):
            with zipfile.ZipFile(data, "r") as zf:
                return _read_parquet_from_zipfile(zf)
        return pd.read_parquet(data)

    # 3) Se nada foi configurado, retorna None (sem DETER)
    print("[DETER] Nenhuma fonte configurada (DETER_PARQUET_PATH/URL).")
    return None


def _canonical_deter_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza as colunas principais do parquet do DETER:
    - latitude, longitude
    - area_ha (se existir)
    - opcionalmente classe/bioma (mantidos como metadado)
    """
    cols = {c.lower(): c for c in df.columns}

    def pick(*names):
        for n in names:
            if n.lower() in cols:
                return cols[n.lower()]
        return None

    c_lat  = pick("latitude", "lat", "y")
    c_lon  = pick("longitude", "lon", "x")
    c_area = pick("area_ha", "areaha", "area_km2", "area", "tam_ha", "desmat_ha")
    c_cls  = pick("classe", "class")
    c_bio  = pick("bioma", "biome")

    if not c_lat or not c_lon:
        raise RuntimeError("Parquet do DETER sem colunas latitude/longitude conhecidos.")

    out = pd.DataFrame()
    out["latitude"]  = df[c_lat].map(parse_float_safe)
    out["longitude"] = df[c_lon].map(parse_float_safe)

    if c_area:
        # se a coluna estiver em km², converte para ha (1 km² = 100 ha)
        if "km2" in c_area.lower():
            out["area_ha"] = df[c_area].map(parse_float_safe).fillna(0.0) * 100.0
        else:
            out["area_ha"] = df[c_area].map(parse_float_safe)
    else:
        out["area_ha"] = None

    if c_cls:
        out["classe"] = df[c_cls]
    if c_bio:
        out["bioma"] = df[c_bio]

    return out


def deter_stats_near(
    lat: float,
    lon: float,
    raio_km: float = 150.0,
) -> Dict[str, Any]:
    """
    Calcula estatísticas simples de desmatamento do DETER em torno de (lat, lon):

    Retorna:
        {
          "count": N_alertas,
          "total_area_ha": soma_area,
          "score_raw": valor_contínuo,
          "score_norm": [0, 1],
        }
    """
    df = load_deter_df()
    if df is None or df.empty:
        return {
            "count": 0,
            "total_area_ha": 0.0,
            "score_raw": 0.0,
            "score_norm": 0.0,
        }

    norm = _canonical_deter_columns(df)

    minx, miny, maxx, maxy = bbox_from_center(lat, lon, float(raio_km))
    mask = (
        norm["longitude"].notna()
        & norm["latitude"].notna()
        & (norm["longitude"] >= minx)
        & (norm["longitude"] <= maxx)
        & (norm["latitude"]  >= miny)
        & (norm["latitude"]  <= maxy)
    )
    sub = norm[mask].copy()

    # Filtro fino com haversine (se disponível)
    try:
        sub["dist_km"] = sub.apply(
            lambda r: haversine_km(lat, lon, r["latitude"], r["longitude"]),
            axis=1
        )
        sub = sub[sub["dist_km"] <= float(raio_km)]
    except Exception:
        pass

    n = int(len(sub))

    if "area_ha" in sub.columns:
        total_area = float(sub["area_ha"].dropna().sum())
    else:
        # se não há área, considera cada alerta como 1 ha apenas para escalar
        total_area = float(n)

    # Heurística simples:
    # - usa log1p para "achatar" valores muito grandes
    # - combina quantidade de alertas e área
    score_raw = math.log1p(total_area) + 0.5 * math.log1p(n)
    # Normaliza em [0,1] assumindo que valores até ~4 já são "bem altos"
    score_norm = max(0.0, min(1.0, score_raw / 4.0))

    return {
        "count": n,
        "total_area_ha": total_area,
        "score_raw": score_raw,
        "score_norm": score_norm,
    }


# ------------------------------------------------------------
# 7.3 — Estatísticas de Focos (INPE CSV)
# ------------------------------------------------------------

def focos_stats_near(
    lat: float,
    lon: float,
    raio_km: float = 150.0,
    scope: str = INPE_DEFAULT_SCOPE,
    region: str = INPE_DEFAULT_REGION,
) -> Dict[str, Any]:
    """
    Usa inpe_focos_near para obter focos próximos e calcula uma
    estatística simplificada de intensidade.

    Retorna:
        {
          "count": N_focos,
          "frp_sum": soma_FRP,
          "score_raw": valor_contínuo,
          "score_norm": [0,1],
          "meta": {...}
        }
    """
    data = inpe_focos_near(
        lat=lat,
        lon=lon,
        raio_km=raio_km,
        scope=scope,
        region=region,
    )

    feats = (data.get("features") or {})
    focos = feats.get("focos") or []
    meta  = feats.get("meta") or {}

    n = int(feats.get("count") or len(focos) or 0)

    frp_vals = []
    for f in focos:
        v = parse_float_safe(f.get("frp"))
        if v is not None:
            frp_vals.append(v)
    frp_sum = float(sum(frp_vals)) if frp_vals else 0.0

    # Heurística:
    # - mais focos + maior FRP => score maior
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
# 7.4 — ConservationScore (heurístico)
# ------------------------------------------------------------

def _extract_chuva_mm(meteo: Dict[str, Any]) -> float:
    # tentamos manter compatibilidade com diferentes nomes
    chuva = (
        meteo.get("chuva_mm")
        or meteo.get("precipitation")
        or meteo.get("rain")
        or 0.0
    )
    return _as_float_or_none(chuva) or 0.0


def compute_conservation_score(
    meteo: Dict[str, Any],
    deter_stats: Dict[str, Any],
    focos_stats: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Implementa o índice heurístico de conservação ambiental:

        ConservationScore = (chuva + qualidade_do_ar) – (desmatamento + focos)

    Todos os termos são trazidos para a faixa [0,1] e depois
    o resultado é mapeado para [0,100].

    Retorna:
        {
          "score": 0–100,
          "level": "Verde"|"Amarelo"|"Vermelho",
          "components": {
              "chuva_norm",
              "air_quality_norm",
              "desmatamento_norm",
              "focos_norm",
              ...
          }
        }
    """

    # --- Componentes de entrada ---

    # Chuva (mm) – quanto mais, melhor (até certo limite)
    chuva_mm = _extract_chuva_mm(meteo)

    # Poluentes – quanto mais acima do limite, pior
    pm25 = _as_float_or_none(meteo.get("pm25")) or 0.0
    pm10 = _as_float_or_none(meteo.get("pm10")) or 0.0

    # Desmatamento & focos – já normalizados em [0,1]
    desm_norm  = float(deter_stats.get("score_norm") or 0.0)
    focos_norm = float(focos_stats.get("score_norm") or 0.0)

    # --- Normalizações ---

    # Chuva: saturamos em 20 mm (valores maiores não aumentam muito o score)
    chuva_norm = max(0.0, min(1.0, chuva_mm / 20.0))

    # Qualidade do ar:
    #  - ratio 1.0 => no limite
    #  - ratio 2.0 => duas vezes o limite (muito ruim)
    pm25_ratio = pm25 / PM25_LIMIT if PM25_LIMIT else 0.0
    pm10_ratio = pm10 / PM10_LIMIT if PM10_LIMIT else 0.0
    pollution_ratio = max(pm25_ratio, pm10_ratio, 0.0)
    pollution_norm = max(0.0, min(2.0, pollution_ratio)) / 2.0  # 0..1
    air_quality_norm = 1.0 - pollution_norm  # 1 = ar limpo; 0 = muito poluído

    # --- Fórmula heurística ---

    raw = (chuva_norm + air_quality_norm) - (desm_norm + focos_norm)
    # raw está aproximadamente em [-2, +2]. Convertemos para [0,1]:
    norm_0_1 = max(0.0, min(1.0, (raw + 2.0) / 4.0))
    score_0_100 = int(round(norm_0_1 * 100))

    if score_0_100 >= THRESHOLD_RED:
        level = "Vermelho"
    elif score_0_100 >= THRESHOLD_YELLOW:
        level = "Amarelo"
    else:
        level = "Verde"

    return {
        "score": score_0_100,
        "level": level,
        "components": {
            "chuva_mm": chuva_mm,
            "pm25": pm25,
            "pm10": pm10,
            "chuva_norm": chuva_norm,
            "air_quality_norm": air_quality_norm,
            "desmatamento_norm": desm_norm,
            "focos_norm": focos_norm,
        },
    }


# ------------------------------------------------------------
# 7.5 — Construção do “contexto” completo p/ IA (Módulo 8)
# ------------------------------------------------------------

def build_observation_context(
    cidade: Optional[str] = None,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    raio_km: float = 150.0,
) -> Dict[str, Any]:
    """
    Constrói um "contexto" único com:
      - localização resolvida
      - meteo normalizado
      - estatísticas DETER
      - estatísticas focos INPE
      - ConservationScore heurístico

    Esse contexto será usado:
      - para resposta dos endpoints (Módulo 9)
      - como base de features para a IA (Módulo 8)
    """

    # 1) Resolve localização
    lat_res, lon_res, loc_meta = _resolve_location(cidade, lat, lon)

    # 2) Clima + ar (Open-Meteo)
    meteo_raw = get_meteo(lat_res, lon_res) or {}
    meteo     = normalize_meteo(meteo_raw)

    # 3) DETER (desmatamento)
    deter_s = deter_stats_near(lat_res, lon_res, raio_km=raio_km)

    # 4) Focos INPE
    focos_s = focos_stats_near(lat_res, lon_res, raio_km=raio_km)

    # 5) ConservationScore heurístico
    cons = compute_conservation_score(meteo, deter_s, focos_s)

    return {
        "location": {
            "lat": lat_res,
            "lon": lon_res,
            **loc_meta,
        },
        "radius_km": float(raio_km),
        "meteo": meteo,
        "deter": deter_s,
        "focos": focos_s,
        "conservation": cons,
    }


# Ordem sugerida das features para o modelo de ML (Módulo 8)
ML_FEATURE_ORDER = [
    "temperatura",
    "umidade",
    "ponto_orvalho",
    "pressao",
    "vento_m_s",
    "rajadas",
    "chuva_mm",
    "pm25",
    "pm10",
    "desmatamento_score",
    "focos_score",
]


def build_ml_features_from_context(ctx: Dict[str, Any]) -> Dict[str, float]:
    """
    Extrai do contexto as features numéricas que alimentarão o modelo
    de ML (RandomForest v10/v11).

    Retorna um dict com chaves em ML_FEATURE_ORDER (algumas podem
    não existir no modelo final, mas já deixamos preparado).
    """

    meteo = ctx.get("meteo", {}) or {}
    deter_s = ctx.get("deter", {}) or {}
    focos_s = ctx.get("focos", {}) or {}

    chuva_mm = _extract_chuva_mm(meteo)

    feats: Dict[str, float] = {
        "temperatura":      _as_float_or_none(meteo.get("temperature_2m")) or 0.0,
        "umidade":          _as_float_or_none(meteo.get("relativehumidity_2m")) or 0.0,
        "ponto_orvalho":    _as_float_or_none(meteo.get("dewpoint_2m")) or 0.0,
        "pressao":          _as_float_or_none(meteo.get("surface_pressure")) or 0.0,
        "vento_m_s":        _as_float_or_none(meteo.get("windspeed_10m")) or 0.0,
        "rajadas":          _as_float_or_none(meteo.get("windgusts_10m")) or 0.0,
        "chuva_mm":         chuva_mm,
        "pm25":             _as_float_or_none(meteo.get("pm25")) or 0.0,
        "pm10":             _as_float_or_none(meteo.get("pm10")) or 0.0,
        "desmatamento_score": float(deter_s.get("score_norm") or 0.0),
        "focos_score":        float(focos_s.get("score_norm") or 0.0),
    }

    return feats



# ============================================================
# 🧩 MÓDULO 8 — IA AmazonSafe v11 (RandomForest)
# ============================================================

from pydantic import BaseModel
import joblib
import numpy as np
from fastapi import HTTPException
import datetime as dt
import pandas as pd

# ------------------------------------------------------------
# 8.0 — CARREGAMENTO DO MODELO FINAL (v11)
# ------------------------------------------------------------

MODEL_PATH = "models/amazonsafe_pipeline_v11.joblib"

try:
    modelo_pipeline = joblib.load(MODEL_PATH)
    print(f"[IA] Modelo AmazonSafe v11 carregado de {MODEL_PATH}")
except Exception as e:
    print(f"[IA] ERRO ao carregar modelo AmazonSafe v11: {e}")
    modelo_pipeline = None

# ------------------------------------------------------------
# 8.1 — FEATURES OFICIAIS USADAS NO TREINAMENTO
# ------------------------------------------------------------
# Estas são as colunas após limpeza no dataset v11
# (não inclui focos_50km, focos_150km, focos_300km)
# Mantém alinhamento estrito com preprocessamento do Colab v11

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
# 8.2 — MODELO DO PAYLOAD /api/risk
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
# 🔧 Função auxiliar: monta vetor ordenado
# ------------------------------------------------------------

def _build_feature_vector(data: dict) -> np.ndarray:
    """
    Constrói vetor X na ordem exata das MODEL_FEATURES.
    Substitui valores ausentes por 0.0.
    """
    row = []
    for col in MODEL_FEATURES:
        val = data.get(col)
        try:
            row.append(float(val) if val is not None else 0.0)
        except Exception:
            row.append(0.0)

    return np.array([row], dtype=float)

# ------------------------------------------------------------
# 🔧 Função usada pelo Módulo 9 (ML híbrido)
# ------------------------------------------------------------

def run_ml_model(ctx: dict) -> dict:
    """
    Recebe ctx produzido pelo Módulo 7, extrai clima atual,
    gera previsão do modelo e devolve score normalizado (0..1)
    + classificação textual.
    """

    if modelo_pipeline is None:
        return {"ml_raw": 0.0, "ml_level": "desconhecido"}

    met = ctx.get("meteo") or {}
    loc = ctx.get("location") or {}

    payload = met.copy()
    payload["latitude"] = loc.get("lat")
    payload["longitude"] = loc.get("lon")

    X = _build_feature_vector(payload)

    try:
        pred = float(modelo_pipeline.predict(X)[0])
    except Exception as e:
        print(f"[IA run_ml_model] ERRO: {e}")
        return {"ml_raw": 0.0, "ml_level": "erro"}

    # Normalização (assume output entre 0–1 já no treinamento)
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
# 8.3 — ENDPOINT OFICIAL /api/risk
# ------------------------------------------------------------

@app.post("/api/risk", tags=["IA"], summary="Previsão de risco ambiental (v11)")
def api_risk(data: RiskInput):

    if modelo_pipeline is None:
        raise HTTPException(status_code=500, detail="Modelo v11 não carregado")

    entrada = data.model_dump()
    X = _build_feature_vector(entrada)

    try:
        pred = float(modelo_pipeline.predict(X)[0])
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao realizar previsão: {e}"
        )

    # Normalização 0..1
    score = max(0.0, min(1.0, pred))

    if score < 0.33:
        risco = "baixo"
    elif score < 0.66:
        risco = "medio"
    else:
        risco = "alto"

    return {
        "modelo": {
            "path": MODEL_PATH,
            "features": MODEL_FEATURES,
        },
        "entrada": entrada,
        "score": score,
        "risco": risco,
    }

# ------------------------------------------------------------
# 8.4 — Coletor Archive (mantido)
# ------------------------------------------------------------

def collect_weather_archive(lat: float, lon: float, ref_date: str, timeout: int = 20):
    """
    Busca clima horário no Open-Meteo ARCHIVE e calcula média por dia.
    Mantém compatibilidade com dataset v11.
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
            serie = [x for x in h[var] if isinstance(x, (int, float)) and not pd.isna(x)]
            clima[var] = float(sum(serie) / len(serie)) if serie else None

        return clima

    except Exception as e:
        print(f"[IA Archive] ERRO: {e}")
        return None

# ------------------------------------------------------------
# 8.5 — Focos por raio (comparação)
# ------------------------------------------------------------

def focos_por_raios_backend(lat: float, lon: float):
    """
    Mantido para validação e comparação com ML.
    """
    try:
        data = inpe_focos_near(lat, lon, raio_km=300)
        focos = data["features"]["focos"]
    except:
        focos = []

    f50 = f150 = f300 = 0

    for f in focos:
        try:
            d = float(f.get("dist_km"))
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
# 🧩 MÓDULO 9 — IA Leve + Scoring Inteligente (v11) — CORRIGIDO
# ============================================================

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime, timezone
import math
import statistics as stats
from fastapi import HTTPException
from sqlmodel import Session, select

# ------------------------------------------------------------
# 9.0 — Limiares / Pesos
# ------------------------------------------------------------

FINAL_THRESHOLDS = {
    "green_lt": 33,     # score < 33  → Verde
    "yellow_lt": 66,    # score < 66  → Amarelo
}

FINAL_WEIGHTS = {
    "heuristic": 0.40,   # ConservationScore (chuva + ar – desmatamento – focos)
    "ml":        0.40,   # modelo RandomForest v11
    "alerts":    0.15,   # severity/duration/frequency/impact
    "mad":       0.05,   # penalizações por outliers
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
# 9.2 — Funções utilitárias (mantidas e revisadas)
# ============================================================

def _clip01(x: float) -> float:
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


def _rainfall_index(mm: Optional[float]) -> float:
    """
    Conversão chuva → risco 0..1 (mantido do módulo antigo)
    0 mm  → 0
    50 mm → 1
    """
    if mm is None:
        return 0.0
    try:
        mm = float(mm)
    except:
        return 0.0

    if mm <= 0:
        return 0.0
    if mm >= 50:
        return 1.0
    return mm / 50.0


# ============================================================
# 9.3 — SCORE DE ALERTA (severity/duration/freq/impact + chuva)
# ============================================================

ALERT_WEIGHTS = {
    "severity": 0.25,
    "duration": 0.25,
    "frequency": 0.25,
    "impact":   0.15,
    "rainfall": 0.10,
}

def compute_alert_score(alert_obs: Dict[str, Any]) -> Dict[str, Any]:
    sev = _clip01(alert_obs.get("severity", 0))
    dur = _clip01(alert_obs.get("duration", 0))
    freq = _clip01(alert_obs.get("frequency", 0))
    imp = _clip01(alert_obs.get("impact", 0))

    precip_24h = (
        alert_obs.get("precip_24h")
        or alert_obs.get("chuva_mm")
        or alert_obs.get("precipitation")
        or (alert_obs.get("meta", {}).get("precipitation"))
        or 0.0
    )

    rainfall = _rainfall_index(precip_24h)

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
            "precip_24h_mm": float(precip_24h),
        }
    }


# ============================================================
# 9.4 — MAD (outliers ambientais)
# ============================================================

def _round_coord(v: float, ndigits: int = 3):
    return round(float(v), ndigits)


def _mad(values: List[float]):
    if not values:
        return None
    med = stats.median(values)
    devs = [abs(x - med) for x in values]
    mad = stats.median(devs)
    return med, mad


def pm_outlier_flags(lat: float, lon: float, pm25, pm10, k=5.0, lookback=50):
    """
    Preservado do módulo antigo.
    Verifica se pm25/pm10 são outliers em relação ao histórico local.
    """
    if pm25 is None and pm10 is None:
        return False, False

    latk = _round_coord(lat)
    lonk = _round_coord(lon)

    # WeatherObs está definido no Módulo 6
    from main import WeatherObs, engine

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

    if pm25 is not None and len(pm25_hist) >= 10:
        mad_res = _mad(pm25_hist)
        if mad_res:
            med, mad = mad_res
            if mad and abs(pm25 - med) > k * 1.4826 * mad:
                flag25 = True

    if pm10 is not None and len(pm10_hist) >= 10:
        mad_res = _mad(pm10_hist)
        if mad_res:
            med, mad = mad_res
            if mad and abs(pm10 - med) > k * 1.4826 * mad:
                flag10 = True

    return flag25, flag10


# ============================================================
# 9.5 — SCORE FINAL (Fusão IA + heurística + alerts + MAD)
# ============================================================

def _extract_heuristic_from_ctx(ctx: Dict[str, Any]) -> Tuple[float, Optional[str]]:
    """
    Extrai (heuristic_score, heuristic_level) do contexto.

    Prioridade:
      1) ctx["conservation"]["score"/"level"]  (Módulo 7)
      2) ctx["heuristic_score"] / ctx["heuristic_level"] (compatibilidade)
    """
    cons = ctx.get("conservation") or {}
    score = cons.get("score")
    level = cons.get("level")

    if score is None:
        score = ctx.get("heuristic_score")

    if level is None:
        level = ctx.get("heuristic_level")

    try:
        score_f = float(score or 0.0)
    except Exception:
        score_f = 0.0

    return score_f, level


def compute_final_score(ctx: Dict[str, Any]) -> FinalScoreResult:
    """
    ctx vem do Módulo 7 (build_observation_context) + possíveis campos extras
    injetados pelos módulos 8/13, e contém tipicamente:

        - ctx["location"]
        - ctx["meteo"]
        - ctx["deter"]
        - ctx["focos"]
        - ctx["conservation"] = {score, level, components}
        - opcionalmente ctx["ml_raw"], ctx["ml_level"]
        - opcionalmente ctx["alert_params"]

    A fusão final combina:
        - heurística (ConservationScore)        → peso 0.40
        - IA (RandomForest v11, ml_raw)        → peso 0.40
        - alerta (severity/duration/freq/...)  → peso 0.15
        - penalidade MAD (outliers PM)         → peso 0.05
    """

    # ----------------------------------------
    # 1) heuristic_score (ConservationScore)
    # ----------------------------------------
    heuristic_score, heuristic_level = _extract_heuristic_from_ctx(ctx)
    h_score_norm = float(heuristic_score) / 100.0  # normaliza 0..1

    # ----------------------------------------
    # 2) ml_raw (já é 0..1 ou valor contínuo [0,1])
    # ----------------------------------------
    ml_raw = ctx.get("ml_raw", 0.0)
    try:
        ml_norm = float(ml_raw)
    except Exception:
        ml_norm = 0.0

    # ----------------------------------------
    # 3) alert_score
    # ----------------------------------------
    alert_obs = ctx.get("alert_params") or {}
    alert_data = compute_alert_score(alert_obs) if alert_obs else {
        "score": 0.0,
        "components": {
            "severity": 0.0,
            "duration": 0.0,
            "frequency": 0.0,
            "impact": 0.0,
            "rainfall": 0.0,
            "precip_24h_mm": 0.0,
        },
    }
    alert_norm = _clip01(alert_data["score"])

    # ----------------------------------------
    # 4) MAD penalties (outliers PM)
    # ----------------------------------------
    met = ctx.get("meteo", {}) or {}
    lat = ctx.get("location", {}).get("lat")
    lon = ctx.get("location", {}).get("lon")

    mad_penalty = 0.0
    if lat is not None and lon is not None:
        p25_flag, p10_flag = pm_outlier_flags(lat, lon, met.get("pm25"), met.get("pm10"))
        if p25_flag:
            mad_penalty += 0.3
        if p10_flag:
            mad_penalty += 0.3
        mad_penalty = min(1.0, mad_penalty)

    # ----------------------------------------
    # 5) FUSÃO FINAL
    # ----------------------------------------

    final = (
        h_score_norm * FINAL_WEIGHTS["heuristic"]
        + ml_norm      * FINAL_WEIGHTS["ml"]
        + alert_norm   * FINAL_WEIGHTS["alerts"]
        - mad_penalty  * FINAL_WEIGHTS["mad"]
    )

    final_score = max(0.0, min(1.0, final)) * 100.0
    level = _classify_level(final_score)

    breakdown = {
        "heuristic_score_norm": h_score_norm,
        "heuristic_raw_score": heuristic_score,
        "heuristic_level": heuristic_level,
        "ml_score_norm": ml_norm,
        "ml_raw": ml_raw,
        "alert_score_norm": alert_norm,
        "mad_penalty": mad_penalty,
        "weights": FINAL_WEIGHTS,
        "components": {
            "alert_details": alert_data,
            "meteo": ctx.get("meteo"),
            "deter": ctx.get("deter"),
            "focos": ctx.get("focos"),
            "conservation": ctx.get("conservation"),
        },
    }

    return FinalScoreResult(
        score=round(final_score, 2),
        level=level,
        breakdown=breakdown,
    )


# ============================================================
# 9.6 — ENDPOINT OFICIAL /api/score_final
# ============================================================

class RiskRequest(BaseModel):
    cidade: str | None = None
    lat: float | None = None
    lon: float | None = None
    raio_km: int = 150   # 🔥 obrigatório

@app.post("/api/score_final", tags=["IA"], summary="Score híbrido (ML + heurística + MAD + alertas)")
def api_score_final(body: RiskRequest):

    ctx = build_observation_context(
        cidade=body.cidade,
        lat=body.lat,
        lon=body.lon,
        raio_km=body.raio_km
    )

    # ML interno
    ml_res = run_ml_model(ctx)
    ctx["ml_raw"] = ml_res.get("ml_raw")
    ctx["ml_level"] = ml_res.get("ml_level")

    # Score final híbrido
    final = compute_final_score(ctx)

    return {
        "ok": True,
        "location": ctx["location"],
        "final_score": final.score,
        "final_level": final.level,
        "breakdown": final.breakdown,
        "context": ctx,
    }



# ============================================================
# 🧩 MÓDULO 10 — COLETORES ATUAIS PARA O DASHBOARD (v11)
# ============================================================

def collect_weather_now(lat: float, lon: float) -> dict:
    """
    Coleta clima atual usando get_meteo() e normalize_meteo().
    Compatível com o dataset v11.
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

    clima_norm = normalize_meteo(clima)

    return {
        "ok": True,
        "fonte": "open-meteo",
        "latitude": lat,
        "longitude": lon,
        "features": clima_norm,
    }


def collect_focos_now(lat: float, lon: float) -> dict:
    """Focos reais em 50, 150 e 300 km — via backend (v11)."""
    try:
        focos = focos_por_raios_backend(lat, lon)
    except Exception as e:
        return {"ok": False, "erro": str(e)}

    return {
        "ok": True,
        "latitude": lat,
        "longitude": lon,
        "focos": focos,
    }


def collect_dashboard_bundle(lat: float, lon: float) -> dict:
    """
    Pacote unificado para o dashboard — versão v11.
    Agora integrado com DETER + clima + focos.
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
    # 1) Resolver localização e gerar contexto completo (v11)
    # ------------------------------------------------------------
    try:
        ctx = build_observation_context(
            cidade=req.cidade,
            lat=req.lat,
            lon=req.lon,
            raio_km=req.raio_km
        )
    except HTTPException as e:
        raise
    except Exception as e:
        raise HTTPException(400, detail=f"Erro ao obter contexto: {e}")

    lat = ctx["location"]["lat"]
    lon = ctx["location"]["lon"]

    # ------------------------------------------------------------
    # 2) Clima atual
    # ------------------------------------------------------------
    clima_now = collect_weather_now(lat, lon)

    # ------------------------------------------------------------
    # 3) Focos reais
    # ------------------------------------------------------------
    focos_now = collect_focos_now(lat, lon)

    # ------------------------------------------------------------
    # 4) DETER (já no ctx via build_observation_context)
    # ------------------------------------------------------------
    deter = ctx.get("deter")

    # ------------------------------------------------------------
    # 5) Heurística + IA v11
    # ------------------------------------------------------------
    heuristico = {
        "score": ctx.get("heuristic_score"),
        "level": ctx.get("heuristic_level"),
    }

    ml = {
        "score_raw": ctx.get("ml_raw"),
        "level": ctx.get("ml_level"),
    }

    # ------------------------------------------------------------
    # 6) Score híbrido final
    # ------------------------------------------------------------
    final = compute_final_score(ctx)

    # ------------------------------------------------------------
    # 7) Resposta completa v11
    # ------------------------------------------------------------
    return {
        "ok": True,
        "local": ctx["location"],
        "clima_atual": clima_now,
        "focos_reais": focos_now,
        "deter": deter,
        "heuristica": heuristico,
        "ml_v11": ml,
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
    # 1) Gerar contexto completo (clima + DETER + focos + heurística)
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
            status_code=400,
            detail=f"Erro ao construir contexto: {e}",
        )

    lat = ctx["location"]["lat"]
    lon = ctx["location"]["lon"]

    # ------------------------------------------------------------
    # 2) Clima atual (Open-Meteo Forecast + AQ)
    # ------------------------------------------------------------
    try:
        clima_now = collect_weather_now(lat, lon)
    except Exception as e:
        clima_now = {
            "ok": False,
            "erro": str(e),
            "fonte": "open-meteo",
            "features": {},
        }

    # ------------------------------------------------------------
    # 3) Focos reais (50 / 150 / 300 km)
    # ------------------------------------------------------------
    try:
        focos_reais = collect_focos_now(lat, lon)
    except Exception as e:
        focos_reais = {"ok": False, "erro": str(e)}

    # ------------------------------------------------------------
    # 4) IA v11 — RandomForest (run_ml_model)
    # ------------------------------------------------------------
    ml_res = run_ml_model(ctx)
    ctx["ml_raw"] = ml_res.get("ml_raw")
    ctx["ml_level"] = ml_res.get("ml_level")

    # ------------------------------------------------------------
    # 5) Heurística (ConservationScore) + Score Final Híbrido
    # ------------------------------------------------------------
    # ConservationScore vem do Módulo 7 (ctx["conservation"])
    cons = ctx.get("conservation") or {}
    heuristic_score = cons.get("score")
    heuristic_level = cons.get("level")

    # Score híbrido final (Módulo 9)
    final = compute_final_score(ctx)

    # Mantemos o espírito do v10:
    # - "risco_simples" = heurístico puro
    # - "score_hibrido" e "risco_hibrido" = score final híbrido
    risco_simples = heuristic_level or "Indefinido"
    score_hibrido = final.score
    risco_hibrido = final.level

    # ------------------------------------------------------------
    # 6) Retorno consolidado para o front (v11)
    # ------------------------------------------------------------
    return {
        "ok": True,
        "local": {
            "cidade_entrada": req.cidade,
            "lat": lat,
            "lon": lon,
            "resolved_by": ctx["location"].get("resolved_by"),
            "display_name": ctx["location"].get("display_name"),
            "radius_km": ctx.get("radius_km"),
        },
        "clima_atual": {
            "fonte": clima_now.get("fonte"),
            "features": clima_now.get("features") or {},
            "raw": clima_now,
        },
        "focos_reais": focos_reais,
        "desmatamento": ctx.get("deter"),
        "heuristica": {
            "score": heuristic_score,
            "level": heuristic_level,
            "components": cons.get("components"),
        },
        "ia": {
            "modelo_path": MODEL_PATH,
            "ml_raw": ml_res.get("ml_raw"),
            "ml_level": ml_res.get("ml_level"),
        },
        "score_hibrido": {
            "score": score_hibrido,
            "level": risco_hibrido,
            "breakdown": final.breakdown,
        },
        # Mantém os nomes antigos para facilitar migração do front:
        "ia_compat": {
            "risco_simples": risco_simples,
            "score_hibrido": round(score_hibrido, 2),
            "risco_hibrido": risco_hibrido,
        },
        "contexto": ctx,
    }

# ============================================================
# 🧩 MÓDULO 13 — Atualização de Alertas + Score Inteligente (v11)
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
    summary="Atualiza alertas e calcula score inteligente (v11)",
)
def api_alertas_update(req: AlertUpdateRequest):
    """
    Fluxo v11:
      1) resolve localização + contexto ambiental completo (build_observation_context)
      2) extrai clima + qualidade do ar + focos + desmatamento
      3) monta alert_obs com dados manuais + ambiente
      4) calcula score inteligente de alerta (compute_alert_score)
      5) injeta alert_params no contexto e calcula score híbrido (compute_final_score)
      6) persiste score/nível em NDJSON (save_alert_score + handle_level_transition)
      7) retorna resposta consolidada
    """

    # --------------------------------------------------------
    # 1) Contexto ambiental completo
    # --------------------------------------------------------
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
        raise HTTPException(400, f"Erro ao construir contexto: {e}")

    loc = ctx["location"]
    lat = loc["lat"]
    lon = loc["lon"]

    meteo = ctx.get("meteo") or {}
    focos_ctx = ctx.get("focos") or {}
    deter_ctx = ctx.get("deter") or {}
    conservation = ctx.get("conservation") or {}

    # --------------------------------------------------------
    # 2) Clima + Qualidade do ar
    # --------------------------------------------------------
    # meteo já vem do get_meteo() + normalize_meteo() no Módulo 7/10.
    # Campos relevantes:
    #  - precipitation (chuva)
    #  - pm25, pm10
    #  - etc.
    precip = (
        meteo.get("chuva_mm")
        or meteo.get("precipitation")
        or meteo.get("precip_24h")
    )

    pm25 = meteo.get("pm25")
    pm10 = meteo.get("pm10")

    # --------------------------------------------------------
    # 3) Focos: quantidade total (no raio)
    # --------------------------------------------------------
    # No contexto, podemos ter tanto um resumo quanto lista.
    # Vamos tentar em ordem:
    focos_total = (
        focos_ctx.get("total")
        or focos_ctx.get("count")
        or focos_ctx.get("focos_total")
    )

    if focos_total is None:
        # fallback: se tiver lista
        items = focos_ctx.get("items") or focos_ctx.get("focos") or []
        focos_total = len(items)

    # --------------------------------------------------------
    # 4) Montagem de alert_obs (entrada do Score Inteligente)
    # --------------------------------------------------------
    alert_obs = {
        "severity": req.severity,
        "duration": req.duration,
        "frequency": req.frequency,
        "impact": req.impact,

        "precipitation": precip,
        "meta": {
            "precipitation": precip,
            "radius_km": req.raio_km,
        },

        "pm25": pm25,
        "pm10": pm10,

        "focos_total": focos_total,
    }

    # --------------------------------------------------------
    # 5) Score Inteligente de alerta (Módulo 9)
    # --------------------------------------------------------
    score_data = compute_alert_score(alert_obs)

    # --------------------------------------------------------
    # 6) Score híbrido v11 (opcional, integrando alerta ao contexto)
    # --------------------------------------------------------
    ctx_alert = dict(ctx)
    ctx_alert["alert_params"] = alert_obs

    final = compute_final_score(ctx_alert)

    # --------------------------------------------------------
    # 7) Persistência NDJSON + transição de nível
    # --------------------------------------------------------
    alert_id = (
        req.cidade.lower().replace(" ", "_")
        if req.cidade
        else f"{lat:.4f},{lon:.4f}"
    )

    save_alert_score(
        alert_id=alert_id,
        score=score_data["score"],
        level=_classify_level(score_data["score"]),  # usa função do Módulo 9
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
        new_level=_classify_level(score_data["score"]),
        score=score_data["score"],
        alert_obs=alert_obs,
    )

    # --------------------------------------------------------
    # 8) Resposta final
    # --------------------------------------------------------
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
            "score": score_data["score"],
            "components": score_data["components"],
            "nivel_simplificado": _classify_level(score_data["score"]),
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
        "version": "1.1.0-v11",
        "model_loaded": modelo_pipeline is not None,
        "model_path": MODEL_PATH,
        "db_url": DB_URL,
        "cache_ttl_sec": CACHE_TTL_SEC,
        "default_scope_inpe": INPE_DEFAULT_SCOPE,
        "default_region_inpe": INPE_DEFAULT_REGION,
    }


# ------------------------------------------------------------
# 14.2 — Diagnóstico do modelo IA (v11)
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

    # Teste Open-Meteo
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
            timeout=10,
        )
        status["inpe"] = "ok" if d else "fail"
    except Exception as e:
        status["inpe"] = f"error: {e}"

    # DETER / Cloudflare (se você tiver um helper tipo get_deter_summary, pode adicionar aqui)
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


# ------------------------------------------------------------
# 14.6 — Limpeza de cache (corrigida para ttl_cache)
# ------------------------------------------------------------
@router_admin.post("/clear_cache", summary="Limpa caches internos (get_meteo / inpe_focos_near)")
def admin_clear_cache():
    cleared = {}

    for fn in (get_meteo, inpe_focos_near):
        name = fn.__name__
        try:
            cache = getattr(fn, "_cache", None)
            if isinstance(cache, dict):
                n = len(cache)
                cache.clear()
                cleared[name] = n
            else:
                cleared[name] = 0
        except Exception:
            cleared[name] = "erro"

    return {"ok": True, "cleared": cleared}


# ------------------------------------------------------------
# Registrar router
# ------------------------------------------------------------
app.include_router(router_admin)



# ============================================================
# 🧩 MÓDULO 15 — Logging, Auditoria e Telemetria Interna (v11)
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
# 15.0 — Diretórios de log v11
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
# 15.1 — Escrita genérica em NDJSON
# ------------------------------------------------------------
def _append_jsonl(path: str, record: Dict[str, Any]):
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        print("[LOG_ERROR]", e)


# ------------------------------------------------------------
# 15.2 — Função central de registro
# ------------------------------------------------------------
def log_event(event_type: str, message: str, extra: Dict[str, Any] | None = None):
    rec = {
        "ts": datetime.utcnow().isoformat() + "Z",
        "type": event_type,
        "message": message,
        "extra": extra or {},
    }
    _append_jsonl(LOG_MAIN, rec)


def log_request(path: str, method: str, status: int, duration_ms: float, client: str):
    rec = {
        "ts": datetime.utcnow().isoformat() + "Z",
        "path": path,
        "method": method,
        "status": status,
        "duration_ms": duration_ms,
        "client": client,
    }
    _append_jsonl(LOG_REQUESTS, rec)


def log_inference(ctx: Dict[str, Any], prediction: Any, final_score: Any = None):
    rec = {
        "ts": datetime.utcnow().isoformat() + "Z",
        "ctx_location": ctx.get("location"),
        "ctx_meteo": ctx.get("meteo"),
        "ctx_focos": ctx.get("focos"),
        "ctx_deter": ctx.get("deter"),
        "ctx_conservation": ctx.get("conservation"),
        "prediction": prediction,
        "final_score": final_score.dict() if hasattr(final_score, "dict") else final_score,
    }
    _append_jsonl(LOG_IA, rec)


# ------------------------------------------------------------
# 15.3 — Middleware unificado v11
# ------------------------------------------------------------
@app.middleware("http")
async def audit_api_calls(request: Request, call_next):
    start = time.time()
    path = request.url.path
    method = request.method

    response = await call_next(request)
    duration = round((time.time() - start) * 1000, 2)

    log_request(
        path=path,
        method=method,
        status=response.status_code,
        duration_ms=duration,
        client=request.client.host if request.client else None,
    )

    return response


# ------------------------------------------------------------
# 15.4 — Endpoints de leitura
# ------------------------------------------------------------
@router_logs.get("/tail", summary="Últimos N logs gerais v11")
def tail_logs(n: int = 50):
    if not os.path.exists(LOG_MAIN):
        return {"ok": False, "msg": "Nenhum log encontrado."}

    with open(LOG_MAIN, "r", encoding="utf-8") as f:
        lines = f.readlines()[-n:]

    return {"ok": True, "events": [json.loads(l) for l in lines]}


@router_logs.get("/tail_ia", summary="Últimos N logs de IA")
def tail_logs_ia(n: int = 50):
    if not os.path.exists(LOG_IA):
        return {"ok": False, "msg": "Nenhum log IA encontrado."}

    with open(LOG_IA, "r", encoding="utf-8") as f:
        lines = f.readlines()[-n:]

    return {"ok": True, "events": [json.loads(l) for l in lines]}


# ------------------------------------------------------------
# 15.5 — Log manual
# ------------------------------------------------------------
@router_logs.post("/push", summary="Gravar log manual")
def push_log(event_type: str, message: str):
    log_event(event_type, message)
    return {"ok": True, "msg": "Log registrado."}


# ------------------------------------------------------------
# Registrar router
# ------------------------------------------------------------
app.include_router(router_logs)

# ============================================================
# 🧩 MÓDULO 16 — Healthcheck, Métricas e Autoverificação (v11)
# ============================================================

import os
import time
import numpy as np
from fastapi import APIRouter
from fastapi.responses import PlainTextResponse

router_health = APIRouter(prefix="/system", tags=["Sistema"])

START_TIME = time.time()


# ------------------------------------------------------------
# 16.1 — Health básico
# ------------------------------------------------------------
@router_health.get("/health", summary="Health básico v11")
def api_health():
    return {
        "ok": True,
        "version": "1.1.0-v11",
        "timestamp": now_utc().isoformat().replace("+00:00", "Z"),
        "model_loaded": modelo_pipeline is not None,
        "db_url": DB_URL,
    }


@app.get("/health", tags=["Sistema"])
def root_health():
    return {"ok": True, "status": "online", "path": "/system/health"}


# ------------------------------------------------------------
# 16.2 — Teste do modelo IA v11 (pipeline + scoring)
# ------------------------------------------------------------
@router_health.get("/health/model_v11")
def api_health_model_v11():

    try:
        # contexto mínimo válido
        ctx = build_observation_context(lat=DEFAULT_LAT, lon=DEFAULT_LON)

        pred = run_ml_model(ctx)
        final = compute_final_score(ctx)

        return {
            "ok": True,
            "prediction": float(pred),
            "final_score": final.score,
            "final_level": final.level,
        }

    except Exception as e:
        return {"ok": False, "error": str(e)}


# ------------------------------------------------------------
# 16.3 — Teste do DETER via Cloudflare R2
# ------------------------------------------------------------
@router_health.get("/health/deter")
def api_health_deter():
    try:
        d = get_deter_summary()  # se você nomeou diferente, só ajustar
        return {"ok": True, "files": len(d)}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ------------------------------------------------------------
# 16.4 — Teste de providers externos (Open-Meteo / INPE / DETER)
# ------------------------------------------------------------
@router_health.get("/health/providers")
def api_health_providers():
    out = {}

    # Open-Meteo
    try:
        c = get_meteo(DEFAULT_LAT, DEFAULT_LON)
        out["open_meteo"] = {"ok": True, "temp": c.get("temperatura")}
    except Exception as e:
        out["open_meteo"] = {"ok": False, "error": str(e)}

    # INPE
    try:
        d = inpe_focos_near(DEFAULT_LAT, DEFAULT_LON, 50)
        n = (d.get("features") or {}).get("count") or 0
        out["inpe"] = {"ok": True, "focos": n}
    except Exception as e:
        out["inpe"] = {"ok": False, "error": str(e)}

    # DETER
    try:
        d = get_deter_summary()
        out["deter"] = {"ok": True, "files": len(d)}
    except:
        out["deter"] = {"ok": False}

    return {"ok": True, "providers": out}


# ------------------------------------------------------------
# 16.5 — Geocode
# ------------------------------------------------------------
@router_health.get("/health/geocode")
def api_health_geocode():
    try:
        info = geocode_city("Belém, PA")
        return {"ok": True, "result": info}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ------------------------------------------------------------
# 16.6 — Disco
# ------------------------------------------------------------
@router_health.get("/health/disk")
def api_health_disk():
    try:
        d = "./runtime_data/healthcheck"
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, "test.txt")
        ts = now_utc().isoformat()
        with open(path, "w", encoding="utf-8") as f:
            f.write(ts)
        return {"ok": True, "file": path}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ------------------------------------------------------------
# 16.7 — DB
# ------------------------------------------------------------
@router_health.get("/health/db")
def api_health_db():
    try:
        with engine.connect() as conn:
            conn.execute("SELECT 1")
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ------------------------------------------------------------
# 16.8 — Métricas estilo Prometheus v11
# ------------------------------------------------------------
@router_health.get("/metrics")
def api_metrics():

    metrics = {
        "uptime_seconds": time.time() - START_TIME,
        "model_v11_loaded": 1 if modelo_pipeline is not None else 0,
        "cache_ttl_seconds": CACHE_TTL_SEC,
    }

    lines = [f"{k} {v}" for k, v in metrics.items()]
    return PlainTextResponse("\n".join(lines))


# ------------------------------------------------------------
# Registrar router
# ------------------------------------------------------------
app.include_router(router_health)



# ============================================================
# 🧩 MÓDULO 17 — Execução Local / Compatibilidade Render (v11)
# ============================================================

import os
import sys
import uvicorn

if __name__ == "__main__":
    """
    Execução local:
        python main.py

    Observações:
    - Este bloco é ignorado no Render.
    - O Render usa automaticamente:
          uvicorn main:app --host 0.0.0.0 --port $PORT
    """

    port = int(os.getenv("PORT", 8000))
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
        print("❌ Erro ao iniciar o servidor Uvicorn:", e)
        sys.exit(1)


# ============================================================
# Nota:
# No Render, este bloco NÃO é executado.
# O Render invoca automaticamente:
#   uvicorn main:app --host 0.0.0.0 --port $PORT
# ============================================================
