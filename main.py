# main.py — extraído automaticamente do notebook AmazonSafe_API_v5.ipynb
# Gerado para deploy (Render/Railway).
# Observações:
# - Magics do Jupyter (!, %, get_ipython) removidos.
# - O processo inicia com:  uvicorn main:app --host 0.0.0.0 --port ${PORT}
# - Para Postgres, defina DATABASE_URL; caso contrário, usa SQLite local.

import os
import sys
import math
import csv
import io
import json
import time
import threading
import datetime as dt

# DB/Engine
from sqlmodel import create_engine, SQLModel

# Fallback de banco: Railway/Render Postgres -> SQLite
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
DB_URL = DATABASE_URL if DATABASE_URL else "sqlite:///./amazonsafe.db"
engine = create_engine(DB_URL, pool_pre_ping=True)

# ===== Imports de terceiros (fixados via requirements.txt) =====
# Log de versões (opcional) para diagnóstico
try:
    import sqlmodel, sqlalchemy, pydantic, fastapi, requests, pandas  # noqa
    print("OK: sqlmodel", sqlmodel.__version__,
          "| SQLAlchemy", sqlalchemy.__version__,
          "| pydantic", pydantic.__version__)
except Exception as e:
    print("AVISO: alguma dependência pode estar faltando:", e)

# FastAPI / server
from fastapi import FastAPI, Query, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware

# Tipos e SQLModel
from typing import Optional, List, Dict, Any, Tuple, Literal
from sqlmodel import Field, Session, select

# HTTP clients / Data
import requests
import pandas as pd
import httpx  # usado nos adapters/resolver de qualidade do ar

# === Análise/IA ===
from analytics.features import build_features                  
from analytics.rules   import classify_by_rules               
MODEL_VERSION = "rf_v1"                                        


# JSON rápido (orjson se disponível)
try:
    import orjson
    _json_dumps = lambda obj: orjson.dumps(obj).decode("utf-8")
except Exception:
    _json_dumps = lambda obj: json.dumps(obj, ensure_ascii=False)

# --- NDJSON de alertas (para Grupo 3/4 lerem sem SQL) ---
DATA_DIR = os.getenv("DATA_DIR", "data")
os.makedirs(DATA_DIR, exist_ok=True)
ALERTAS_NDJSON_PATH = os.path.join(DATA_DIR, "alertas.ndjson")

def _utc_now_iso():
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def _append_alerta_ndjson(alert_id: str, sr, alert_obs: dict):
    """
    Grava uma linha NDJSON com o último score/nível e alguns metadados úteis.
    """
    try:
        meta = (alert_obs or {}).get("meta", {})
        rec = {
            "ts": _utc_now_iso(),
            "alert_id": alert_id,
            "score": round(sr.score, 3),
            "level": sr.level,
            # breakdown do score (se existir)
            "severity": (sr.breakdown or {}).get("severity"),
            "duration": (sr.breakdown or {}).get("duration"),
            "frequency": (sr.breakdown or {}).get("frequency"),
            "impact": (sr.breakdown or {}).get("impact"),
            # alguns campos de meta para gráficos
            "pm25": meta.get("pm25"),
            "pm10": meta.get("pm10"),
            "precipitation": meta.get("precipitation"),
            "wind_speed_10m": meta.get("wind_speed_10m"),
            "focos_count": meta.get("focos_count"),
            "observed_at": meta.get("observed_at"),
        }
        with open(ALERTAS_NDJSON_PATH, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception as e:
        # não quebra o endpoint se falhar a escrita; apenas registra
        print("[warn] append NDJSON:", e)

# ===== Config de chaves/constantes =====
# Coordenadas padrão (Belém)
DEFAULT_LAT = float(os.getenv("DEFAULT_LAT", "-1.4558"))
DEFAULT_LON = float(os.getenv("DEFAULT_LON", "-48.5039"))

# OpenAQ v3
OPENAQ_API_KEY = os.getenv("OPENAQ_API_KEY", "f6713d7b945cc5d989cdc08bcb44b62c0f343f11e0f1080555d0b768283ce101")

# ANA (SOAP antigo para inventário)
ANA_SOAP_BASE = "https://telemetriaws1.ana.gov.br/ServiceANA.asmx"

# INPE (Queimadas) - CSV (dataserver COIDS)
INPE_CSV_BASE = os.getenv(
    "INPE_CSV_BASE",
    "https://dataserver-coids.inpe.br/queimadas/queimadas/focos/csv"
).rstrip("/")

# Escopo/recorte padrão para o INPE
INPE_DEFAULT_SCOPE = os.getenv("INPE_DEFAULT_SCOPE", "diario").lower()  # "diario" ou "mensal"
INPE_DEFAULT_REGION = os.getenv("INPE_DEFAULT_REGION", "Brasil")        # "Brasil" ou "America_Sul"

# Cache simples (TTL em segundos)
CACHE_TTL_SEC = 300
HTTP_TIMEOUT = int(os.getenv("HTTP_TIMEOUT", "60"))

# Limiares/weights do score de risco (mantidos para compat)
PM25_LIMIT = 35
PM10_LIMIT = 50
PRECIP_GOOD_MM = 5.0
WEIGHT_PM25 = 40
WEIGHT_PM10 = 30
WEIGHT_DRY  = 30
THRESHOLD_YELLOW = 40
THRESHOLD_RED    = 70

# Timezone/UTC helper
UTC = dt.timezone.utc
def now_utc():
    return dt.datetime.now(UTC)

print("Config ok.",
      "\n- INPE_CSV_BASE:", INPE_CSV_BASE,
      "\n- INPE_DEFAULT_SCOPE:", INPE_DEFAULT_SCOPE,
      "\n- INPE_DEFAULT_REGION:", INPE_DEFAULT_REGION)

# OpenWeatherMap - Config de chave
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "").strip()

# >>> AIR-CONFIG-START
AIR_TIMEOUT_SECS = float(os.getenv("AIR_TIMEOUT_SECS", "7.0"))
AIR_STALE_MINUTES = int(os.getenv("AIR_STALE_MINUTES", "180"))
AIR_RADIUS_SEQ = os.getenv("AIR_RADIUS_SEQ", "10000,25000,50000,75000")
AIR_ENABLE_MODEL_FALLBACK = os.getenv("AIR_ENABLE_MODEL_FALLBACK", "1") == "1"
# >>> AIR-CONFIG-END


# ==== [cell] =============================================
## 1) Utilitários (cache TTL, geocoder, haversine, helpers)
# Cache TTL simples em memória
import time

def ttl_cache(ttl_seconds: int | None = None):
    """
    Decorator de cache em memória com TTL.
    Ex.: @ttl_cache(3600)  # 1 hora
    """
    if ttl_seconds is None:
        # fallback para o valor global, se existir
        ttl_seconds = globals().get("CACHE_TTL_SEC", 300)

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


def cache_stats():
    # soma entradas aproximadas
    return {"entries": sum(len(getattr(v, "_cache", {})) for v in globals().values() if callable(v))}


# --- Geocoding robusto: Nominatim + fallback Open-Meteo ---------------------
import re
from typing import Optional

GEOCODE_UA = "AmazonSafe/3 (+https://amazonsafe-api.onrender.com; contato: seu-email@exemplo)"
UF2STATE = {
    "AC":"Acre","AL":"Alagoas","AP":"Amapá","AM":"Amazonas","BA":"Bahia","CE":"Ceará",
    "DF":"Distrito Federal","ES":"Espírito Santo","GO":"Goiás","MA":"Maranhão","MT":"Mato Grosso",
    "MS":"Mato Grosso do Sul","MG":"Minas Gerais","PA":"Pará","PB":"Paraíba","PR":"Paraná",
    "PE":"Pernambuco","PI":"Piauí","RJ":"Rio de Janeiro","RN":"Rio Grande do Norte",
    "RS":"Rio Grande do Sul","RO":"Rondônia","RR":"Roraima","SC":"Santa Catarina",
    "SP":"São Paulo","SE":"Sergipe","TO":"Tocantins"
}

def _split_city_state(q: str) -> tuple[str, Optional[str]]:
    # aceita "Palmas, TO" | "Palmas - TO" | "Palmas TO"
    s = q.strip()
    m = re.split(r"\s*[,;-]\s*|\s{2,}", s, maxsplit=1)
    if len(m) == 2:
        city, st = m[0].strip(), m[1].strip()
        return city, st
    return s, None

def _normalize_state(st: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    if not st: return None, None
    up = st.strip().upper()
    if up in UF2STATE:    # UF -> nome
        return up, UF2STATE[up]
    # pode ter vindo o nome do estado já
    for uf, name in UF2STATE.items():
        if up.lower() == name.lower():
            return uf, name
    return None, st

def _geocode_nominatim(q: str, state_name: Optional[str]) -> Optional[dict]:
    import requests as _rq
    base = "https://nominatim.openstreetmap.org/search"
    headers = {"User-Agent": GEOCODE_UA, "Accept": "application/json"}
    # tente variantes com viés BR
    variants = []
    if state_name:
        variants += [f"{q}, {state_name}, Brasil", f"{q}, {state_name}, BR"]
    variants += [f"{q}, Brasil", f"{q}, BR", q]

    for txt in variants:
        try:
            r = _rq.get(
                base,
                params={"q": txt, "format": "jsonv2", "addressdetails": 1, "limit": 5, "countrycodes": "br"},
                headers=headers,
                timeout=HTTP_TIMEOUT,
            )
            r.raise_for_status()
            arr = r.json() or []
            if not arr:
                continue
            # escolha a melhor: se state_name foi pedido, prioriza quem bate o estado
            best = None; best_score = -1
            for it in arr:
                lat = it.get("lat"); lon = it.get("lon")
                if not lat or not lon: 
                    continue
                score = 1
                if state_name:
                    addr = it.get("address") or {}
                    nm = (addr.get("state") or addr.get("region") or "").lower()
                    if state_name.lower() in nm:
                        score += 2
                if score > best_score:
                    best, best_score = it, score
            if best:
                return {
                    "lat": float(best["lat"]),
                    "lon": float(best["lon"]),
                    "display_name": best.get("display_name") or q,
                    "source": "nominatim",
                }
        except Exception:
            # tenta a próxima variante / fallback
            pass
    return None

def _geocode_open_meteo(q: str, state_name: Optional[str]) -> Optional[dict]:
    import requests as _rq
    base = "https://geocoding-api.open-meteo.com/v1/search"
    try:
        r = _rq.get(
            base,
            params={"name": q, "count": 10, "language": "pt", "format": "json", "country": "BR"},
            timeout=HTTP_TIMEOUT,
            headers={"User-Agent": GEOCODE_UA},
        )
        r.raise_for_status()
        js = r.json() or {}
        arr = js.get("results") or []
        if not arr:
            return None
        # prioriza match do estado (admin1)
        best = None; best_score = -1
        for it in arr:
            score = 1
            if state_name and (it.get("admin1") or "").lower() == state_name.lower():
                score += 2
            if score > best_score:
                best, best_score = it, score
        if best:
            disp = f"{best.get('name')}, {best.get('admin1') or ''}, {best.get('country') or 'Brasil'}".strip(", ")
            return {"lat": float(best["latitude"]), "lon": float(best["longitude"]), "display_name": disp, "source": "open-meteo"}
    except Exception:
        pass
    return None

def geocode_city(raw_q: str) -> Optional[dict]:
    if not raw_q or not raw_q.strip():
        return None
    city, st = _split_city_state(raw_q)
    uf, state_name = _normalize_state(st)
    q = city
    # 1) tenta Nominatim com bias Brasil
    res = _geocode_nominatim(q, state_name)
    if res:
        return res
    # 2) fallback Open-Meteo
    res = _geocode_open_meteo(q, state_name)
    if res:
        return res
    # 3) última cartada: tenta city + "Brasil" em Nominatim sem state
    res = _geocode_nominatim(city, None)
    return res
# ---------------------------------------------------------------------------



# Distância geodésica aproximada (km)
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dlmb/2)**2
    return 2 * R * math.asin(math.sqrt(a))


# ===== Helpers extras (INPE e tempo) =====

# Time/UTC helpers (evita utcnow deprecado)
def now_utc() -> dt.datetime:
    return dt.datetime.now(UTC)

def iso_utc(ts: dt.datetime) -> str:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    return ts.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

# Datas para montar nomes de arquivos do INPE
def ymd(d: dt.date) -> str:
    return d.strftime("%Y%m%d")

def ym(d: dt.date) -> str:
    return d.strftime("%Y%m")

# Conversões de coordenadas/raio -> bbox (INPE)
def _deg_per_km_lat() -> float:
    # ~111.32 km por grau de latitude
    return 1.0 / 111.32

def _deg_per_km_lon(lat: float) -> float:
    # ~111.32 * cos(lat) km por grau de longitude
    return 1.0 / (111.32 * max(0.01, math.cos(math.radians(lat))))

def bbox_from_center(lat: float, lon: float, raio_km: float) -> tuple[float, float, float, float]:
    """Retorna (minLon, minLat, maxLon, maxLat)"""
    dlat = raio_km * _deg_per_km_lat()
    dlon = raio_km * _deg_per_km_lon(lat)
    return (lon - dlon, lat - dlat, lon + dlon, lat + dlat)

# Parsing numérico tolerante (csv INPE pode vir com vírgula)
def parse_float_safe(x) -> float | None:
    try:
        return float(str(x).replace(",", "."))
    except Exception:
        return None

# ==== [cell] =============================================
## 2) Coletores HTTP (Open-Meteo, OpenAQ v3 + sensors, ANA/INPE CSV)
# @title Coletores HTTP
import os, time, requests, datetime as dt
import pandas as pd
import io, json
from typing import Iterable, Tuple, Optional, Dict, Any
from requests.adapters import HTTPAdapter
try:
    from urllib3.util.retry import Retry
except Exception:
    Retry = None

HTTP_TIMEOUT = globals().get("HTTP_TIMEOUT", 30)

UTC = dt.timezone.utc
def now_utc():
    return dt.datetime.now(UTC)

def iso_utc(ts: dt.datetime) -> str:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    return ts.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

def make_retrying_session(total=3, backoff=0.5, status=(408, 429, 500, 502, 503, 504)):
    s = requests.Session()
    if Retry is not None:
        r = Retry(
            total=total, read=total, connect=total,
            backoff_factor=backoff, status_forcelist=status,
            allowed_methods=frozenset({"GET", "HEAD", "OPTIONS"}),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=r, pool_connections=20, pool_maxsize=50)
        s.mount("http://", adapter); s.mount("https://", adapter)
    return s

_HTTP = make_retrying_session()

def http_get(url, *, params=None, headers=None, timeout=HTTP_TIMEOUT):
    to = timeout if isinstance(timeout, (int, float, tuple)) else HTTP_TIMEOUT
    if isinstance(to, (int, float)):
        to = (min(5, to), to)
    resp = _HTTP.get(url, params=params, headers=headers, timeout=to)
    resp.raise_for_status()
    return resp

# ---------------------------
# Open-Meteo
# ---------------------------
@ttl_cache(ttl_seconds=CACHE_TTL_SEC)
def get_open_meteo(lat: float, lon: float, timeout: int = HTTP_TIMEOUT) -> dict:
    """
    Busca:
      - current: temperatura/umidade/vento (instantâneo)
      - hourly: precipitation (para somar últimas 24h)
      - daily: precipitation_sum (fallback)
    Tudo em UTC para evitar ambiguidade; normalização cuida do 24h.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "timezone": "UTC",
        # instantâneo (continua útil para outros KPIs):
        "current": "temperature_2m,relative_humidity_2m,wind_speed_10m",
        # série horária para acumulado 24h:
        "hourly": "precipitation",
        # série diária como fallback rápido:
        "daily": "precipitation_sum",
        # traz passado recente + hoje (garante termos as últimas 24h)
        "past_days": 2,
        "forecast_days": 1,
    }
    r = http_get(url, params=params, timeout=timeout)
    return r.json()


# ---------------------------
# OpenAQ v3
# ---------------------------
def openaq_headers():
    key = os.getenv("OPENAQ_API_KEY", "") or globals().get("OPENAQ_API_KEY", "")
    if not key:
        raise RuntimeError("Defina OPENAQ_API_KEY com a chave da API v3 da OpenAQ.")
    return {"X-API-Key": key}

@ttl_cache(ttl_seconds=CACHE_TTL_SEC)
def openaq_locations(lat: float, lon: float, radius_m: int = 10000, limit: int = 5, timeout: int = HTTP_TIMEOUT) -> dict:
    radius_m = max(1000, min(int(radius_m), 25000))
    url = "https://api.openaq.org/v3/locations"
    params = {"coordinates": f"{lat:.6f},{lon:.6f}", "radius": radius_m, "limit": limit}
    try:
        r = http_get(url, params=params, headers=openaq_headers(), timeout=timeout)
        out = r.json(); out["_used_radius"] = radius_m; return out
    except requests.HTTPError as e:
        status = getattr(e.response, "status_code", None)
        if status in (408, 429, 500, 502, 503, 504) or isinstance(e, requests.Timeout):
            r = http_get(url, params={"coordinates": f"{lat:.6f},{lon:.6f}", "radius": 25000, "limit": limit},
                         headers=openaq_headers(), timeout=timeout)
            out = r.json(); out["_used_radius"] = 25000; return out
        raise
    except (requests.Timeout, requests.ConnectionError):
        time.sleep(0.6)
        r = http_get(url, params=params, headers=openaq_headers(), timeout=timeout + 5)
        out = r.json(); out["_used_radius"] = radius_m; return out

@ttl_cache(ttl_seconds=CACHE_TTL_SEC)
def openaq_latest_by_location_id(location_id: int, timeout: int = HTTP_TIMEOUT) -> dict:
    url = f"https://api.openaq.org/v3/locations/{location_id}/latest"
    r = http_get(url, headers=openaq_headers(), timeout=timeout)
    return r.json()

@ttl_cache(ttl_seconds=CACHE_TTL_SEC)
def openaq_sensors_by_location_id(location_id: int, timeout: int = HTTP_TIMEOUT) -> dict:
    url = f"https://api.openaq.org/v3/locations/{location_id}/sensors"
    r = http_get(url, headers=openaq_headers(), timeout=timeout)
    return r.json()

@ttl_cache(ttl_seconds=CACHE_TTL_SEC)
def openaq_measurements_by_location_id(location_id: int, limit: int = 100, recency_days: int = 3, timeout: int = HTTP_TIMEOUT) -> dict:
    url = "https://api.openaq.org/v3/measurements"
    date_to = iso_utc(now_utc())
    date_from = iso_utc(now_utc() - dt.timedelta(days=max(1, int(recency_days))))
    params = {
        "locations_id": location_id,
        "parameters_id": "2,5",
        "date_from": date_from, "date_to": date_to,
        "limit": max(10, min(limit, 200)), "sort": "desc"
    }
    r = http_get(url, params=params, headers=openaq_headers(), timeout=timeout)
    return r.json()

@ttl_cache(ttl_seconds=CACHE_TTL_SEC)
def openaq_measurements_by_sensor_id(sensor_id: int, limit: int = 100, recency_days: int = 3, timeout: int = HTTP_TIMEOUT) -> dict:
    url = f"https://api.openaq.org/v3/sensors/{sensor_id}/measurements"
    date_to = iso_utc(now_utc())
    date_from = iso_utc(now_utc() - dt.timedelta(days=max(1, int(recency_days))))
    params = {"date_from": date_from, "date_to": date_to, "limit": max(10, min(limit, 200)), "sort": "desc"}
    r = http_get(url, params=params, headers=openaq_headers(), timeout=timeout)
    return r.json()

# ---------------------------
# ANA (stub)
# ---------------------------
@ttl_cache(ttl_seconds=CACHE_TTL_SEC)
def ana_estacoes_stub(cidade: str, raio_km: int = 150, limit: int = 5) -> dict:
    info = geocode_city(cidade) if cidade else None
    return {"query": {"cidade": cidade, "raio_km": raio_km, "limit": limit}, "center": info or {}, "estacoes": []}

# ---------------------------
# INPE (Queimadas) - CSV (diário/mensal)
# ---------------------------

# Helpers de datas e base, vindos da Célula 0/1:
INPE_CSV_BASE = globals().get("INPE_CSV_BASE", "https://dataserver-coids.inpe.br/queimadas/queimadas/focos/csv")
INPE_DEFAULT_SCOPE = globals().get("INPE_DEFAULT_SCOPE", "diario")
INPE_DEFAULT_REGION = globals().get("INPE_DEFAULT_REGION", "Brasil")

def ymd(d: dt.date) -> str:
    return d.strftime("%Y%m%d")

def ym(d: dt.date) -> str:
    return d.strftime("%Y%m")

def _csv_url_diario(region: str, d: dt.date) -> str:
    reg = (region or "Brasil").strip()
    return f"{INPE_CSV_BASE}/diario/{reg}/focos_diario_{'br' if reg.lower()=='brasil' else 'as'}_{ymd(d)}.csv"

def _csv_url_mensal(region: str, d: dt.date) -> str:
    reg = (region or "Brasil").strip()
    return f"{INPE_CSV_BASE}/mensal/{reg}/focos_mensal_{'br' if reg.lower()=='brasil' else 'as'}_{ym(d)}.csv"

def _try_read_csv(url: str, timeout: int = HTTP_TIMEOUT) -> pd.DataFrame:
    r = http_get(url, timeout=timeout)
    # arquivos podem ser ISO-8859-1; vamos tentar utf-8 e fallback
    try:
        df = pd.read_csv(io.BytesIO(r.content), encoding="utf-8")
    except Exception:
        df = pd.read_csv(io.BytesIO(r.content), encoding="latin1")
    return df

@ttl_cache(ttl_seconds=CACHE_TTL_SEC)
def inpe_fetch_csv(scope: str | None = None, region: str | None = None,
                   ref_date: Optional[dt.date] = None, timeout: int = HTTP_TIMEOUT) -> Dict[str, Any]:
    """
    Baixa um CSV do INPE (diário ou mensal), com fallback para dia/mês anterior se 404.
    Retorna dict: {"df": DataFrame, "url": str, "scope": "diario|mensal", "region": "Brasil|America_Sul", "ref": "YYYYMMDD|YYYYMM"}
    """
    sc = (scope or INPE_DEFAULT_SCOPE or "diario").lower().strip()
    reg = (region or INPE_DEFAULT_REGION or "Brasil").strip()
    today = now_utc().date()
    d = ref_date or today

    tried = []
    if sc == "diario":
        # tenta hoje, depois ontem
        for delta in (0, 1, 2):  # tenta até D-2
            dd = d - dt.timedelta(days=delta)
            url = _csv_url_diario(reg, dd)
            try:
                df = _try_read_csv(url, timeout=timeout)
                return {"df": df, "url": url, "scope": "diario", "region": reg, "ref": ymd(dd)}
            except requests.HTTPError as e:
                if e.response is not None and e.response.status_code == 404:
                    tried.append(url); continue
                raise
    else:
        # mensal: tenta mês corrente, depois mês anterior
        for delta_m in (0, 1, 2):
            mm = (d.replace(day=1) - dt.timedelta(days=delta_m * 30)).replace(day=1)
            url = _csv_url_mensal(reg, mm)
            try:
                df = _try_read_csv(url, timeout=timeout)
                return {"df": df, "url": url, "scope": "mensal", "region": reg, "ref": ym(mm)}
            except requests.HTTPError as e:
                if e.response is not None and e.response.status_code == 404:
                    tried.append(url); continue
                raise

    raise requests.HTTPError(f"INPE CSV não encontrado; tentativas: {tried}")

def _canonical_inpe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Padroniza nomes mais comuns do CSV do INPE em colunas canônicas.
    Mantém colunas originais em 'raw_*' quando necessário.
    """
    cols = {c.lower(): c for c in df.columns}
    def pick(*names):
        for n in names:
            if n in cols: return cols[n]
        return None

    c_lat = pick("lat", "latitude", "y")
    c_lon = pick("lon", "longitude", "x")
    c_dt  = pick("datahora", "data_hora", "data")
    c_sat = pick("satelite", "satellite", "sat")
    c_bio = pick("bioma", "biome")
    c_uf  = pick("estado", "uf", "estado_sigla")
    c_mun = pick("municipio", "nm_municipio", "cidade")
    c_frp = pick("frp", "power", "radiative_power")
    c_risk = pick("risco_fogo", "risco", "risk")

    out = pd.DataFrame()
    out["latitude"]  = df[c_lat] if c_lat else None
    out["longitude"] = df[c_lon] if c_lon else None
    out["datahora"]  = df[c_dt]  if c_dt  else None
    out["satelite"]  = df[c_sat] if c_sat else None
    out["bioma"]     = df[c_bio] if c_bio else None
    out["uf"]        = df[c_uf]  if c_uf  else None
    out["municipio"] = df[c_mun] if c_mun else None
    out["frp"]       = df[c_frp] if c_frp else None
    out["risco_fogo"]= df[c_risk] if c_risk else None
    return out

@ttl_cache(ttl_seconds=CACHE_TTL_SEC)
def inpe_focos_near_old(lat: float, lon: float, raio_km: int = 150, *,
                    scope: str | None = None, region: str | None = None,
                    limit: int = 1000, timeout: int = HTTP_TIMEOUT) -> Dict[str, Any]:
    """
    Baixa o CSV (diário/mensal), normaliza e filtra por raio a partir de (lat,lon).
    Retorna dict com 'features' (lista) e 'meta'.
    """
    payload = inpe_fetch_csv(scope=scope, region=region, timeout=timeout)
    df = payload["df"]
    norm = _canonical_inpe_columns(df).copy()

    # parsing numérico tolerante
    norm["latitude"]  = norm["latitude"].map(parse_float_safe)
    norm["longitude"] = norm["longitude"].map(parse_float_safe)

    # Filtro por bbox para acelerar; depois checamos distância (opcional)
    minx, miny, maxx, maxy = bbox_from_center(lat, lon, float(raio_km))
    bbox_mask = (
        (norm["longitude"].notna()) & (norm["latitude"].notna()) &
        (norm["longitude"] >= minx) & (norm["longitude"] <= maxx) &
        (norm["latitude"]  >= miny) & (norm["latitude"]  <= maxy)
    )
    sub = norm[bbox_mask].copy()

    # (opcional) filtro fino por haversine — se curva de desempenho ficar pesada, remova
    def _hv(row):
        return haversine_km(lat, lon, row["latitude"], row["longitude"])
    try:
        sub["dist_km"] = sub.apply(_hv, axis=1)
        sub = sub[sub["dist_km"] <= float(raio_km)]
    except Exception:
        pass

    # limita quantidade
    if limit:
        sub = sub.head(int(limit))

    # monta saída
    focos = []
    for _, r in sub.iterrows():
        focos.append({
            "latitude": r.get("latitude"),
            "longitude": r.get("longitude"),
            "datahora": r.get("datahora"),
            "satelite": r.get("satelite"),
            "bioma": r.get("bioma"),
            "uf": r.get("uf"),
            "municipio": r.get("municipio"),
            "frp": r.get("frp"),
            "risco_fogo": r.get("risco_fogo"),
            "dist_km": r.get("dist_km"),
        })

    meta = {
        "source": "inpe_csv",
        "scope": payload["scope"],
        "region": payload["region"],
        "reference": payload["ref"],  # YYYYMMDD (diario) ou YYYYMM (mensal)
        "url": payload["url"],
        "bbox": {"minlon": minx, "minlat": miny, "maxlon": maxx, "maxlat": maxy},
        "count": len(focos),
        "observed_at_utc": iso_utc(now_utc()),
    }
    return {"features": {"focos": focos, "count": len(focos), "meta": meta}, "payload": {"csv_url": payload["url"]}}

    ## Coletor + Normalizador OpenWeatherMap (OWM)

def _ms_to_kmh(x: float | None) -> float | None:
    if x is None:
        return None
    try:
        return float(x) * 3.6
    except Exception:
        return None

@ttl_cache(ttl_seconds=CACHE_TTL_SEC)
def get_openweather(lat: float, lon: float, timeout: int = HTTP_TIMEOUT) -> dict:
    """
    Coleta clima atual do OpenWeatherMap (endpoint 'weather').
    Retorna o JSON bruto do OWM.
    """
    if not OPENWEATHER_API_KEY:
        return {}
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "lat": lat,
        "lon": lon,
        "appid": OPENWEATHER_API_KEY,
        "units": "metric",   # °C
        "lang": "pt_br",
    }
    r = http_get(url, params=params, timeout=timeout)
    return r.json()

def normalize_openweather(data: dict) -> dict:
    """
    Converte o payload do OWM para o mesmo formato usado pelo Open-Meteo:
      temperature_2m, relative_humidity_2m, precipitation, wind_speed_10m, observed_at, meta.units
    """
    main = (data or {}).get("main") or {}
    wind = (data or {}).get("wind") or {}
    rain = (data or {}).get("rain") or {}
    # mm de chuva: prioriza 1h, depois 3h / 3
    precip_1h = rain.get("1h")
    precip_3h = rain.get("3h")
    if isinstance(precip_1h, (int, float)):
        precip = float(precip_1h)
    elif isinstance(precip_3h, (int, float)):
        precip = float(precip_3h) / 3.0
    else:
        precip = 0.0

    # timestamp (UTC)
    try:
        ts = dt.datetime.utcfromtimestamp(int((data or {}).get("dt", 0))).replace(tzinfo=dt.timezone.utc)
        observed_at = ts.strftime("%Y-%m-%dT%H:%M")
    except Exception:
        observed_at = None

    out = {
        "temperature_2m": main.get("temp"),
        "relative_humidity_2m": main.get("humidity"),
        "precipitation": precip,
        "wind_speed_10m": _ms_to_kmh(wind.get("speed")),  # OWM fornece m/s
        "observed_at": observed_at,
        "meta": {"units": {
            "temperature_2m": "°C",
            "relative_humidity_2m": "%",
            "precipitation": "mm",
            "wind_speed_10m": "km/h",
        }}
    }
    # saneamento anti-NaN/inf
    for k in ("temperature_2m", "relative_humidity_2m", "precipitation", "wind_speed_10m"):
        v = out.get(k)
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            out[k] = None
    return out


# ==== [cell] =============================================
# @title HOTFIX INPE: sanitizar NaN/Inf no JSON
import math
import pandas as pd

def _json_safe(v):
    """Converte NaN/Inf/NA para None, preserva tipos simples."""
    try:
        # Pandas & numpy NA
        if pd.isna(v):
            return None
    except Exception:
        pass
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return None
    return v

def inpe_focos_near(lat: float, lon: float, raio_km: int = 150, *,
                    scope: str | None = None, region: str | None = None,
                    limit: int = 1000, timeout: int = HTTP_TIMEOUT):
    """
    Baixa o CSV (diário/mensal), normaliza e filtra por raio a partir de (lat,lon).
    Retorna dict com 'features' (lista) e 'meta'. (Hotfix com saneamento de NaN)
    """
    payload = inpe_fetch_csv(scope=scope, region=region, timeout=timeout)
    df = payload["df"]

    # Normalização básica (mesma da célula 2)
    cols = {c.lower(): c for c in df.columns}
    def pick(*names):
        for n in names:
            if n in cols: return cols[n]
        return None

    c_lat = pick("lat", "latitude", "y")
    c_lon = pick("lon", "longitude", "x")
    c_dt  = pick("datahora", "data_hora", "data")
    c_sat = pick("satelite", "satellite", "sat")
    c_bio = pick("bioma", "biome")
    c_uf  = pick("estado", "uf", "estado_sigla")
    c_mun = pick("municipio", "nm_municipio", "cidade")
    c_frp = pick("frp", "power", "radiative_power")
    c_risk = pick("risco_fogo", "risco", "risk")

    norm = pd.DataFrame()
    norm["latitude"]  = df[c_lat] if c_lat else None
    norm["longitude"] = df[c_lon] if c_lon else None
    norm["datahora"]  = df[c_dt]  if c_dt  else None
    norm["satelite"]  = df[c_sat] if c_sat else None
    norm["bioma"]     = df[c_bio] if c_bio else None
    norm["uf"]        = df[c_uf]  if c_uf  else None
    norm["municipio"] = df[c_mun] if c_mun else None
    norm["frp"]       = df[c_frp] if c_frp else None
    norm["risco_fogo"]= df[c_risk] if c_risk else None

    # Parsing numérico tolerante
    norm["latitude"]  = norm["latitude"].map(parse_float_safe)
    norm["longitude"] = norm["longitude"].map(parse_float_safe)

    # Filtro por bbox
    minx, miny, maxx, maxy = bbox_from_center(lat, lon, float(raio_km))
    bbox_mask = (
        (norm["longitude"].notna()) & (norm["latitude"].notna()) &
        (norm["longitude"] >= minx) & (norm["longitude"] <= maxx) &
        (norm["latitude"]  >= miny) & (norm["latitude"]  <= maxy)
    )
    sub = norm[bbox_mask].copy()

    # Distância (pode gerar NaN, depois saneamos)
    try:
        sub["dist_km"] = sub.apply(lambda r: haversine_km(lat, lon, r["latitude"], r["longitude"]), axis=1)
        sub = sub[sub["dist_km"] <= float(raio_km)]
    except Exception:
        pass

    # Limita quantidade
    if limit:
        sub = sub.head(int(limit))

    focos = []
    for _, r in sub.iterrows():
        focos.append({
            "latitude":   _json_safe(r.get("latitude")),
            "longitude":  _json_safe(r.get("longitude")),
            "datahora":   _json_safe(r.get("datahora")),
            "satelite":   _json_safe(r.get("satelite")),
            "bioma":      _json_safe(r.get("bioma")),
            "uf":         _json_safe(r.get("uf")),
            "municipio":  _json_safe(r.get("municipio")),
            "frp":        _json_safe(r.get("frp")),
            "risco_fogo": _json_safe(r.get("risco_fogo")),
            "dist_km":    _json_safe(r.get("dist_km")),
        })

    meta = {
        "source": "inpe_csv",
        "scope": payload["scope"],
        "region": payload["region"],
        "reference": payload["ref"],
        "url": payload["url"],
        "bbox": {"minlon": minx, "minlat": miny, "maxlon": maxx, "maxlat": maxy},
        "count": len(focos),
        "observed_at_utc": iso_utc(now_utc()),
    }
    return {"features": {"focos": focos, "count": len(focos), "meta": meta},
            "payload": {"csv_url": payload["url"]}}

# ==== [cell] =============================================
## 3) Normalização e Score por limiares
# @title Normalizadores e Score
def _param_name_to_str(param_obj) -> str:
    if param_obj is None:
        return ""
    if isinstance(param_obj, str):
        return param_obj.strip().lower()
    if isinstance(param_obj, dict):
        for k in ("id", "code", "name", "parameter", "slug"):
            v = param_obj.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip().lower()
    if isinstance(param_obj, list) and param_obj:
        return _param_name_to_str(param_obj[0])
    return ""

def _canonical_unit(unit: str | None, param: str | None) -> str | None:
    if not unit:
        if (param or "").lower() in {"t", "temp", "temperature"}:
            return "°C"
        return None
    u = unit.strip().lower()
    if u in {"c", "°c", "degc"}: return "°C"
    if u in {"percent", "%"}: return "%"
    if u in {"ug/m3", "µg/m3", "μg/m3"}: return "µg/m³"
    return unit

# Helper p/ parse ISO em UTC
def _parse_iso_utc(s: str):
    if not s:
        return None
    try:
        if s.endswith("Z"):
            return dt.datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(dt.timezone.utc)
        return dt.datetime.fromisoformat(s).astimezone(dt.timezone.utc)
    except Exception:
        return None

def normalize_open_meteo(resp: dict) -> dict:
    """
    Normaliza Open-Meteo:
      - temperature_2m, relative_humidity_2m, wind_speed_10m (instantâneo)
      - precipitation  = ACUMULADO 24h (mm)  ← usado no KPI/score
      - observed_at    = timestamp do 'current'
      - meta.units     = unidades combinadas (current/hourly/daily)
    Requer que get_open_meteo peça 'hourly=precipitation' e 'daily=precipitation_sum'.
    """
    if not isinstance(resp, dict):
        return {"meta": {"units": {}}}

    out = {"meta": {"units": {}}}

    # --- current (instantâneo)
    cur = (resp.get("current") or {})
    out["temperature_2m"] = cur.get("temperature_2m")
    out["relative_humidity_2m"] = cur.get("relative_humidity_2m")
    out["wind_speed_10m"] = cur.get("wind_speed_10m")
    out["observed_at"] = cur.get("time")

    # --- units (combina as disponibilizadas)
    units = {}
    for k in ("current_units", "hourly_units", "daily_units"):
        u = resp.get(k) or {}
        if isinstance(u, dict):
            units.update(u)
    out["meta"]["units"] = units

    # --- acumulado 24h preferindo série horária
    precip_24h = None
    hourly = resp.get("hourly") or {}
    hrs_time = hourly.get("time") or []
    hrs_prec = hourly.get("precipitation") or []

    if hrs_time and hrs_prec and len(hrs_time) == len(hrs_prec):
        now_ts = _parse_iso_utc(out["observed_at"]) or _parse_iso_utc(hrs_time[-1]) or dt.datetime.now(dt.timezone.utc)
        start = now_ts - dt.timedelta(hours=24)

        total = 0.0
        for t_str, val in zip(hrs_time, hrs_prec):
            ts = _parse_iso_utc(t_str)
            if ts is None:
                continue
            if start < ts <= now_ts:
                try:
                    v = float(val)
                except Exception:
                    v = 0.0
                if v > 0:
                    total += v
        precip_24h = round(total, 2)

    # --- fallback diário (pega o dia mais recente)
    if precip_24h is None:
        daily = resp.get("daily") or {}
        d_sum = daily.get("precipitation_sum") or []
        if d_sum:
            try:
                precip_24h = round(float(d_sum[-1]), 2)
            except Exception:
                precip_24h = 0.0

    # Se nada veio, deixa 0.0 (front trata)
    out["precipitation"] = precip_24h if precip_24h is not None else 0.0
    return out

def normalize_openaq_v3_latest(latest: dict, sensors: dict | None = None) -> dict:
    vals = {"pm25": [], "pm10": []}
    last_local = None; last_utc = None; units = {}
    results = (latest or {}).get("results") or []
    for it in results:
        for m in it.get("measurements", []) or []:
            param = _param_name_to_str(m.get("parameter")); val = m.get("value")
            if param in {"pm25", "pm2.5"} and isinstance(val, (int, float)):
                vals["pm25"].append(float(val)); units["pm25"] = _canonical_unit(m.get("unit"), "pm25")
            if param in {"pm10"} and isinstance(val, (int, float)):
                vals["pm10"].append(float(val)); units["pm10"] = _canonical_unit(m.get("unit"), "pm10")
            last_local = m.get("lastUpdated") or last_local; last_utc = m.get("lastUpdatedUtc") or last_utc
        for s in it.get("sensors", []) or []:
            param = _param_name_to_str((s.get("parameter") or {}))
            lv = s.get("lastValue") or {}; val = lv.get("value")
            unit = lv.get("unit") or (s.get("parameter") or {}).get("units")
            if isinstance(val, (int, float)):
                if param in {"pm25", "pm2.5"}:
                    vals["pm25"].append(float(val)); units["pm25"] = _canonical_unit(unit, "pm25") or units.get("pm25")
                if param == "pm10":
                    vals["pm10"].append(float(val)); units["pm10"] = _canonical_unit(unit, "pm10") or units.get("pm10")
            last_local = lv.get("lastUpdated") or last_local; last_utc = lv.get("lastUpdatedUtc") or last_utc
    out = {"pm25": None, "pm10": None, "meta": {"units": units}}
    if vals["pm25"]: out["pm25"] = sum(vals["pm25"]) / len(vals["pm25"])
    if vals["pm10"]: out["pm10"] = sum(vals["pm10"]) / len(vals["pm10"])
    out["meta"]["last_local"] = last_local; out["meta"]["last_utc"] = last_utc
    return out

def score_risk(weather: dict, air: dict) -> tuple[int, str]:
    pm25 = (air or {}).get("pm25") or 0
    pm10 = (air or {}).get("pm10") or 0
    precip = (weather or {}).get("precipitation") or 0
    score = 0
    if pm25 >= PM25_LIMIT: score += WEIGHT_PM25
    if pm10 >= PM10_LIMIT: score += WEIGHT_PM10
    if precip <= PRECIP_GOOD_MM: score += WEIGHT_DRY
    score = max(0, min(100, int(score)))
    if score >= THRESHOLD_RED: level = "Vermelho"
    elif score >= THRESHOLD_YELLOW: level = "Amarelo"
    else: level = "Verde"
    return score, level

# -------------------------------
# Normalizador INPE (CSV focos)
# -------------------------------
def normalize_inpe_csv(df, *, center_lat: float | None = None, center_lon: float | None = None,
                       raio_km: float | None = None, limit: int | None = None,
                       meta_source: str = "inpe_csv", scope: str | None = None,
                       region: str | None = None, reference: str | None = None,
                       csv_url: str | None = None) -> dict:
    """
    Normaliza um DataFrame do INPE (CSV) para o formato padrão:
    { "features": { "focos": [...], "count": N, "meta": {...} }, "payload": {...} }
    Se center_lat/lon e raio_km forem passados, aplica filtro espacial (bbox + haversine).
    """
    import pandas as pd
    if not isinstance(df, pd.DataFrame):
        raise ValueError("normalize_inpe_csv: 'df' precisa ser um pandas.DataFrame")

    # Mapear colunas comuns
    cols = {c.lower(): c for c in df.columns}
    def pick(*names):
        for n in names:
            if n in cols: return cols[n]
        return None

    c_lat = pick("lat", "latitude", "y")
    c_lon = pick("lon", "longitude", "x")
    c_dt  = pick("datahora", "data_hora", "data")
    c_sat = pick("satelite", "satellite", "sat")
    c_bio = pick("bioma", "biome")
    c_uf  = pick("estado", "uf", "estado_sigla")
    c_mun = pick("municipio", "nm_municipio", "cidade")
    c_frp = pick("frp", "power", "radiative_power")
    c_risk = pick("risco_fogo", "risco", "risk")

    norm = pd.DataFrame()
    norm["latitude"]  = df[c_lat] if c_lat else None
    norm["longitude"] = df[c_lon] if c_lon else None
    norm["datahora"]  = df[c_dt]  if c_dt  else None
    norm["satelite"]  = df[c_sat] if c_sat else None
    norm["bioma"]     = df[c_bio] if c_bio else None
    norm["uf"]        = df[c_uf]  if c_uf  else None
    norm["municipio"] = df[c_mun] if c_mun else None
    norm["frp"]       = df[c_frp] if c_frp else None
    norm["risco_fogo"]= df[c_risk] if c_risk else None

    # Parsing numérico e filtro espacial (se solicitado)
    norm["latitude"]  = norm["latitude"].map(parse_float_safe)
    norm["longitude"] = norm["longitude"].map(parse_float_safe)

    bbox_used = None
    if center_lat is not None and center_lon is not None and raio_km is not None:
        minx, miny, maxx, maxy = bbox_from_center(float(center_lat), float(center_lon), float(raio_km))
        bbox_used = {"minlon": minx, "minlat": miny, "maxlon": maxx, "maxlat": maxy}
        bbox_mask = (
            (norm["longitude"].notna()) & (norm["latitude"].notna()) &
            (norm["longitude"] >= minx) & (norm["longitude"] <= maxx) &
            (norm["latitude"]  >= miny) & (norm["latitude"]  <= maxy)
        )
        sub = norm[bbox_mask].copy()
        # filtro fino opcional por haversine
        try:
            sub["dist_km"] = sub.apply(lambda r: haversine_km(center_lat, center_lon, r["latitude"], r["longitude"]), axis=1)
            sub = sub[sub["dist_km"] <= float(raio_km)]
        except Exception:
            pass
    else:
        sub = norm.copy()

    if limit:
        sub = sub.head(int(limit))

    focos = []
    for _, r in sub.iterrows():
        focos.append({
            "latitude": r.get("latitude"),
            "longitude": r.get("longitude"),
            "datahora": r.get("datahora"),
            "satelite": r.get("satelite"),
            "bioma": r.get("bioma"),
            "uf": r.get("uf"),
            "municipio": r.get("municipio"),
            "frp": r.get("frp"),
            "risco_fogo": r.get("risco_fogo"),
            "dist_km": r.get("dist_km"),
        })

    meta = {
        "source": meta_source,
        "scope": scope,
        "region": region,
        "reference": reference,
        "url": csv_url,
        "bbox": bbox_used,
        "count": len(focos),
        "observed_at_utc": iso_utc(now_utc()),
    }
    return {"features": {"focos": focos, "count": len(focos), "meta": meta}}

# Azulejo: conversor de payload produzido pelos coletores (se quiser reaproveitar)
def normalize_inpe_payload(payload: dict, *, center_lat: float | None = None, center_lon: float | None = None,
                           raio_km: float | None = None, limit: int | None = None) -> dict:
    """
    Aceita o payload retornado por inpe_fetch_csv (com 'df', 'scope', 'region', 'ref', 'url')
    e retorna o JSON normalizado padrão.
    """
    df = (payload or {}).get("df")
    return normalize_inpe_csv(
        df,
        center_lat=center_lat, center_lon=center_lon, raio_km=raio_km, limit=limit,
        meta_source="inpe_csv",
        scope=payload.get("scope"), region=payload.get("region"),
        reference=payload.get("ref"), csv_url=payload.get("url"),
    )

# ==== [cell] =============================================
## 4) Banco de Dados (SQLModel + SQLite)
# @title DB: engine e modelos (SQLModel + SQLite) — sem helpers aqui
from typing import Optional
from sqlmodel import SQLModel, Field, create_engine
import datetime as dt

DB_URL = DB_URL  # reuse global DB_URL
# engine: reuse global engine

UTC = dt.timezone.utc
def _now_utc():
    return dt.datetime.now(UTC)

# -------------------------
# Modelos (com extend_existing p/ reexecução em notebook)
# -------------------------
class WeatherObs(SQLModel, table=True):
    __tablename__ = "weatherobs"
    __table_args__ = {"extend_existing": True}

    id: Optional[int] = Field(default=None, primary_key=True)
    lat: float
    lon: float
    temperature_2m: Optional[float] = None
    relative_humidity_2m: Optional[float] = None
    precipitation: Optional[float] = None
    wind_speed_10m: Optional[float] = None
    observed_at: dt.datetime = Field(default_factory=_now_utc)
    raw_json: Optional[str] = None  # snapshot do provedor

class AirObs(SQLModel, table=True):
    __tablename__ = "airobs"
    __table_args__ = {"extend_existing": True}

    id: Optional[int] = Field(default=None, primary_key=True)
    lat: float
    lon: float
    fonte: str = "openaq"
    location_id: Optional[int] = None
    pm25: Optional[float] = None
    pm10: Optional[float] = None
    params_json: Optional[str] = None
    units_json: Optional[str] = None
    observed_at: dt.datetime = Field(default_factory=_now_utc)
    raw_json: Optional[str] = None

class AlertObs(SQLModel, table=True):
    __tablename__ = "alertobs"
    __table_args__ = {"extend_existing": True}

    id: Optional[int] = Field(default=None, primary_key=True)
    lat: float
    lon: float
    tipo: str
    payload: Optional[str] = None
    observed_at: dt.datetime = Field(default_factory=_now_utc)

class FireObs(SQLModel, table=True):
    __tablename__ = "fireobs"
    __table_args__ = {"extend_existing": True}

    id: Optional[int] = Field(default=None, primary_key=True)
    lat: float
    lon: float
    fonte: str = "inpe_csv"  # padrão: INPE via CSV (dataserver COIDS)
    payload: Optional[str] = None      # JSON com subset/URL do CSV
    observed_at: dt.datetime = Field(default_factory=_now_utc)

# Cria as tabelas (idempotente)
print("DB pronto: tabelas criadas/atualizadas em", DB_URL)

# ==== [cell] =============================================
## 5) Helpers de persistência (save_* / get_last)
# @title Helpers de persistência (SQLModel + alertas/nível)
from typing import Optional, Dict, Any
from sqlmodel import Session, select
import os, json, time

# Usa: engine, WeatherObs, AirObs, FireObs, _now_utc definidos na Célula 4

# ------------------------------------------------------------------------------
# Saves (inserts)
# ------------------------------------------------------------------------------
def save_weather(lat: float, lon: float, features: Dict[str, Any], raw: Dict[str, Any] | None = None) -> int:
    rec = WeatherObs(
        lat=lat, lon=lon,
        temperature_2m=(features or {}).get("temperature_2m"),
        relative_humidity_2m=(features or {}).get("relative_humidity_2m"),
        precipitation=(features or {}).get("precipitation"),
        wind_speed_10m=(features or {}).get("wind_speed_10m"),
        observed_at=_now_utc(),
        raw_json=json.dumps(raw or {}, ensure_ascii=False),
    )
    with Session(engine) as sess:
        sess.add(rec)
        sess.commit()
        sess.refresh(rec)
        return rec.id

def save_air(lat: float, lon: float, fonte: str, location_id: Optional[int],
             features: Dict[str, Any] | None, raw: Dict[str, Any] | None = None) -> int:
    units = (((features or {}).get("meta") or {}).get("units")) or {}
    rec = AirObs(
        lat=lat, lon=lon, fonte=fonte, location_id=location_id,
        pm25=(features or {}).get("pm25"),
        pm10=(features or {}).get("pm10"),
        params_json=None,
        units_json=json.dumps(units, ensure_ascii=False) if units else None,
        observed_at=_now_utc(),
        raw_json=json.dumps(raw or {}, ensure_ascii=False),
    )
    with Session(engine) as sess:
        sess.add(rec)
        sess.commit()
        sess.refresh(rec)
        return rec.id

def save_fire(lat: float, lon: float, fonte: str = "inpe_csv",
              payload: Dict[str, Any] | None = None) -> int:
    rec = FireObs(
        lat=lat, lon=lon, fonte=fonte,
        payload=json.dumps(payload or {}, ensure_ascii=False),
        observed_at=_now_utc(),
    )
    with Session(engine) as sess:
        sess.add(rec)
        sess.commit()
        sess.refresh(rec)
        return rec.id

# ------------------------------------------------------------------------------
# Gets (consultas simples)
# ------------------------------------------------------------------------------
def get_last_weather() -> Optional[WeatherObs]:
    with Session(engine) as sess:
        return sess.exec(select(WeatherObs).order_by(WeatherObs.id.desc()).limit(1)).first()

def get_last_air() -> Optional[AirObs]:
    with Session(engine) as sess:
        return sess.exec(select(AirObs).order_by(AirObs.id.desc()).limit(1)).first()

def get_last_fire() -> Optional[FireObs]:
    with Session(engine) as sess:
        return sess.exec(select(FireObs).order_by(FireObs.id.desc()).limit(1)).first()

# ------------------------------------------------------------------------------
# Alertas: score/nível + eventos de transição + notificação (filesystem NDJSON)
# ------------------------------------------------------------------------------
# Estado global (idempotente; seguro para reexecutar a célula)
try:
    _LAST_LEVEL
except NameError:
    _LAST_LEVEL = {}          # {alert_id: {"level": "...", "ts": "<iso>"}}

try:
    _LAST_NOTIFY
except NameError:
    _LAST_NOTIFY = {}         # debounce: {alert_id: epoch_seconds}

try:
    NOTIFY_DEBOUNCE_SEC
except NameError:
    NOTIFY_DEBOUNCE_SEC = 600  # 10 min

try:
    WEBHOOK_URL
except NameError:
    WEBHOOK_URL = os.environ.get("ALERTS_WEBHOOK_URL")

def _ensure_alerts_dir() -> str:
    d = "/mnt/data/alerts"
    os.makedirs(d, exist_ok=True)
    return d

def save_alert_score(alert_id: str, score: float, level: str,
                     alert_obs: Dict[str, Any], params: Dict[str, Any] | None = None) -> None:
    """
    Persistência simples do score/nível + AlertObs em NDJSON.
    Troque por sua camada real (DB/Tabela) quando quiser.
    """
    try:
        d = _ensure_alerts_dir()
        rec = {
            "ts": _now_utc().isoformat().replace("+00:00", "Z") if hasattr(_now_utc(), "isoformat") else str(_now_utc()),
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

def _persist_level_event(alert_id: str, old_level: str | None, new_level: str, payload: dict) -> None:
    try:
        d = _ensure_alerts_dir()
        ev = {
            "ts": _now_utc().isoformat().replace("+00:00", "Z") if hasattr(_now_utc(), "isoformat") else str(_now_utc()),
            "alert_id": alert_id,
            "from": old_level,
            "to": new_level,
            "payload": payload,
        }
        with open(os.path.join(d, "level_events.ndjson"), "a", encoding="utf-8") as f:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")
    except Exception as e:
        print("[level_event] WARN:", e)

def _notify_level_change(alert_id: str, old_level: str | None, new_level: str, score: float, obs: dict) -> None:
    """
    Notifica mudança de nível (webhook se ALERTS_WEBHOOK_URL; caso contrário, loga em arquivo).
    Respeita debounce por alert_id (NOTIFY_DEBOUNCE_SEC).
    """
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
        "when": _now_utc().isoformat().replace("+00:00", "Z") if hasattr(_now_utc(), "isoformat") else str(_now_utc()),
        "obs": {k: obs.get(k) for k in ("severity", "duration", "frequency", "impact")},
    }

    try:
        if WEBHOOK_URL:
            import requests
            requests.post(WEBHOOK_URL, json=msg, timeout=8)
        else:
            d = _ensure_alerts_dir()
            with open(os.path.join(d, "notifications.ndjson"), "a", encoding="utf-8") as f:
                f.write(json.dumps(msg, ensure_ascii=False) + "\n")
    except Exception as e:
        print("[notify] WARN:", e)

def handle_level_transition(alert_id: str, new_level: str, score: float,
                            alert_obs: Dict[str, Any], extra: Dict[str, Any] | None = None,
                            notify_on_bootstrap: bool = False) -> None:
    """
    Detecta transição de nível; persiste evento e notifica (com debounce).
    """
    old_level = (_LAST_LEVEL.get(alert_id) or {}).get("level")
    is_bootstrap = (old_level is None)

    if new_level != old_level:
        _persist_level_event(alert_id, old_level, new_level, {
            "score": score,
            "obs": alert_obs,
            **(extra or {})
        })
        if (not is_bootstrap) or notify_on_bootstrap:
            _notify_level_change(alert_id, old_level, new_level, score, alert_obs)

        _LAST_LEVEL[alert_id] = {"level": new_level, "ts": _now_utc()}

# ==== Scoring helpers (colem esta célula acima dos endpoints) ====
from dataclasses import dataclass
from typing import Dict, Any, Optional
from datetime import datetime, timezone
import math

THRESHOLDS = {
    "green_lt": 0.33,   # score < 0.33 => VERDE
    "yellow_lt": 0.66,  # 0.33 <= score < 0.66 => AMARELO
    # score >= 0.66 => VERMELHO
}

WEIGHTS = {
    "severity": 0.25,
    "duration": 0.25,
    "frequency": 0.25,
    "impact": 0.15,
    "rainfall": 0.10,  # peso da chuva
}

@dataclass
class ScoreResult:
    score: float
    level: str  # "verde" | "amarelo" | "vermelho"
    breakdown: Dict[str, float]

def _clip01(x: float) -> float:
    try:
        return max(0.0, min(1.0, float(x)))
    except Exception:
        return 0.0

def _classify_level(score: float) -> str:
    if score < THRESHOLDS["green_lt"]:
        return "verde"
    if score < THRESHOLDS["yellow_lt"]:
        return "amarelo"
    return "vermelho"

def _norm_pm(pm_val: Optional[float], pm_kind: str) -> float:
    """Normaliza PM em 0..1 — pm2.5: 0..75; pm10: 0..150 (ajuste se quiser)."""
    if pm_val is None:
        return 0.0
    ref = 75.0 if pm_kind == "pm25" else 150.0
    return _clip01(pm_val / ref)

def _hours_since(iso_str: Optional[str]) -> Optional[float]:
    if not iso_str:
        return None
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        return max(0.0, (now - dt).total_seconds() / 3600.0)
    except Exception:
        return None

# --- função auxiliar: índice de risco de chuva (alagamentos/enxurradas)
def _rainfall_risk_u(mm: Optional[float]) -> float:
    """
    Calcula um índice de risco de chuva entre 0..1
    - 0: sem chuva
    - 1: chuva muito intensa (>50mm)
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

# ✅ alias compatível — compute_alert_score chama aqui
def _rainfall_index(mm: Optional[float]) -> float:
    return _rainfall_risk_u(mm)


def compute_alert_score(alert_obs: Dict[str, Any],
                        weights: Dict[str, float] = WEIGHTS) -> ScoreResult:
    """
    Calcula o score final (0..1) com base em severity/duration/frequency/impact
    e também no índice pluviométrico (rainfall).

    Breakdown retorna todos os componentes individuais + valores brutos.
    """

    # --- 1) Campos principais (clip 0..1 por segurança)
    sev = _clip01(alert_obs.get("severity", 0))
    dur = _clip01(alert_obs.get("duration", 0))
    freq = _clip01(alert_obs.get("frequency", 0))
    imp = _clip01(alert_obs.get("impact", 0))

    # --- 2) Índice pluviométrico
    precip_24h = (
        alert_obs.get("precip_24h")
        or alert_obs.get("precipitation")
        or (alert_obs.get("meta", {}).get("precipitation"))
        or 0.0
    )
    rainfall = _rainfall_index(precip_24h)

    # --- 3) Score final ponderado
    score = (
        sev * weights.get("severity", 0.25) +
        dur * weights.get("duration", 0.25) +
        freq * weights.get("frequency", 0.25) +
        imp * weights.get("impact", 0.15) +
        rainfall * weights.get("rainfall", 0.10)
    )

    # --- 4) Classificação qualitativa
    level = _classify_level(score)

    # --- 5) Retorno estruturado
    return ScoreResult(
        score=round(score, 4),
        level=level,
        breakdown={
            "severity": sev,
            "duration": dur,
            "frequency": freq,
            "impact": imp,
            "precip_index": rainfall,  # índice 0..1
            "precip_24h_mm": float(precip_24h) if precip_24h is not None else None,
        },
    )


# ==== [cell] =============================================
# @title IA leve: detecção robusta de outliers (MAD) para PM2.5/PM10
from sqlmodel import Session, select
import statistics as stats
from typing import Tuple, Optional

def _round_coord(v: float, ndigits: int = 3) -> float:
    # agrupa leituras próximas (≈100m-1km, útil p/ “mesma área”)
    return round(float(v), ndigits)

def _mad(values):
    if not values:
        return None
    med = stats.median(values)
    devs = [abs(x - med) for x in values]
    mad = stats.median(devs)
    # converte MAD para sigma (~1.4826) se quiser comparar com Z-score
    return med, mad

def pm_outlier_flags(
    lat: float, lon: float, pm25: Optional[float], pm10: Optional[float],
    k: float = 5.0, lookback: int = 50
) -> Tuple[bool, bool]:
    """
    Marca outliers usando mediana/MAD das últimas 'lookback' medições
    salvas no DB para a mesma área (lat/lon arredondados).
    Retorna (pm25_outlier, pm10_outlier).
    """
    if pm25 is None and pm10 is None:
        return False, False

    latk, lonk = _round_coord(lat), _round_coord(lon)
    with Session(engine) as s:
        rows = s.exec(
            select(AirObs)
            .where(AirObs.lat.between(latk-0.001, latk+0.001))
            .where(AirObs.lon.between(lonk-0.001, lonk+0.001))
            .order_by(AirObs.observed_at.desc())
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

# ==== [cell] =============================================

from typing import Optional, List, Dict, Any
import requests, pandas as pd
from urllib.parse import urlencode, quote
from datetime import datetime, timezone

from fastapi import FastAPI, Query, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse

# ------------------------------------------------------------------------------
# App + CORS
# ------------------------------------------------------------------------------
app = FastAPI(title="AmazonSafe API (v3) - Colab")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)



@app.on_event("startup")
def _on_startup():
    try:
        # cria/atualiza as tabelas quando a API inicia
        from sqlmodel import SQLModel
        SQLModel.metadata.create_all(engine)
    except Exception as e:
        # não derruba o serviço caso o DB não esteja disponível no boot
        print("DB init error:", e)
        pass


DEFAULT_LAT_f = DEFAULT_LAT
DEFAULT_LON_f = DEFAULT_LON
# ------------------------------------------------------------------------------
# Variáveis globais do serviço (metadados e controle)
# ------------------------------------------------------------------------------
import os, time

APP_NAME = os.getenv("APP_NAME", "AmazonSafe API")
APP_VERSION = os.getenv("APP_VERSION", "v3")
ENV = os.getenv("ENV", "development")

# Marca o tempo em que a aplicação iniciou (para calcular uptime no /health)
STARTED_AT = globals().get("_APP_STARTED_AT") or time.time()
globals()["_APP_STARTED_AT"] = STARTED_AT

# ------------------------------------------------------------------------------
# Health
# ------------------------------------------------------------------------------
@app.get("/health", summary="Healthcheck e estatísticas de cache", tags=["Infra"])
def health():
    """Indica se o serviço está no ar, mostra uptime e retorna resumo do cache."""
    uptime_seconds = round(time.time() - STARTED_AT, 3)
    return {
        "ok": True,
        "status": "healthy",
        "service": APP_NAME,
        "version": APP_VERSION,
        "env": ENV,
        "uptime_seconds": uptime_seconds,
        "cache": cache_stats(),
    }

# ========= DEBUG: schema das tabelas =========
from sqlalchemy import inspect, text

@app.get("/debug/openmeteo_raw")
def debug_openmeteo_raw(lat: float, lon: float):
    raw = get_open_meteo(lat, lon, timeout=HTTP_TIMEOUT)
    norm = normalize_open_meteo(raw)

    hourly = (raw or {}).get("hourly") or {}
    hrs_time = hourly.get("time") or []
    hrs_prec = hourly.get("precipitation") or []
    tail = list(zip(hrs_time[-6:], hrs_prec[-6:]))

    return {
        "observed_at": norm.get("observed_at"),
        "precip_24h_norm": norm.get("precipitation"),
        "hourly_tail": tail,   # últimos 6 pontos (hora, precip)
        "counts": {"hourly_time": len(hrs_time), "hourly_prec": len(hrs_prec)},
        "units": (raw or {}).get("hourly_units") or {},
    }


@app.get("/debug/alertobs_schema", tags=["Infra"])
def debug_alertobs_schema():
    """Retorna os nomes das colunas da tabela 'alertobs' (se existir)."""
    insp = inspect(engine)
    try:
        cols = [col["name"] for col in insp.get_columns("alertobs")]
        return {"table": "alertobs", "columns": cols}
    except Exception as e:
        return {"table": "alertobs", "columns": [], "note": "Tabela não encontrada.", "error": str(e)}

@app.get("/debug/alertas_schema", tags=["Infra"])
def debug_alertas_schema_backward_compat():
    """
    Mantém a rota antiga, mas explica que a tabela correta é 'alertobs'.
    Útil para quem ainda chama /debug/alertas_schema.
    """
    return {
        "note": "Não há tabela 'alertas' no schema atual. Use /debug/alertobs_schema.",
        "tables_hint": ["weatherobs", "airobs", "fireobs", "alertobs"],
    }

@app.get("/debug/tables", tags=["Infra"])
def debug_tables():
    """Lista tabelas existentes no banco."""
    insp = inspect(engine)
    return {"tables": insp.get_table_names()}

@app.get("/debug/schema_raw/{table_name}", tags=["Infra"])
def debug_schema_raw(table_name: str):
    """PRAGMA direto no SQLite para qualquer tabela."""
    try:
        with engine.connect() as conn:
            rows = conn.exec_driver_sql(f"PRAGMA table_info({table_name});").fetchall()
        return {
            "table": table_name,
            "columns": [r[1] for r in rows],
            "detail": [
                {"name": r[1], "type": r[2], "notnull": r[3], "default": r[4], "pk": r[5]}
                for r in rows
            ],
        }
    except Exception as e:
        return {"table": table_name, "columns": [], "detail": [], "error": str(e)}

@app.get("/debug/ndjson_info", tags=["Infra"])
def debug_ndjson_info():
    try:
        size = os.path.getsize(ALERTAS_NDJSON_PATH) if os.path.exists(ALERTAS_NDJSON_PATH) else 0
        lines = 0
        if size:
            with open(ALERTAS_NDJSON_PATH, encoding="utf-8") as fh:
                for _ in fh:
                    lines += 1
        return {"path": ALERTAS_NDJSON_PATH, "exists": os.path.exists(ALERTAS_NDJSON_PATH), "size_bytes": size, "lines": lines}
    except Exception as e:
        return {"error": str(e)}


# ======[API] Último alerta gravado (para o dashboard) --------------------------


def _read_last_ndjson(path: str) -> dict | None:
    """Lê a ÚLTIMA linha válida de um arquivo NDJSON (retorna dict)."""
    if not os.path.exists(path):
        return None
    last = None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                last = json.loads(line)
            except Exception:
                continue
    return last

@app.get("/api/alertas", tags=["Alertas"], summary="Lista alertas (NDJSON) recentes")
def api_alertas(limit: int = 50, alert_id: Optional[str] = None):
    """
    Lê as últimas linhas do arquivo data/alertas.ndjson.
    Use ?limit= e/ou ?alert_id= para filtrar.
    """
    try:
        if not os.path.exists(ALERTAS_NDJSON_PATH):
            return {"ok": True, "has_data": False, "note": "Nenhum alerta persistido ainda."}

        rows: List[Dict] = []
        with open(ALERTAS_NDJSON_PATH, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if alert_id and obj.get("alert_id") != alert_id:
                    continue
                rows.append(obj)

        if not rows:
            return {"ok": True, "has_data": False, "note": "Sem registros para o filtro atual."}

        rows = rows[-max(1, min(limit, 500)):]  # segurança: máx 500
        return {"ok": True, "count": len(rows), "items": rows}
    except Exception as e:
        return {"ok": False, "error": str(e)}



# ------------------------------------------------------------------------------
# Root (/): metadados e links úteis
# ------------------------------------------------------------------------------
from fastapi import Request
from fastapi.responses import JSONResponse

@app.get("/", response_class=JSONResponse, summary="Index e metadados", tags=["Infra"])
def root(request: Request):
    base = str(request.base_url).rstrip("/")
    return {
        "ok": True,
        "service": APP_NAME,
        "version": APP_VERSION,
        "env": ENV,
        "docs": {
            "openapi": f"{base}/openapi.json",
            "swagger": f"{base}/docs",
            "redoc": f"{base}/redoc",
        },
        "endpoints": {
            "health": f"{base}/health",
            "weather": f"{base}/api/weather?lat=-1.4558&lon=-48.5039",
            "air_openaq": f"{base}/api/air/openaq?lat=-1.4558&lon=-48.5039&radius_m=10000",
            "inpe_focos": f"{base}/api/inpe/focos?lat=-1.4558&lon=-48.5039&raio_km=150&scope=diario&region=Brasil&limit=200",
            "data": f"{base}/api/data?cidade=Bel%C3%A9m%2C%20PA&raio_km=150&scope=diario&region=Brasil",
        },
    }


# ------------------------------------------------------------------------------
# Weather (Open-Meteo)
# ------------------------------------------------------------------------------
@app.get(
    "/api/weather",
    summary="Condições atuais de tempo (Open-Meteo)",
    tags=["Weather"],
    openapi_extra={
        "examples": {
            "Belém/PA": {"summary": "Belém/PA (default)", "value": {"lat": -1.4558, "lon": -48.5039}},
            "Manaus/AM": {"summary": "Manaus/AM", "value": {"lat": -3.1316, "lon": -59.9825}},
        }
    },
)
def api_weather(
    lat: float = Query(DEFAULT_LAT_f, description="Latitude (ex.: -1.4558 = Belém/PA)."),
    lon: float = Query(DEFAULT_LON_f, description="Longitude (ex.: -48.5039 = Belém/PA)."),
):
    """
    Retorna observação atual do tempo (Open-Meteo):
    temperature_2m, relative_humidity_2m, precipitation, wind_speed_10m, observed_at.
    """
    try:
        wr = get_open_meteo(lat, lon, timeout=HTTP_TIMEOUT)
        feats = normalize_open_meteo(wr)
        units = (wr or {}).get("current_units") or {}
        observed_at = ((wr or {}).get("current") or {}).get("time")
        feats.setdefault("meta", {}).setdefault("units", units)
        if "observed_at" not in feats and observed_at:
            feats["observed_at"] = observed_at
        try:
            save_weather(lat, lon, feats, wr)
        except Exception:
            pass
        return {"fonte": "open-meteo", "features": feats, "payload": wr}
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


# >>> AIR-ADAPTERS-START
async def _fetch_openaq_pm25(lat: float, lon: float, radius_m: int) -> Optional[Dict[str, Any]]:
    """
    Busca a medição mais recente de PM2.5 no OpenAQ (estação mais próxima dentro do raio).
    """
    url = (
        "https://api.openaq.org/v3/measurements"
        f"?coordinates={lat:.6f},{lon:.6f}"
        f"&radius={radius_m}"
        "&parameter=pm25&limit=1&sort=desc&order_by=datetime"
    )
    try:
        async with httpx.AsyncClient(timeout=AIR_TIMEOUT_SECS) as cli:
            r = await cli.get(url)
            r.raise_for_status()
            js = r.json()
            results = js.get("results") or []
            if not results:
                return None
            m = results[0]
            pm25 = m.get("value")
            loc = m.get("location")
            prov = m.get("provider") or "openaq"
            dtime = (m.get("date") or {}).get("utc") or (m.get("date") or {}).get("local")
            if not _is_valid_pm25(pm25):
                return {
                    "pm25": None, "source": "station", "estimated": False,
                    "radius_m": radius_m, "station": {"name": loc, "provider": prov, "datetime": dtime}
                }
            return {
                "pm25": float(pm25), "source": "station", "estimated": False,
                "radius_m": radius_m, "station": {"name": loc, "provider": prov, "datetime": dtime}
            }
    except Exception:
        return None

async def _fetch_openmeteo_model_pm25(lat: float, lon: float) -> Optional[Dict[str, Any]]:
    """
    Fallback por modelo global (Open-Meteo Air Quality).
    """
    url = (
        "https://air-quality-api.open-meteo.com/v1/air-quality"
        f"?latitude={lat:.6f}&longitude={lon:.6f}"
        "&hourly=pm2_5&forecast_days=0&past_days=1&timezone=UTC"
    )
    try:
        async with httpx.AsyncClient(timeout=AIR_TIMEOUT_SECS) as cli:
            r = await cli.get(url)
            r.raise_for_status()
            js = r.json()
            hourly = (js.get("hourly") or {})
            times = hourly.get("time") or []
            pm25s = hourly.get("pm2_5") or []
            if not times or not pm25s:
                return None
            pm25 = pm25s[-1]
            if not _is_valid_pm25(pm25):
                return {"pm25": None, "source": "model", "estimated": True, "radius_m": None, "station": None}
            return {"pm25": float(pm25), "source": "model", "estimated": True, "radius_m": None, "station": None}
    except Exception:
        return None
# >>> AIR-ADAPTERS-END

# >>> AIR-RESOLVER-START
async def resolve_air_quality(lat: float, lon: float, air_radius_m: Optional[int] = None) -> Tuple[Dict[str, Any], str]:
    """
    Tenta estações (OpenAQ) com raios progressivos; se não houver dado válido, cai no modelo (Open-Meteo).
    Retorna (air_dict, air_status), onde air_status ∈ {"ok", "sem_estacao", "sem_dado"}.
    """
    seq = [int(x) for x in AIR_RADIUS_SEQ.split(",") if x.strip().isdigit()]
    if air_radius_m and air_radius_m not in seq:
        seq = [air_radius_m] + [r for r in seq if r != air_radius_m]

    last_station_meta = None

    # 1) Estações
    for radius in seq:
        r = await _fetch_openaq_pm25(lat, lon, radius)
        if r is None:
            continue
        last_station_meta = r
        if _is_valid_pm25(r.get("pm25")):
            return (r, "ok")

    # 2) Fallback por modelo
    if AIR_ENABLE_MODEL_FALLBACK:
        fm = await _fetch_openmeteo_model_pm25(lat, lon)
        if fm and _is_valid_pm25(fm.get("pm25")):
            return (fm, "ok")
        return (_coalesce(fm, {"pm25": None, "source": "model", "estimated": True, "radius_m": None, "station": None}), "sem_dado")

    # 3) Sem fallback habilitado
    if last_station_meta is None:
        return ({"pm25": None, "source": "station", "estimated": False, "radius_m": None, "station": None}, "sem_estacao")
    else:
        return (last_station_meta, "sem_dado")
# >>> AIR-RESOLVER-END


# ------------------------------------------------------------------------------
# Air (OpenAQ v3) + fallbacks
# ------------------------------------------------------------------------------
@app.get(
    "/api/air/openaq",
    summary="Qualidade do ar (OpenAQ v3) — PM2.5/PM10 com fallbacks",
    tags=["Air"],
    openapi_extra={
        "examples": {
            "Belém/PA (10km)": {"summary": "Busca location a 10 km", "value": {"lat": -1.4558, "lon": -48.5039, "radius_m": 10000}},
            "Manaus/AM (25km)": {"summary": "Aumenta raio para achar estações", "value": {"lat": -3.1316, "lon": -59.9825, "radius_m": 25000}},
        }
    },
)
def api_air(
    lat: float = Query(DEFAULT_LAT_f, description="Latitude."),
    lon: float = Query(DEFAULT_LON_f, description="Longitude."),
    radius_m: int = Query(10000, ge=1000, le=100000, description="Raio (m) para buscar estações."),
):
    """
    Fluxo:
      1) locations próximos; 2) latest+sensors; 3) fallback por location; 4) fallback por sensor.
      features.meta.source: latest | measurements_fallback | sensor_fallback
    """
    try:
        # 1) locations (com fallback de raio)
        locs = openaq_locations(lat, lon, radius_m=radius_m, limit=1, timeout=HTTP_TIMEOUT)
        results = locs.get("results", [])
        if not results and (locs.get("_used_radius", 0) < 25000):
            locs = openaq_locations(lat, lon, radius_m=25000, limit=1, timeout=HTTP_TIMEOUT)
            results = locs.get("results", [])

        a_norm, loc_id, latest, sensors, fallback_meas = {}, None, {}, {}, {}

        if results:
            loc_id = results[0].get("id")
            latest = openaq_latest_by_location_id(loc_id, timeout=HTTP_TIMEOUT)
            sensors = openaq_sensors_by_location_id(loc_id, timeout=HTTP_TIMEOUT)
            a_norm = normalize_openaq_v3_latest(latest, sensors)

            need_sensor_fallback = (a_norm.get("pm25") is None and a_norm.get("pm10") is None)
            if need_sensor_fallback:
                try:
                    fallback_meas = openaq_measurements_by_location_id(loc_id, limit=120, recency_days=3, timeout=HTTP_TIMEOUT)
                    vals = {"pm25": [], "pm10": []}; units = {}; last_local, last_utc = None, None
                    for m in (fallback_meas or {}).get("results", []) or []:
                        param = _param_name_to_str(m.get("parameter"))
                        val = m.get("value"); unit = m.get("unit") or ((m.get("parameter") or {}).get("units"))
                        date_obj = m.get("date") or {}
                        if date_obj.get("local") and (not last_local or date_obj["local"] > last_local): last_local = date_obj["local"]
                        if date_obj.get("utc") and (not last_utc or date_obj["utc"] > last_utc):     last_utc   = date_obj["utc"]
                        if isinstance(val, (int, float)):
                            if param in {"pm25","pm2.5"}: vals["pm25"].append(float(val)); units["pm25"] = _canonical_unit(unit, "pm25") or units.get("pm25")
                            elif param == "pm10":         vals["pm10"].append(float(val)); units["pm10"] = _canonical_unit(unit, "pm10") or units.get("pm10")
                    if vals["pm25"] or vals["pm10"]:
                        a_norm = {
                            "pm25": (sum(vals["pm25"])/len(vals["pm25"])) if vals["pm25"] else None,
                            "pm10": (sum(vals["pm10"])/len(vals["pm10"])) if vals["pm10"] else None,
                            "meta": {"units": units, "last_local": last_local, "last_utc": last_utc}
                        }
                        need_sensor_fallback = False
                except requests.HTTPError as e:
                    status = getattr(e.response, "status_code", None)
                    if status != 404:
                        raise

            # 4) fallback por sensor
            if need_sensor_fallback:
                vals = {"pm25": [], "pm10": []}; units = {}; last_local, last_utc = None, None
                for s in (sensors or {}).get("results", []) or []:
                    pname = _param_name_to_str((s.get("parameter") or {}))
                    if pname in {"pm25","pm2.5","pm10"}:
                        sid = s.get("id")
                        if isinstance(sid, int):
                            try:
                                sm = openaq_measurements_by_sensor_id(sid, limit=80, recency_days=3, timeout=HTTP_TIMEOUT)
                                for m in (sm or {}).get("results", []) or []:
                                    val = m.get("value"); unit = m.get("unit") or ((m.get("parameter") or {}).get("units"))
                                    date_obj = m.get("date") or {}
                                    if date_obj.get("local") and (not last_local or date_obj["local"] > last_local): last_local = date_obj["local"]
                                    if date_obj.get("utc") and (not last_utc or date_obj["utc"] > last_utc):     last_utc   = date_obj["utc"]
                                    if isinstance(val, (int, float)):
                                        if pname in {"pm25","pm2.5"}: vals["pm25"].append(float(val)); units["pm25"] = _canonical_unit(unit, "pm25") or units.get("pm25")
                                        elif pname == "pm10":         vals["pm10"].append(float(val)); units["pm10"] = _canonical_unit(unit, "pm10") or units.get("pm10")
                            except Exception:
                                pass
                if vals["pm25"] or vals["pm10"]:
                    a_norm = {
                        "pm25": (sum(vals["pm25"])/len(vals["pm25"])) if vals["pm25"] else None,
                        "pm10": (sum(vals["pm10"])/len(vals["pm10"])) if vals["pm10"] else None,
                        "meta": {"units": units, "last_local": last_local, "last_utc": last_utc}
                    }

        # persistência leve
        try:
            save_air(lat, lon, "openaq", loc_id, a_norm, {"latest": latest, "sensors": sensors, "measurements": fallback_meas})
        except Exception:
            pass

        # origem
        if a_norm:
            src = "latest"
            if fallback_meas: src = "measurements_fallback"
            elif a_norm.get("meta") and not a_norm["meta"].get("last_local") and not a_norm["meta"].get("last_utc"): src = "sensor_fallback"
            a_norm.setdefault("meta", {})["source"] = src

        return {
            "fonte": "openaq",
            "location_id": loc_id,
            "features": a_norm or {"pm25": None, "pm10": None, "meta": {"units": {}, "last_local": None, "last_utc": None, "source": "none"}},
            "locations": locs
        }
    except Exception as e:
        msg = str(e)
        if "Timeout" in msg or "timed out" in msg:
            raise HTTPException(status_code=504, detail=msg)
        raise HTTPException(status_code=502, detail=msg)

# ------------------------------------------------------------------------------
# INPE (Queimadas) - CSV diário/mensal
# ------------------------------------------------------------------------------
@app.get(
    "/api/inpe/focos",
    summary="Focos de queimadas do INPE (CSV diário/mensal)",
    tags=["INPE"],
    openapi_extra={
        "examples": {
            "Ex.: Diário (Belém/PA, 150 km)": {
                "summary": "Diário, Brasil, 150 km em torno de Belém/PA",
                "value": {"lat": -1.4558, "lon": -48.5039, "raio_km": 150, "scope": "diario", "region": "Brasil", "limit": 200},
            },
            "Ex.: Mensal (Manaus/AM, 250 km)": {
                "summary": "Mensal, Brasil, 250 km em torno de Manaus/AM",
                "value": {"lat": -3.1316, "lon": -59.9825, "raio_km": 250, "scope": "mensal", "region": "Brasil", "limit": 300},
            },
        }
    },
)
def api_inpe_focos(
    lat: float = Query(DEFAULT_LAT_f, description="Latitude do ponto de referência."),
    lon: float = Query(DEFAULT_LON_f, description="Longitude do ponto de referência."),
    raio_km: int = Query(150, ge=1, le=500, description="Raio de busca (km)."),
    scope: str = Query(INPE_DEFAULT_SCOPE, pattern="^(diario|mensal)$", description="Período: 'diario' ou 'mensal'."),
    region: str = Query(INPE_DEFAULT_REGION, description="Região do dataset INPE: 'Brasil' ou 'America_Sul'."),
    limit: int = Query(500, ge=1, le=2000, description="Máximo de focos após filtro espacial.")
):
    try:
        data = inpe_focos_near(lat=lat, lon=lon, raio_km=raio_km, scope=scope, region=region, limit=limit, timeout=HTTP_TIMEOUT)
        try:
            save_fire(lat=lat, lon=lon, fonte="inpe_csv", payload=data.get("payload"))
        except Exception:
            pass
        return {"fonte": "inpe_csv", "features": data.get("features"), "payload": data.get("payload")}
    except requests.HTTPError as e:
        code = e.response.status_code if getattr(e, "response", None) is not None else 502
        raise HTTPException(status_code=code, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

# ------------------------------------------------------------------------------
# ANA (stub)
# ------------------------------------------------------------------------------
@app.get("/api/ana/estacoes", summary="Estações ANA (stub com geocodificação e parâmetros)", tags=["ANA"])
def api_ana_estacoes(
    cidade: str = Query("", description="Ex.: 'Manaus, AM'"),
    raio_km: int = Query(150, ge=10, le=1000, description="Raio de busca (km)."),
    limit: int = Query(5, ge=1, le=100, description="Máximo de estações a listar."),
):
    try:
        data = ana_estacoes_stub(cidade=cidade, raio_km=raio_km, limit=limit)
        return data
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

# ------------------------------------------------------------------------------
# Demo (risco agregado)
# ------------------------------------------------------------------------------
@app.get("/api/demo/risk", summary="Demonstração: risco agregado", tags=["Demo"])
def api_demo_risk(
    lat: float = Query(DEFAULT_LAT_f, description="Latitude."),
    lon: float = Query(DEFAULT_LON_f, description="Longitude."),
    radius_m: int = Query(10000, description="Raio (m) para OpenAQ."),
):
    try:
        wraw = get_open_meteo(lat, lon, timeout=HTTP_TIMEOUT)
        w = normalize_open_meteo(wraw)
    except Exception:
        w = {}
    try:
        locs = openaq_locations(lat, lon, radius_m=radius_m, limit=1, timeout=HTTP_TIMEOUT)
        results = locs.get("results", [])
        if not results and (locs.get("_used_radius", 0) < 25000):
            locs = openaq_locations(lat, lon, radius_m=25000, limit=1, timeout=HTTP_TIMEOUT); results = locs.get("results", [])
        if results:
            loc_id = results[0]["id"]
            latest = openaq_latest_by_location_id(loc_id, timeout=HTTP_TIMEOUT)
            sensors = openaq_sensors_by_location_id(loc_id, timeout=HTTP_TIMEOUT)
            a = normalize_openaq_v3_latest(latest, sensors)
            if a.get("pm25") is None and a.get("pm10") is None:
                vals = {"pm25": [], "pm10": []}
                for s in (sensors or {}).get("results", []) or []:
                    pname = _param_name_to_str((s.get("parameter") or {}))
                    if pname in {"pm25","pm2.5","pm10"}:
                        sid = s.get("id")
                        if isinstance(sid, int):
                            sm = openaq_measurements_by_sensor_id(sid, limit=60, recency_days=3, timeout=HTTP_TIMEOUT)
                            for m in (sm or {}).get("results", []) or []:
                                val = m.get("value")
                                if isinstance(val, (int, float)):
                                    if pname in {"pm25","pm2.5"}: vals["pm25"].append(float(val))
                                    elif pname == "pm10":         vals["pm10"].append(float(val))
                a = {
                    "pm25": (sum(vals["pm25"])/len(vals["pm25"])) if vals["pm25"] else None,
                    "pm10": (sum(vals["pm10"])/len(vals["pm10"])) if vals["pm10"] else None,
                    "meta": {"units": {}}
                }
        else:
            a = {}
    except Exception:
        a = {}
    score, level = score_risk(w, a)
    units = ((a or {}).get("meta") or {}).get("units") or {}
    return {"weather": w, "air": a, "score": score, "level": level, "units": units}



@app.get("/api/check_model")
def check_model():
    from models.ai_model import has_model, MODEL_PATH
    return {"exists": has_model(), "path": MODEL_PATH}

# ------------------------------------------------------------------------------
# Index
# ------------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index():
    return """
    <html>
      <head><meta charset="utf-8"><title>AmazonSafe API (v3)</title></head>
      <body style="font-family:system-ui,Segoe UI,Roboto,Arial; line-height:1.45">
        <h1>AmazonSafe API (v3)</h1>
        <p>Endpoints úteis:</p>
        <ul>
          <li><a href="/alertas.html">/alertas.html</a> (tabela dos últimos alertas + CSV)</li>
          <li><a href="/api/alertas">/api/alertas</a> (JSON dos alertas)</li>
          <li><a href="/api/alertas.csv">/api/alertas.csv</a> (CSV dos alertas)</li>
          <li><a href="/docs">/docs</a> (Swagger UI)</li>
          <li><a href="/health">/health</a> (ok & cache)</li>
          <li><a href="/api/weather">/api/weather</a> (Open-Meteo)</li>
          <li><a href="/api/air/openaq">/api/air/openaq</a> (OpenAQ v3)</li>
          <li><a href="/api/inpe/focos">/api/inpe/focos</a> (INPE CSV diário/mensal)</li>
          <li><a href="/api/ana/estacoes?cidade=Manaus, AM&raio_km=150&limit=3">/api/ana/estacoes</a> (stub)</li>
          <li><a href="/api/demo/risk">/api/demo/risk</a> (demonstração)</li>
        </ul>
      </body>
    </html>
    """

# ------------------------------------------------------------------------------
# Endpoint OpenWeatherMap (OWM)
# ------------------------------------------------------------------------------
@app.get("/api/weather/owm", tags=["Weather"], summary="Condições atuais do tempo (OpenWeatherMap)")
def api_weather_owm(
    lat: float = Query(DEFAULT_LAT, description="Latitude"),
    lon: float = Query(DEFAULT_LON, description="Longitude")
):
    try:
        raw = get_openweather(lat, lon, timeout=HTTP_TIMEOUT)
        feats = normalize_openweather(raw)
        try:
            save_weather(lat, lon, feats, raw)
        except Exception:
            pass
        return {"fonte": "openweathermap", "features": feats, "payload": raw}
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

# ------------------------------------------------------------------------------
# Leituras de alertas (NDJSON) + JSON/CSV + HTML estilizado
# ------------------------------------------------------------------------------
ALERTS_NDJSON_PATH = "/mnt/data/alerts/alerts.ndjson"

def _read_alerts_ndjson(max_lines: int = 1000) -> List[Dict[str, Any]]:
    if not os.path.exists(ALERTS_NDJSON_PATH):
        return []
    with open(ALERTS_NDJSON_PATH, "rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        to_read = min(size, 2 * 1024 * 1024)
        f.seek(-to_read, os.SEEK_END)
        chunk = f.read().decode("utf-8", errors="ignore")
    lines = [ln for ln in chunk.splitlines() if ln.strip()]
    lines = lines[-max_lines:]
    out = []
    for ln in lines:
        try:
            out.append(json.loads(ln))
        except Exception:
            pass
    return out

@app.get("/api/alertas", summary="Lista alertas (NDJSON) recentes", tags=["Alertas"])
def api_alertas(
    limit: int = Query(50, ge=1, le=1000, description="Máximo de itens a retornar."),
    cidade: Optional[str] = Query(None, description="Filtra pelo alert_id (ex.: 'Belém, PA')."),
):
    items = _read_alerts_ndjson(max_lines=max(1000, limit*5))
    if cidade:
        items = [it for it in items if (it.get("alert_id") or "").strip() == cidade.strip()]

    def _parse_ts(x):
        try:
            return datetime.fromisoformat(x.replace("Z", "+00:00"))
        except Exception:
            return datetime.min.replace(tzinfo=timezone.utc)

    items.sort(key=lambda it: _parse_ts(it.get("ts") or ""), reverse=True)
    return items[:limit]

@app.get("/api/alertas/latest", summary="Mais recente por cidade", tags=["Alertas"])
def api_alertas_latest(
    limit_cidades: int = Query(50, ge=1, le=1000, description="Máximo de cidades."),
):
    items = _read_alerts_ndjson(max_lines=5000)
    latest_by_city: Dict[str, Dict[str, Any]] = {}
    for it in items:
        aid = (it.get("alert_id") or "").strip()
        if not aid:
            continue
        if aid not in latest_by_city or (it.get("ts") or "") > (latest_by_city[aid].get("ts") or ""):
            latest_by_city[aid] = it
    ordered = sorted(latest_by_city.values(), key=lambda x: x.get("ts",""), reverse=True)
    return ordered[:limit_cidades]

@app.get("/api/alertas.csv", summary="Exporta alertas recentes em CSV", tags=["Alertas"])
def api_alertas_csv(limit: int = 200, cidade: str | None = None):
    items = _read_alerts_ndjson(max_lines=max(2000, limit*10))
    if cidade:
        items = [it for it in items if (it.get("alert_id") or "").strip() == cidade.strip()]
    items.sort(key=lambda it: it.get("ts",""), reverse=True)
    items = items[:limit]

    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow([
        "ts","alert_id","score","level",
        "severity","duration","frequency","impact",
        "pm25","pm10","wind_speed_10m","precipitation","focos_count","observed_at"
    ])
    for it in items:
        obs  = it.get("alert_obs") or {}
        meta = obs.get("meta") or {}
        w.writerow([
            it.get("ts",""),
            it.get("alert_id",""),
            it.get("score",""),
            it.get("level",""),
            obs.get("severity",""),
            obs.get("duration",""),
            obs.get("frequency",""),
            obs.get("impact",""),
            (meta.get("pm25","")),
            (meta.get("pm10","")),
            (meta.get("wind_speed_10m","")),
            (meta.get("precipitation","")),
            (meta.get("focos_count","")),
            (meta.get("observed_at","")),
        ])
    buf.seek(0)
    return StreamingResponse(buf, media_type="text/csv",
                             headers={"Content-Disposition":"attachment; filename=alertas.csv"})

@app.get("/alertas.html", response_class=HTMLResponse, include_in_schema=False)
def alertas_html(limit: int = 50, cidade: str | None = None, auto_refresh: int = 0):
    items = _read_alerts_ndjson(max_lines=max(2000, limit*10))
    if cidade:
        items = [it for it in items if (it.get("alert_id") or "").strip() == cidade.strip()]
    items.sort(key=lambda it: it.get("ts",""), reverse=True)
    items = items[:limit]

    base_params = {"limit": limit}
    if cidade: base_params["cidade"] = cidade
    q_html  = urlencode(base_params, doseq=False)
    q_csv   = urlencode(base_params, doseq=False)
    q_auto  = urlencode({**base_params, "auto_refresh": 30}, doseq=False)

    def _badge(level: str) -> str:
        level = (level or "").lower()
        colors = {
            "verde":   ("#0f5132", "#d1e7dd", "#198754"),
            "amarelo": ("#664d03", "#fff3cd", "#ffc107"),
            "vermelho":("#842029", "#f8d7da", "#dc3545"),
        }
        fg, bg, br = colors.get(level, ("#084298","#cfe2ff","#0d6efd"))
        return f'<span class="pill" style="color:{fg};background:{bg};border:1px solid {br}">{(level or "").title()}</span>'

    def _bar(v: float, w: int = 82) -> str:
        try:
            x = max(0.0, min(1.0, float(v)))
        except Exception:
            x = 0.0
        px = int(w * x)
        return f'<div class="bar"><div style="width:{px}px"></div></div>'

    rows = []
    for it in items:
        obs  = it.get("alert_obs") or {}
        meta = obs.get("meta") or {}
        rows.append(f"""
        <tr>
          <td class="nowrap">{(it.get("ts",""))}</td>
          <td class="nowrap">{(it.get("alert_id",""))}</td>
          <td class="num">{it.get("score",0):.4f}</td>
          <td class="center">{_badge(it.get("level",""))}</td>
          <td>{_bar(obs.get("severity",0))}</td>
          <td>{_bar(obs.get("duration",0))}</td>
          <td>{_bar(obs.get("frequency",0))}</td>
          <td>{_bar(obs.get("impact",0))}</td>
          <td class="num">{"" if meta.get("pm25") is None else f"{meta.get('pm25'):,.1f}".replace(",", "_").replace(".", ",").replace("_", ".")}</td>
          <td class="num">{"" if meta.get("pm10") is None else f"{meta.get('pm10'):,.1f}".replace(",", "_").replace(".", ",").replace("_", ".")}</td>
          <td class="num">{"" if meta.get("wind_speed_10m") is None else f"{meta.get('wind_speed_10m'):,.1f}".replace(",", "_").replace(".", ",").replace("_", ".")}</td>
          <td class="num">{"" if meta.get("precipitation") is None else f"{meta.get('precipitation'):,.1f}".replace(",", "_").replace(".", ",").replace("_", ".")}</td>
          <td class="num">{"" if meta.get("focos_count") is None else meta.get("focos_count")}</td>
          <td class="nowrap">{(meta.get("observed_at") or "")}</td>
        </tr>
        """)

    meta_refresh = f'<meta http-equiv="refresh" content="{int(auto_refresh)}"/>' if auto_refresh > 0 else ""

    html = f"""
<!doctype html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8"/>
  {meta_refresh}
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Alertas recentes</title>
  <style>
    :root {{
      --primary:#0d6efd; --border:#e5e5e5; --bg:#fafafa; --muted:#666;
    }}
    body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial; margin: 18px; color:#111; }}
    h1 {{ font-size: 20px; margin:0 0 8px }}
    .meta {{ color:var(--muted); margin:4px 0 14px }}
    .toolbar {{ display:flex; flex-wrap:wrap; gap:8px; margin:10px 0 14px }}
    .btn {{
      display:inline-block; padding:8px 12px; border-radius:10px; border:1px solid var(--border);
      background:#fff; color:#111; text-decoration:none; font-weight:600; font-size:14px;
    }}
    .btn:hover {{ border-color:#c8c8c8; box-shadow:0 1px 0 rgba(0,0,0,.04) inset; }}
    .btn.primary {{ background:#0d6efd; border-color:#0d6efd; color:#fff; }}
    .btn.outline  {{ color:#0d6efd; border-color:#0d6efd; background:#fff; }}
    table {{ border-collapse:collapse; width:100%; font-size:14px; }}
    th, td {{ border:1px solid var(--border); padding:6px 8px; }}
    th {{ background:#f7f7f7; position:sticky; top:0; z-index:1; text-align:left }}
    tr:nth-child(even) {{ background:#fcfcfc }}
    .num {{ text-align:right }}
    .center {{ text-align:center }}
    .nowrap {{ white-space:nowrap }}
    .pill {{ padding:2px 8px; border-radius:999px; font-weight:600; font-size:12px; display:inline-block }}
    .bar {{ width:86px; height:8px; background:#eee; border-radius:6px; overflow:hidden }}
    .bar > div {{ height:100%; background:#0d6efd }}
    .hint {{ font-size:12px; color:var(--muted); margin-top:-8px; margin-bottom:10px }}
  </style>
</head>
<body>
  <h1>Alertas recentes</h1>
  <div class="meta">
    Mostrando <b>{len(items)}</b> itens{(' para ' + (cidade or '')) if cidade else ''} (limit={limit}).
  </div>

  <div class="toolbar">
    <a class="btn" href="/alertas.html?{q_html}">Atualizar</a>
    <a class="btn" href="/alertas.html?{q_auto}">Auto-refresh 30s</a>
    <a class="btn outline" href="/alertas.html?limit=50&cidade={quote('Belém, PA')}">Filtrar Belém, PA</a>
    <a class="btn" href="/api/alertas?{q_html}">Ver JSON</a>
    <a class="btn primary" href="/api/alertas.csv?{q_csv}">📥 Exportar CSV</a>
  </div>
  <div class="hint">Dica: o export respeita os filtros atuais (cidade/limit).</div>

  <div style="overflow:auto; max-height:75vh; border:1px solid var(--border)">
  <table>
    <thead>
      <tr>
        <th>Quando (UTC)</th>
        <th>Cidade</th>
        <th class="num">Score</th>
        <th class="center">Nível</th>
        <th>Severity</th>
        <th>Duração</th>
        <th>Frequência</th>
        <th>Impacto</th>
        <th class="num">PM2.5</th>
        <th class="num">PM10</th>
        <th class="num">Vento (m/s)</th>
        <th class="num">Chuva (mm)</th>
        <th class="num">Focos</th>
        <th>Observed@</th>
      </tr>
    </thead>
    <tbody>
      {''.join(rows) if rows else '<tr><td colspan="14">Sem dados ainda.</td></tr>'}
    </tbody>
  </table>
  </div>
</body>
</html>
"""
    return HTMLResponse(content=html, status_code=200)

# ==== [cell] =============================================

from typing import Optional, Literal
from fastapi import Query, Body, HTTPException
from pydantic import BaseModel
import math

def safe_number(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

def level_to_color(level: str) -> str:
    lv = (level or "").strip().lower()
    if lv.startswith("vermelh"):
        return "#ef4444"
    if lv.startswith("amarel"):
        return "#eab308"
    return "#22c55e"

# >>> AIR-HELPERS-START
import time  # (mantenha no topo do arquivo se preferir)

def _now_unix() -> int:
    return int(time.time())

def _coalesce(*vals, default=None):
    for v in vals:
        if v is not None:
            return v
    return default

def _is_valid_pm25(x: Optional[float]) -> bool:
    """
    Muitos provedores devolvem 0.0 quando não há medição.
    Ajuste o limiar se quiser (ex.: >= 0.5). Mantive >= 1.0 µg/m³.
    """
    try:
        xf = float(x)
        return math.isfinite(xf) and xf >= 1.0
    except Exception:
        return False
# >>> AIR-HELPERS-END


# -------------------------
# Helpers locais
# -------------------------

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

def _resolve_location(cidade: Optional[str], lat: Optional[float], lon: Optional[float]) -> tuple[float, float, dict]:
    lat = _as_float_or_none(lat)
    lon = _as_float_or_none(lon)

    # 1) Se cidade foi informada, ela manda (independente de lat/lon)
    if cidade:
        info = geocode_city(cidade)
        if not info:
            raise HTTPException(status_code=404, detail=f"Não foi possível geocodificar a cidade '{cidade}'.")
        return float(info["lat"]), float(info["lon"]), {
            "resolved_by": "geocode",
            "display_name": info.get("display_name")
        }

    # 2) Senão, usa lat/lon se ambas válidas
    if lat is not None and lon is not None:
        return lat, lon, {"resolved_by": "direct_params"}

    # 3) Fallback: defaults
    return float(DEFAULT_LAT), float(DEFAULT_LON), {"resolved_by": "default"}


def _collect_weather(lat: float, lon: float, provider: Literal["open-meteo", "owm"] = "open-meteo") -> dict:
    """
    Retorna um dicionário padronizado:
      { "fonte": <str>, "features": {...}, "payload": <raw json> }
    """
    if provider == "owm":
        raw = get_openweather(lat, lon, timeout=HTTP_TIMEOUT)
        feats = normalize_openweather(raw)
        fonte = "openweathermap"
    else:
        raw = get_open_meteo(lat, lon, timeout=HTTP_TIMEOUT)
        feats = normalize_open_meteo(raw)
        # garantir observed_at e units (Open-Meteo)
        units = (raw or {}).get("current_units") or {}
        observed_at = ((raw or {}).get("current") or {}).get("time")
        feats.setdefault("meta", {}).setdefault("units", units)
        if "observed_at" not in feats and observed_at:
            feats["observed_at"] = observed_at
        fonte = "open-meteo"

    # saneamento defensivo
    for k in ("temperature_2m", "relative_humidity_2m", "precipitation", "wind_speed_10m"):
        v = feats.get(k)
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            feats[k] = None

    return {"fonte": fonte, "features": feats, "payload": raw}

def _collect_air(lat: float, lon: float, radius_m: int = 10000) -> dict:
    locs = openaq_locations(lat, lon, radius_m=radius_m, limit=1, timeout=HTTP_TIMEOUT)
    results = locs.get("results", [])
    if not results and (locs.get("_used_radius", 0) < 25000):
        locs = openaq_locations(lat, lon, radius_m=25000, limit=1, timeout=HTTP_TIMEOUT)
        results = locs.get("results", [])

    a_norm, loc_id, latest, sensors, fallback_meas = {}, None, {}, {}, {}

    if results:
        loc_id = results[0].get("id")
        latest = openaq_latest_by_location_id(loc_id, timeout=HTTP_TIMEOUT)
        sensors = openaq_sensors_by_location_id(loc_id, timeout=HTTP_TIMEOUT)
        a_norm = normalize_openaq_v3_latest(latest, sensors)

        need_sensor_fallback = (a_norm.get("pm25") is None and a_norm.get("pm10") is None)
        if need_sensor_fallback:
            try:
                fallback_meas = openaq_measurements_by_location_id(
                    loc_id, limit=120, recency_days=3, timeout=HTTP_TIMEOUT
                )
                vals = {"pm25": [], "pm10": []}
                units = {}
                last_local, last_utc = None, None
                for m in (fallback_meas or {}).get("results", []) or []:
                    param = _param_name_to_str(m.get("parameter"))
                    val = m.get("value")
                    unit = m.get("unit") or ((m.get("parameter") or {}).get("units"))
                    date_obj = m.get("date") or {}
                    if date_obj.get("local") and (not last_local or date_obj["local"] > last_local):
                        last_local = date_obj["local"]
                    if date_obj.get("utc") and (not last_utc or date_obj["utc"] > last_utc):
                        last_utc = date_obj["utc"]
                    if isinstance(val, (int, float)):
                        if param in {"pm25", "pm2.5"}:
                            vals["pm25"].append(float(val))
                            units["pm25"] = _canonical_unit(unit, "pm25") or units.get("pm25")
                        elif param == "pm10":
                            vals["pm10"].append(float(val))
                            units["pm10"] = _canonical_unit(unit, "pm10") or units.get("pm10")
                if vals["pm25"] or vals["pm10"]:
                    a_norm = {
                        "pm25": (sum(vals["pm25"]) / len(vals["pm25"])) if vals["pm25"] else None,
                        "pm10": (sum(vals["pm10"]) / len(vals["pm10"])) if vals["pm10"] else None,
                        "meta": {"units": units, "last_local": last_local, "last_utc": last_utc}
                    }
                    need_sensor_fallback = False
            except requests.HTTPError as e:
                if getattr(e.response, "status_code", None) != 404:
                    raise

        if need_sensor_fallback:
            vals = {"pm25": [], "pm10": []}
            units = {}
            last_local, last_utc = None, None
            for s in (sensors or {}).get("results", []) or []:
                pname = _param_name_to_str((s.get("parameter") or {}))
                if pname in {"pm25", "pm2.5", "pm10"}:
                    sid = s.get("id")
                    if isinstance(sid, int):
                        try:
                            sm = openaq_measurements_by_sensor_id(sid, limit=80, recency_days=3, timeout=HTTP_TIMEOUT)
                            for m in (sm or {}).get("results", []) or []:
                                val = m.get("value")
                                unit = m.get("unit") or ((m.get("parameter") or {}).get("units"))
                                date_obj = m.get("date") or {}
                                if date_obj.get("local") and (not last_local or date_obj["local"] > last_local):
                                    last_local = date_obj["local"]
                                if date_obj.get("utc") and (not last_utc or date_obj["utc"] > last_utc):
                                    last_utc = date_obj["utc"]
                                if isinstance(val, (int, float)):
                                    if pname in {"pm25", "pm2.5"}:
                                        vals["pm25"].append(float(val))
                                        units["pm25"] = _canonical_unit(unit, "pm25") or units.get("pm25")
                                    elif pname == "pm10":
                                        vals["pm10"].append(float(val))
                                        units["pm10"] = _canonical_unit(unit, "pm10") or units.get("pm10")
                        except Exception:
                            pass
            if vals["pm25"] or vals["pm10"]:
                a_norm = {
                    "pm25": (sum(vals["pm25"]) / len(vals["pm25"])) if vals["pm25"] else None,
                    "pm10": (sum(vals["pm10"]) / len(vals["pm10"])) if vals["pm10"] else None,
                    "meta": {"units": units, "last_local": last_local, "last_utc": last_utc}
                }

    if a_norm:
        src = "latest"
        if fallback_meas:
            src = "measurements_fallback"
        elif a_norm.get("meta") and not a_norm["meta"].get("last_local") and not a_norm["meta"].get("last_utc"):
            src = "sensor_fallback"
        a_norm.setdefault("meta", {})["source"] = src

    return {
        "fonte": "openaq",
        "location_id": loc_id,
        "features": a_norm or {"pm25": None, "pm10": None, "meta": {"units": {}, "source": "none"}},
        "locations": locs
    }

def _collect_inpe(lat: float, lon: float, raio_km: int, scope: str, region: str, limit: int) -> dict:
    data = inpe_focos_near(
        lat=lat, lon=lon, raio_km=raio_km,
        scope=scope, region=region, limit=limit, timeout=HTTP_TIMEOUT
    )
    return {"fonte": "inpe_csv", "features": data.get("features"), "payload": data.get("payload")}

# -------------------------
# -------------------------
@app.get(
    "/api/data",
    summary="Consolidado: clima + ar + focos INPE por cidade/coordenadas",
    tags=["Data"]
)
def api_data(
    cidade: str = Query(..., description="Ex.: 'Belém, PA' ou 'Manaus, AM'"),
    raio_km: int = Query(150, ge=1, le=500, description="Raio (km) para focos do INPE."),
    air_radius_m: int = Query(10000, ge=1000, le=100000, description="Raio (m) para estações OpenAQ."),
    scope: str = Query(INPE_DEFAULT_SCOPE, pattern="^(diario|mensal)$", description="INPE: 'diario' ou 'mensal'."),
    region: str = Query(INPE_DEFAULT_REGION, description="INPE: 'Brasil' ou 'America_Sul'."),
    limit: int = Query(300, ge=1, le=2000, description="INPE: máximo de focos após filtro espacial."),
    weather_provider: Literal["open-meteo", "owm"] = Query("open-meteo", description="Provedor de clima")
):
    lat, lon, meta_loc = _resolve_location(cidade=cidade, lat=None, lon=None)
    out = {"cidade": cidade, "lat": lat, "lon": lon, "resolve": meta_loc, "weather_provider": weather_provider}

    # Clima
    try:
        out["weather"] = _collect_weather(lat, lon, provider=weather_provider)
        try:
            save_weather(lat, lon, out["weather"]["features"], out["weather"]["payload"])
        except Exception:
            pass
    except Exception as e:
        out["weather_error"] = str(e)

    # Ar
    try:
        out["air"] = _collect_air(lat, lon, radius_m=air_radius_m)
        try:
            save_air(
                lat, lon, "openaq", out["air"].get("location_id"),
                out["air"].get("features"), {"locations": out["air"].get("locations")}
            )
        except Exception:
            pass
    except Exception as e:
        out["air_error"] = str(e)

    # INPE
    try:
        out["focos"] = _collect_inpe(lat, lon, raio_km, scope, region, limit)
        try:
            save_fire(lat, lon, "inpe_csv", payload=out["focos"].get("payload"))
        except Exception:
            pass
    except Exception as e:
        out["focos_error"] = str(e)

    return out
    
# ============ GARANTIA DE SCORING (fallbacks se faltarem) ============

from dataclasses import dataclass
from typing import Dict, Any, Optional
from datetime import datetime, timezone
import math

# Limiares e pesos básicos (use os seus, se já existirem)
THRESHOLDS = globals().get("THRESHOLDS", {
    "green_lt": 0.33,
    "yellow_lt": 0.66,
})
WEIGHTS = globals().get("WEIGHTS", {
    "severity": 0.25,
    "duration": 0.25,
    "frequency": 0.25,
    "impact":   0.15,
    "rainfall": 0.10,
})

@dataclass
class ScoreResult:
    score: float
    level: str
    breakdown: Dict[str, float]

def _clip01(x: float) -> float:
    try: return max(0.0, min(1.0, float(x)))
    except Exception: return 0.0

def _classify_level(score: float) -> str:
    if score < THRESHOLDS["green_lt"]: return "verde"
    if score < THRESHOLDS["yellow_lt"]: return "amarelo"
    return "vermelho"

# ---- índice de chuva (0..1) ----
if "_rainfall_index" not in globals():
    def _rainfall_index(mm: Optional[float]) -> float:
        """Normaliza precipitação (mm) para 0..1 (cap em 50mm)."""
        if mm is None: return 0.0
        try: v = float(mm)
        except Exception: return 0.0
        if v <= 0: return 0.0
        return 1.0 if v >= 50.0 else (v / 50.0)

# ---- consolidador de observações (se faltar) ----
if "build_alert_obs" not in globals():
    def build_alert_obs(weather_features: Dict[str, Any],
                        air_features: Dict[str, Any],
                        fire_features: Dict[str, Any]) -> Dict[str, Any]:
        pm25 = (air_features or {}).get("pm25")
        pm10 = (air_features or {}).get("pm10")
        precip = (weather_features or {}).get("precipitation")
        wind = (weather_features or {}).get("wind_speed_10m")
        observed_at = (weather_features or {}).get("observed_at")
        focos = ((fire_features or {}).get("focos") or [])
        focos_count = len(focos)

        # severity: pior dos PMs normalizado
        def _norm_pm(val, kind):
            if val is None: return 0.0
            ref = 75.0 if kind == "pm25" else 150.0
            try: return _clip01(float(val) / ref)
            except Exception: return 0.0
        sev = max(_norm_pm(pm25, "pm25"), _norm_pm(pm10, "pm10"))

        # duration: horas sem chuva (observed_at) -> 0..1
        def _hours_since(iso_str: Optional[str]) -> Optional[float]:
            if not iso_str: return None
            try:
                dt_ = datetime.fromisoformat(str(iso_str).replace("Z", "+00:00"))
                now_ = datetime.now(timezone.utc)
                return max(0.0, (now_ - dt_).total_seconds()/3600.0)
            except Exception:
                return None
        if precip and float(precip) > 0:
            dur = 0.0
        else:
            hrs = _hours_since(observed_at) or 12.0
            dur = _clip01(hrs / 12.0)

        # frequency: nº de focos normalizado
        freq = _clip01(focos_count / 50.0)

        # impact: vento normalizado (60km/h ~ 1.0)
        try:  imp = _clip01(float(wind or 0.0) / 60.0)
        except Exception: imp = 0.0

        return {
            "severity": round(sev, 4),
            "duration": round(dur, 4),
            "frequency": round(freq, 4),
            "impact":   round(imp, 4),
            "meta": {
                "pm25": pm25, "pm10": pm10,
                "precipitation": precip,
                "observed_at": observed_at,
                "wind_speed_10m": wind,
                "focos_count": focos_count,
            },
        }

# ---- cálculo do score (se faltar) ----
if "compute_alert_score" not in globals():
    def compute_alert_score(alert_obs: Dict[str, Any],
                            weights: Dict[str, float] = WEIGHTS) -> ScoreResult:
        sev = _clip01(alert_obs.get("severity", 0))
        dur = _clip01(alert_obs.get("duration", 0))
        freq = _clip01(alert_obs.get("frequency", 0))
        imp = _clip01(alert_obs.get("impact", 0))
        precip_24h = (
            alert_obs.get("precip_24h")
            or alert_obs.get("precipitation")
            or (alert_obs.get("meta", {}).get("precipitation"))
            or 0.0
        )
        rain = _rainfall_index(precip_24h)
        score = (
            sev * weights.get("severity", 0.25) +
            dur * weights.get("duration", 0.25) +
            freq * weights.get("frequency", 0.25) +
            imp * weights.get("impact",   0.15) +
            rain* weights.get("rainfall", 0.10)
        )
        level = _classify_level(score)
        return ScoreResult(
            score=round(score, 4),
            level=level,
            breakdown={
                "severity": sev, "duration": dur,
                "frequency": freq, "impact": imp,
                "precip_index": rain,
                "precip_24h_mm": float(precip_24h) if precip_24h is not None else None,
            },
        )
# -------------------------
# POST /api/alertas_update (persistência + score/nível)
# -------------------------
from typing import Optional, Literal
import asyncio  # <- necessário para o wrapper síncrono do resolver

class AlertasUpdateBody(BaseModel):
    cidade: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    raio_km: int = 150
    air_radius_m: int = 10000
    scope: str = INPE_DEFAULT_SCOPE    # 'diario' ou 'mensal'
    region: str = INPE_DEFAULT_REGION  # 'Brasil' ou 'America_Sul'
    limit: int = 300
    weather_provider: Literal["open-meteo", "owm"] = "open-meteo"

# --- Wrapper síncrono para chamar o resolver assíncrono de qualidade do ar ----
def resolve_air_quality_sync(lat: float, lon: float, air_radius_m: Optional[int] = None):
    """
    Chama resolve_air_quality (async) a partir de um endpoint síncrono.
    Usa o loop corrente se existir; caso contrário, cria um novo.
    """
    try:
        loop = asyncio.get_running_loop()
        return loop.run_until_complete(resolve_air_quality(lat, lon, air_radius_m=air_radius_m))
    except RuntimeError:
        # Sem loop ativo: cria um novo
        return asyncio.run(resolve_air_quality(lat, lon, air_radius_m=air_radius_m))


@app.post(
    "/api/alertas_update",
    summary="Atualiza observações no banco (clima, ar, focos INPE) e calcula score/nível",
    tags=["Persistência"]
)
def api_alertas_update(body: AlertasUpdateBody = Body(...)):
    # Resolve coordenadas
    lat, lon, meta_loc = _resolve_location(cidade=body.cidade, lat=body.lat, lon=body.lon)

    persisted = {"weather": 0, "air": 0, "fire": 0, "alert_score": 0}
    errors = {}

    # -------- Weather --------
    weather = {}
    try:
        weather = _collect_weather(lat, lon, provider=body.weather_provider)
        # >>> FIX 1: passar a fonte também
        save_weather(
            lat, lon,
            weather.get("fonte") or body.weather_provider,
            weather.get("features") or {},
            weather.get("payload")
        )
        persisted["weather"] += 1
    except Exception as e:
        errors["weather"] = str(e)

        # -------- Air (resolver com raio progressivo + fallback de modelo) --------
    air = {}
    air_status = "sem_dado"
    try:
        air_dict, air_status = resolve_air_quality_sync(lat, lon, air_radius_m=body.air_radius_m)
        air = {
            "pm25": air_dict.get("pm25"),
            "source": air_dict.get("source"),
            "estimated": bool(air_dict.get("estimated")),
            "radius_m": air_dict.get("radius_m"),
            "station": air_dict.get("station"),
            "features": {"pm25": air_dict.get("pm25")}
        }
        try:
            save_air(
                lat, lon,
                air.get("source") or "openaq/model",
                (air.get("station") or {}).get("id"),
                air.get("features") or {},
                raw=air
            )
        except Exception:
            pass
        persisted["air"] += 1
    except Exception as e:
        errors["air"] = str(e)

    # -------- INPE / Fire --------
    focos = {}
    try:
        focos = _collect_inpe(lat, lon, body.raio_km, body.scope, body.region, body.limit)
        save_fire(lat, lon, "inpe_csv", payload=focos.get("payload"))
        persisted["fire"] += 1
    except Exception as e:
        errors["fire"] = str(e)

    # ---- Score & Persistência do alerta --------------------------------------
    score_payload = {}
    try:
        alert_obs = build_alert_obs(
            weather_features=(weather or {}).get("features") or {},
            air_features=(air or {}).get("features") or {},
            fire_features=(focos or {}).get("features") or {},
        )

        sr = compute_alert_score(alert_obs)

        # --- IA COM NOVO PIPELINE ------------------------------------
        try:
            weather_f = (weather or {}).get("features") or {}
            air_f     = (air or {}).get("features") or {}
            fire_f    = (focos or {}).get("features") or {}

            X_model = [[
                weather_f.get("chuva_mm", 0),
                air_f.get("pm25", 0),
                air_f.get("pm10", 0),
                weather_f.get("vento_m_s", 0),
                fire_f.get("frp", 0),
                fire_f.get("focos", 0),
            ]]

            pred = pipeline.predict(X_model)[0]
            proba = pipeline.predict_proba(X_model)[0].tolist() if hasattr(pipeline, "predict_proba") else []

            labels = {0: "verde", 1: "amarelo", 2: "vermelho"}
            alerta_final = labels.get(int(pred), "desconhecido")

            pred_json = {
                "modelo": "pipeline_v1",
                "label": alerta_final,
                "proba": proba,
                "features": {
                    "chuva_mm": X_model[0][0],
                    "pm25":     X_model[0][1],
                    "pm10":     X_model[0][2],
                    "vento_m_s": X_model[0][3],
                    "frp":       X_model[0][4],
                    "focos":     X_model[0][5],
                }
            }

            score_payload["ai_alert"] = {"label": alerta_final}
            score_payload["ai_pred"] = pred_json

            try:
                alert_obs["ai"] = {"alerta": alerta_final, "predicao": pred_json}
            except Exception:
                pass

        except Exception as _ai_err:
            score_payload["ai_error"] = str(_ai_err)

        alert_id = (body.cidade or f"geo:{lat:.4f},{lon:.4f}").strip()

        save_alert_score(
            alert_id=alert_id,
            score=sr.score,
            level=sr.level,
            alert_obs=alert_obs,
            params={
                "cidade": body.cidade,
                "lat": lat,
                "lon": lon,
                "resolve": meta_loc,
                "weather_provider": body.weather_provider,
                "air_radius_m": body.air_radius_m,
                "raio_km": body.raio_km,
                "scope": body.scope,
                "region": body.region,
                "limit": body.limit,
            }
        )

        handle_level_transition(
            alert_id=alert_id,
            new_level=sr.level,
            score=sr.score,
            alert_obs=alert_obs,
            extra={"cidade": body.cidade, "lat": lat, "lon": lon}
        )

        _append_alerta_ndjson(alert_id, sr, alert_obs)

        score_payload.update({
            "alert_id": alert_id,
            "score": sr.score,
            "level": sr.level,
            "level_color": level_to_color(sr.level),
            "breakdown": sr.breakdown,
            "alert_obs": alert_obs
        })

        score_payload.setdefault("ai_alert", {"label": "desconhecido"})
        score_payload.setdefault("ai_pred", {"modelo": "none", "proba": {}, "features": {}})
        score_payload["air_status"] = air_status

        persisted["alert_score"] += 1

    except Exception as e:
        errors["alert_score"] = str(e)

    # --------- Resposta ao front ---------
    return {
        "ok": True,
        "params": {
            "cidade": body.cidade,
            "lat": lat,
            "lon": lon,
            "resolve": meta_loc,
            "weather_provider": body.weather_provider,
            "air_radius_m": body.air_radius_m,
            "raio_km": body.raio_km,
            "scope": body.scope,
            "region": body.region,
            "limit": body.limit,
        },
        "weather": (weather or {}).get("features") or {},
        "air_status": air_status,
        "air": air or {},
        "focos": (focos or {}).get("features") or {},
        "score": score_payload,
        "persisted": persisted,
        "errors": errors,
    }


from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

# ⚠️ Garanta que o app ainda esteja instanciado no seu arquivo!
app = FastAPI()

@app.get("/health")
def health():
    return {"ok": True}

# ✅ Carrega o pipeline completo treinado (modelo + imputer + scaler)
pipeline = joblib.load("models/amazonsafe_pipeline.joblib")

class EntradaModelo(BaseModel):
    chuva_mm: float
    pm25: float | None = None
    pm10: float | None = None
    vento_m_s: float
    frp: float
    focos: int

@app.post("/api/ai_score")
def prever_risco(entrada: EntradaModelo):
    try:
        entrada_lista = [[
            entrada.chuva_mm,
            entrada.pm25,
            entrada.pm10,
            entrada.vento_m_s,
            entrada.frp,
            entrada.focos
        ]]

        predicao = pipeline.predict(entrada_lista)[0]
        descricao = {0: "Risco Verde", 1: "Risco Amarelo", 2: "Risco Vermelho"}[predicao]

        return {"risco_predito": int(predicao), "descricao": descricao}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
