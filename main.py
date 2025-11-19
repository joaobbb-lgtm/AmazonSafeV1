# ================================================================
#  AmazonSafe API v3 - Clean Edition (2025)
#  João Bittencourt + IA (modelo real com pipeline treinado)
# ================================================================

# -----------------------------
# Imports Gerais
# -----------------------------
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import sqlite3
import requests
import joblib
import numpy as np
import datetime
import uvicorn


# ================================================================
#  APP FASTAPI
# ================================================================
app = FastAPI(
    title="AmazonSafe API - v3",
    description="API oficial do projeto AmazonSafe (I2A2) com IA real.",
    version="3.0"
)

# CORS liberado para o front-end
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ================================================================
#  FUNÇÕES UTILITÁRIAS
# ================================================================
def now_utc_iso():
    return datetime.datetime.utcnow().isoformat(timespec="seconds")


def db_connect():
    return sqlite3.connect("amazonsafe.db")


# ================================================================
#  INICIALIZAÇÃO DO BANCO
# ================================================================
def init_db():
    try:
        conn = db_connect()
        cur = conn.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS alertas (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cidade TEXT,
                estado TEXT,
                dataset_clima TEXT,
                dataset_ar TEXT,
                dataset_inpe TEXT,
                score INTEGER,
                nivel TEXT,
                created_at TEXT
            )
        """)

        conn.commit()
        conn.close()
        print("DB pronto: tabela alertas OK.")
    except Exception as e:
        print(f"[ERRO] Falha ao iniciar DB: {e}")


init_db()


# ================================================================
#  PROVEDORES DE DADOS (Open-Meteo, AQ, INPE etc.)
# ================================================================
def obter_clima(cidade: str):
    url = "https://api.open-meteo.com/v1/forecast"
    try:
        r = requests.get(url, timeout=10, params={
            "latitude": -1.455,   
            "longitude": -48.490, 
            "hourly": "temperature_2m,precipitation",
            "timezone": "UTC"
        })
        data = r.json()
        return {
            "temp": data["hourly"]["temperature_2m"][-1],
            "rain_mm_24h": sum(data["hourly"]["precipitation"][-24:])
        }
    except:
        return {"temp": None, "rain_mm_24h": None}


def obter_qualidade_ar(cidade: str):
    url = "https://api.openaq.org/v2/latest"
    try:
        r = requests.get(url, timeout=10, params={"city": cidade})
        data = r.json()
        pm25 = None
        pm10 = None
        if "results" in data and len(data["results"]):
            for m in data["results"][0]["measurements"]:
                if m["parameter"] == "pm25":
                    pm25 = m["value"]
                if m["parameter"] == "pm10":
                    pm10 = m["value"]
        return {"pm25": pm25, "pm10": pm10}
    except:
        return {"pm25": None, "pm10": None}


def obter_focos_inpe(cidade: str):
    try:
        return {
            "count": 3,
            "frp": 12.0
        }
    except:
        return {"count": None, "frp": None}


# ================================================================
#  ENDPOINT DE SAÚDE
# ================================================================
@app.get("/health")
def health():
    return {"status": "ok", "hora": now_utc_iso()}


# ================================================================
#  ENDPOINT: Atualiza observações e grava score
# ================================================================
class PedidoUpdate(BaseModel):
    cidade: str
    estado: str | None = None


@app.post("/api/alertas_update")
def atualizar_alerta(pedido: PedidoUpdate):
    cidade = pedido.cidade

    clima = obter_clima(cidade)
    ar = obter_qualidade_ar(cidade)
    inpe = obter_focos_inpe(cidade)

    # Score simples "v1"
    focos = inpe.get("count") or 0
    rain = clima.get("rain_mm_24h") or 0
    pm25 = ar.get("pm25") or 0

    # Exemplo de regra simples (não IA)
    if focos > 5 or pm25 > 80:
        score = 2
        nivel = "vermelho"
    elif focos > 2 or pm25 > 35:
        score = 1
        nivel = "amarelo"
    else:
        score = 0
        nivel = "verde"

    # Persiste no banco
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO alertas (cidade, estado, dataset_clima, dataset_ar, dataset_inpe, score, nivel, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        cidade,
        pedido.estado,
        str(clima),
        str(ar),
        str(inpe),
        score,
        nivel,
        now_utc_iso()
    ))
    conn.commit()
    conn.close()

    return {
        "cidade": cidade,
        "score": score,
        "nivel": nivel,
        "dados": {
            "clima": clima,
            "ar": ar,
            "inpe": inpe
        }
    }


# ================================================================
#  IA REAL - Pipeline Amazonsafe v3
# ================================================================
# Carregar modelo treinado
try:
    pipeline = joblib.load("models/amazonsafe_pipeline.joblib")
    print("Modelo IA carregado 🔥")
except Exception as e:
    pipeline = None
    print(f"[ERRO] Falha ao carregar modelo IA: {e}")


class EntradaModelo(BaseModel):
    chuva_mm: float
    pm25: float | None = None
    pm10: float | None = None
    vento_m_s: float
    frp: float
    focos: int


@app.post("/api/ai_score")
def prever_risco(entrada: EntradaModelo):
    if pipeline is None:
        raise HTTPException(
            status_code=500,
            detail="Modelo de IA não carregado no servidor."
        )

    try:
        entrada_lista = [[
            entrada.chuva_mm,
            entrada.pm25,
            entrada.pm10,
            entrada.vento_m_s,
            entrada.frp,
            entrada.focos
        ]]

        predicao = int(pipeline.predict(entrada_lista)[0])

        descricao = {
            0: "Risco Verde",
            1: "Risco Amarelo",
            2: "Risco Vermelho"
        }.get(predicao, "Desconhecido")

        return {
            "risco_predito": predicao,
            "descricao": descricao
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao processar previsão: {str(e)}"
        )


# ================================================================
#  MAIN (LOCAL)
# ================================================================
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=10000, reload=True)
