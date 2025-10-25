# AmazonSafe API — Deploy & Operação (Render)

Este guia resume como executar, testar e operar a **AmazonSafe API** no Render.

## 🔗 Base URL
```
https://amazonsafe-api.onrender.com
```

## 🚀 Start Command (Render)
Configure o serviço Web com o comando:
```
uvicorn main:app --host 0.0.0.0 --port $PORT
```
> **Importante:** o arquivo `main.py` deve exportar `app = FastAPI(...)` no nível do módulo.  
> Não execute threads, `uvicorn.run(...)`, cloudflared, mini-cron ou `pip install` dentro do código.

## 🧪 Endpoints principais
Abra a documentação interativa (Swagger):
```
/docs
```
Ou Redoc:
```
/redoc
```

### Infra
- `GET /` → resposta simples (opcional, recomendado)
- `GET /health` → status de saúde/uptime

### Weather
- `GET /api/weather` → condições atuais (Open‑Meteo)
- `GET /api/weather/owm` → condições atuais (OpenWeatherMap)

### Air
- `GET /api/air/openqa` → qualidade do ar (OpenAQ v3) + PM2.5/PM10 com fallbacks

### INPE
- `GET /api/inpe/focos?scope=diario|mensal&region=Brasil|Amazônia|UF|...` → focos de queimadas (INPE, CSV)

### ANA
- `GET /api/ana/estacoes` → estações ANA (com geocodificação e parâmetros)

### Demo
- `GET /api/demo/risk` → demonstração: risco agregado

### Alertas
- `GET /api/alertas` → lista de alertas recentes (NDJSON)
- `GET /api/alertas/latest` → alerta mais recente do período
- `GET /api/alertas.csv` → exporta alertas recentes em CSV

### Data
- `GET /api/data` → consolidados (clima + ar + focos INPE) por cidade/coordenadas

### Persistência
- `POST /api/alertas_update` → **atualiza observações** (clima, ar, focos INPE) e calcula score/nível

> Os paths acima são os atualmente publicados (confirme sempre em `/docs`).

## 📦 Variáveis de Ambiente (opcional)
Você pode ajustar defaults via **Environment → Environment Variables** no Render:
- `INPE_CSV_BASE` (default: `https://dataserver-coids.inpe.br/queimadas/queimadas/focos/csv`)
- `INPE_DEFAULT_SCOPE` (`diario` | `mensal`, default: `diario`)
- `INPE_DEFAULT_REGION` (ex.: `Brasil`, default: `Brasil`)

Outras integrações (Open‑Meteo/OpenAQ) não exigem chave; para OWM, configure `OWM_API_KEY` se necessário.

## 🧰 Testes rápidos (curl)
```bash
# Saúde
curl -s https://amazonsafe-api.onrender.com/health

# INPE (diário, Brasil)
curl -s 'https://amazonsafe-api.onrender.com/api/inpe/focos?scope=diario&region=Brasil' | head

# Último alerta (após popular via /api/alertas_update)
curl -s https://amazonsafe-api.onrender.com/api/alertas/latest
```

## ⏱️ Agendamento (Cron) no Render
Para manter os dados atualizados, agende chamadas periódicas ao `POST /api/alertas_update`.

### Opção A — Scheduled Job (recomendado)
Crie um **Cron Job** no Render que execute a cada X minutos:
- **Command**:
  ```bash
  curl -fsS -X POST     -H "Content-Type: application/json"     -d '{"cidade":"Belém, PA","raio_km":150,"air_radius_m":10000,"scope":"diario","region":"Brasil","limit":300,"weather_provider":"open-meteo"}'     https://amazonsafe-api.onrender.com/api/alertas_update
  ```
- **Schedule**: `*/5 * * * *` (a cada 5 minutos) — ajuste conforme necessário.

> Se preferir segredo, mova o JSON para uma variável de ambiente (ex.: `ALERTS_BODY`) e use `-d "$ALERTS_BODY"`.

### Opção B — Background Worker
Crie um **Background Worker** em outro serviço que execute um script Python chamando a API em loop com `sleep`. Útil se a lógica de atualização for mais complexa.

## 📄 Requirements & runtime
Exemplo de `requirements.txt` alinhado ao código atual:
```
fastapi==0.120.0
uvicorn[standard]==0.38.0
sqlmodel==0.0.27
SQLAlchemy==2.0.44
pydantic==2.12.3
requests==2.32.5
pandas==2.3.3
orjson==3.11.4
python-dotenv==1.1.1
```

## 🩺 Troubleshooting
- **“No open ports detected”**: geralmente a importação do módulo falhou. Verifique logs:
  - `main.py` deve **exportar `app`**; não usar `nest_asyncio`; não rodar `uvicorn.run(...)` no import; não abrir threads/túneis.
- **ModuleNotFoundError: nest_asyncio**: remova todo `import nest_asyncio` e `nest_asyncio.apply()` — eram apenas para Colab.
- **404 em um endpoint**: confira o path correto em `/docs` (os caminhos oficiais são `/api/...`).

## 🔐 Segurança
- Se expor o `POST /api/alertas_update` publicamente, considere proteger com chave (header) ou caminho privado.
- Configure CORS para domínios específicos do seu frontend em produção.

---

**Pronto!** Com isso, o serviço sobe, atualiza e você monitora via `/health` e logs do Render. 
