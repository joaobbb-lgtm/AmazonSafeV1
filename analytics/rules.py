def classify_by_rules(feat: dict) -> tuple[str, dict]:
    """
    Heurística inicial:
    - vermelho: risco alto (muitos focos/FRP) OU pm2.5 alto + clima seco
    - amarelo: risco moderado (alguns focos/FRP) OU pm2.5 moderado
    - verde: caso contrário
    """
    focos = feat.get("focos", 0.0)
    pm25 = feat.get("pm25", 0.0)
    rain = feat.get("rain_mm_24h", 0.0)
    frp = feat.get("frp_med", 0.0)

    score_fogo = (focos / 50.0) + (frp / 40.0)   # normalização simples
    score_ar   = (pm25 / 75.0)                   # ~OMS 24h 15/25/50/75…
    seco       = 1.0 if rain < 1.0 else 0.0

    risco = 0.5*score_fogo + 0.5*score_ar + 0.2*seco
    risco = max(0.0, min(1.5, risco))

    p_verm = max(0.0, min(1.0, (risco - 0.8) / 0.7))
    p_am   = max(0.0, min(1.0, (risco - 0.4) / 0.6))
    p_verde= max(0.0, 1.0 - max(p_verm, p_am))

    if risco >= 0.9 or (pm25 >= 100 and seco):
        alerta = "vermelho"
    elif risco >= 0.5 or (pm25 >= 50):
        alerta = "amarelo"
    else:
        alerta = "verde"

    return alerta, {
        "risco": round(risco, 3),
        "p": {
            "verde": round(p_verde, 2),
            "amarelo": round(p_am, 2),
            "vermelho": round(p_verm, 2),
        },
    }
