from datetime import datetime

def _coalesce(*vals, default=None):
    for v in vals:
        if v is not None:
            return v
    return default

def build_features(obs: dict) -> dict:
    """
    Constrói features a partir do consolidado "obs" (clima + ar + focos INPE [+ ANA]).
    Espera, tipicamente:
      obs["weather"] {...}, obs["air"] {...}, obs["inpe"] {"count": int, "focos":[...]}
    """
    focos = obs.get("inpe", {}).get("count", 0)
    frp_med = 0.0
    focos_list = obs.get("inpe", {}).get("focos", [])
    if focos_list:
        frps = [f.get("frp") for f in focos_list if isinstance(f.get("frp"), (int, float))]
        if frps:
            frp_med = sum(frps) / len(frps)

    pm25 = _coalesce(obs.get("air", {}).get("pm25"), default=0.0)
    pm10 = _coalesce(obs.get("air", {}).get("pm10"), default=0.0)
    temp = _coalesce(obs.get("weather", {}).get("temp"), default=0.0)
    rain_mm_24h = _coalesce(obs.get("weather", {}).get("rain_24h_mm"), default=0.0)
    uvi = _coalesce(obs.get("weather", {}).get("uvi"), default=0.0)

    secos = 1.0 if rain_mm_24h < 1.0 else 0.0
    pm_ratio = (pm25 / pm10) if (pm10 and pm10 > 0) else 0.0

    return {
        "focos": float(focos),
        "frp_med": float(frp_med),
        "pm25": float(pm25),
        "pm10": float(pm10),
        "pm_ratio": float(pm_ratio),
        "temp": float(temp),
        "rain_mm_24h": float(rain_mm_24h),
        "uvi": float(uvi),
        "secos": float(secos),
        "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
    }
