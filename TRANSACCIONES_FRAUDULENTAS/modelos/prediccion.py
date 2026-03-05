# modelos/prediccion.py
from __future__ import annotations

from typing import Any, Dict, Union
import pandas as pd
import joblib


def cargar_modelo(ruta: str = "modelos/modelo_mlp.joblib"):
    return joblib.load(ruta)


def preparar_entradas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Debe coincidir EXACTO con entrenar.py:
    numéricas: monto, monto_promedio, ratio_monto, hora
    categóricas: canal, geolocalizacion
    booleanas: dispositivo_confianza (0/1)
    """
    out = df.copy()

    # --- hora desde horatransaccion ---
    if "hora" not in out.columns:
        if "horatransaccion" in out.columns:
            out["horatransaccion"] = pd.to_datetime(out["horatransaccion"], errors="coerce")
            out["hora"] = out["horatransaccion"].dt.hour
        else:
            out["hora"] = pd.NA

    # --- numéricos ---
    for col in ["monto", "monto_promedio"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    # --- ratio_monto ---
    if "ratio_monto" not in out.columns:
        if "monto" in out.columns and "monto_promedio" in out.columns:
            out["ratio_monto"] = out["monto"] / out["monto_promedio"]

    # --- normalizar texto ---
    for col in ["canal", "geolocalizacion"]:
        if col in out.columns:
            out[col] = out[col].astype(str).str.strip().str.upper()

    # --- boolean a 0/1 ---
    if "dispositivo_confianza" in out.columns:
        out["dispositivo_confianza"] = (
            out["dispositivo_confianza"]
            .astype(str).str.strip().str.upper()
            .map({"TRUE": 1, "FALSE": 0, "1": 1, "0": 0, "SI": 1, "NO": 0})
        )
        out["dispositivo_confianza"] = pd.to_numeric(out["dispositivo_confianza"], errors="coerce")

    columnas = [
        "monto", "monto_promedio", "ratio_monto", "hora",
        "canal", "geolocalizacion",
        "dispositivo_confianza",
    ]
    existentes = [c for c in columnas if c in out.columns]
    return out[existentes].copy()


def predecir_scores(modelo, df_limpio: pd.DataFrame) -> pd.Series:
    X = preparar_entradas(df_limpio)
    proba = modelo.predict_proba(X)[:, 1]
    return pd.Series(proba, index=df_limpio.index, name="score_riesgo")


def predecir_una_transaccion(modelo, payload: dict) -> float:
    """
    Devuelve SOLO el score (float).
    Si quieres devolver un dict “tipo microservicio”, hazlo en el app, no aquí.
    """
    alias = {
        "promedio_monto": "monto_promedio",
        "hora_transaccion": "horatransaccion",
        "timestamp": "horatransaccion",
        "zona": "geolocalizacion",
        "geo": "geolocalizacion",
        "dispositivo_de_confianza": "dispositivo_confianza",
    }

    normalizado = {}
    for k, v in payload.items():
        normalizado[alias.get(k, k)] = v

    df_one = pd.DataFrame([normalizado])
    X = preparar_entradas(df_one)

    score = float(modelo.predict_proba(X)[:, 1][0])
    return score


def _extraer_score(score: Union[float, int, Dict[str, Any]]) -> float:
    """
    Permite que decidir() reciba:
    - float/int
    - dict con llaves típicas: score_riesgo, score_final, score
    """
    if isinstance(score, (float, int)):
        return float(score)

    if isinstance(score, dict):
        for k in ("score_riesgo", "score_final", "score"):
            if k in score:
                return float(score[k])
        raise ValueError("El dict de score no trae 'score_riesgo'/'score_final'/'score'.")

    # último intento: castear
    return float(score)


def decidir(score: Union[float, int, Dict[str, Any]], umbral_revisar: float, umbral_bloquear: float) -> str:
    score_final = _extraer_score(score)

    if score_final >= umbral_bloquear:
        return "BLOQUEAR"
    if score_final >= umbral_revisar:
        return "REVISAR"
    return "APROBAR"