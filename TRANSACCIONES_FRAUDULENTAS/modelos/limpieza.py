import pandas as pd
import numpy as np

def limpiar_dataset(df: pd.DataFrame):

    df_original = df.copy()

    # Normalización básica
    for col in ["canal","estatus","geolocalizacion"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.upper()

    if "horatransaccion" in df.columns:
        df["horatransaccion"] = pd.to_datetime(df["horatransaccion"], errors="coerce")

    if "monto" in df.columns:
        df["monto"] = pd.to_numeric(df["monto"], errors="coerce")

    if "ataque" in df.columns:
        df["ataque"] = pd.to_numeric(df["ataque"], errors="coerce")

    df_rechazados = []
    motivos = []

    for idx,row in df.iterrows():
        motivo = None

        if pd.isna(row["idtransaccion"]):
            motivo="ID_TRANSACCION_NULO"
        elif df["idtransaccion"].duplicated(keep=False)[idx]:
            motivo="ID_TRANSACCION_DUPLICADO"
        elif pd.isna(row["idcliente"]):
            motivo="ID_CLIENTE_NULO"
        elif pd.isna(row["monto"]) or row["monto"]<=0:
            motivo="MONTO_INVALIDO"
        elif pd.isna(row["horatransaccion"]):
            motivo="FECHA_INVALIDA"
        elif row["ataque"] not in [0,1]:
            motivo="ATAQUE_INVALIDO"

        if motivo:
            fila = row.to_dict()
            fila["motivo_rechazo"]=motivo
            df_rechazados.append(fila)
            motivos.append(idx)

    df_limpio = df.drop(index=motivos).copy()

    # Feature derivada
    df_limpio["hora"] = df_limpio["horatransaccion"].dt.hour
    if "monto_promedio" in df_limpio.columns:
        df_limpio["ratio_monto"] = df_limpio["monto"] / df_limpio["monto_promedio"]

    df_rechazados = pd.DataFrame(df_rechazados)

    reporte = {
        "total_original": len(df_original),
        "total_limpio": len(df_limpio),
        "total_rechazado": len(df_rechazados),
        "porcentaje_rechazado": round(len(df_rechazados)/max(1,len(df_original))*100,2)
    }

    return df_limpio, df_rechazados, reporte