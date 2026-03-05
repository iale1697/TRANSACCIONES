#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generador de dataset para prototipo antifraude (MLP) - V4

Mejoras:
- Fraude (~5%) con señales coherentes (patrón aprendible).
- No todos los clientes tienen fraude.
- “Suciedad” opcional para que limpieza.py tenga trabajo:
  - monto con " MXN"
  - valores nulos
  - canal/geo en minúsculas o con espacios
  - dispositivo_confianza como "SI"/"NO"
  - fechas inválidas
  - duplicados en idtransaccion
"""

from __future__ import annotations
import argparse
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
import pandas as pd

ZONAS = ["CDMX", "GDL", "MTY", "OTRA"]
CANALES = ["APP", "WEB", "ATM", "SUCURSAL"]

HORAS_RARAS = {0,1,2,3,4,5,22,23}

@dataclass
class PerfilCliente:
    idcliente: int
    monto_base: float
    horario: str
    dispositivo_base: bool

def elegir_hora(perfil: PerfilCliente, fecha_base: datetime, forzar_riesgo: bool) -> datetime:
    if forzar_riesgo:
        hora = random.choice(list(HORAS_RARAS))
    else:
        if perfil.horario == "todo_dia":
            hora = random.randint(0, 23)
        elif perfil.horario == "noche":
            hora = random.choice([22, 23, 0, 1, 2, 3, 4, 5])
        elif perfil.horario == "tarde_noche_madrugada":
            hora = random.choice([18, 19, 20, 21, 22, 23, 0, 1, 2, 3])
        elif perfil.horario == "laboral_lunes":
            while fecha_base.weekday() != 0:
                fecha_base += timedelta(days=1)
            hora = random.randint(9, 15)
        else:
            hora = random.randint(0, 23)

    return fecha_base.replace(
        hour=hora, minute=random.randint(0, 59), second=random.randint(0, 59), microsecond=0
    )

def elegir_canal(forzar_riesgo: bool) -> str:
    if forzar_riesgo:
        return random.choice(["WEB", "APP"])
    return random.choice(CANALES)

def elegir_geo(forzar_riesgo: bool, p_cdmx_no_fraude: float) -> str:
    if forzar_riesgo:
        # “República”: repartido
        return random.choice(ZONAS)
    # no fraude: domina CDMX
    if random.random() < p_cdmx_no_fraude:
        return "CDMX"
    return random.choice([z for z in ZONAS if z != "CDMX"])

def derivar_monto(perfil: PerfilCliente, monto_prom: float, forzar_riesgo: bool) -> float:
    # normal alrededor del base
    monto = max(1.0, perfil.monto_base * random.uniform(0.7, 1.3))
    # fraude/riesgo: mucha más prob. de “salirse”
    if forzar_riesgo:
        monto = max(monto, monto_prom * random.uniform(2.2, 6.0))
    return round(monto, 2)

def api_disp(perfil: PerfilCliente, forzar_riesgo: bool) -> bool:
    if forzar_riesgo:
        # fraude: MUY probable que sea no confiable
        return random.random() < 0.15  # 15% True, 85% False
    # normal: casi siempre su patrón
    if random.random() < 0.05:
        return not perfil.dispositivo_base
    return perfil.dispositivo_base

def score_reglas(hora: int, geo: str, disp: bool, monto: float, mp: float) -> float:
    score = 0.0
    if not disp:
        score += 0.45
    if geo != "CDMX":
        score += 0.20
    if hora in HORAS_RARAS:
        score += 0.15
    if mp and monto > mp * 2.5:
        score += 0.25
    elif mp and monto > mp * 1.8:
        score += 0.15
    return score

def ensuciar(df: pd.DataFrame, dirty_rate: float, seed: int) -> pd.DataFrame:
    random.seed(seed + 999)
    out = df.copy()

    # Para permitir strings mezcladas sin warnings:
    out["monto"] = out["monto"].astype(object)
    out["dispositivo_confianza"] = out["dispositivo_confianza"].astype(object)
    out["canal"] = out["canal"].astype(object)
    out["geolocalizacion"] = out["geolocalizacion"].astype(object)
    out["horatransaccion"] = out["horatransaccion"].astype(object)

    n = len(out)
    k = max(1, int(n * dirty_rate))
    idxs = random.sample(range(n), k=k)

    for i in idxs:
        t = random.choice(["monto_mxn", "nulls", "minusculas", "si_no", "fecha_invalida"])
        if t == "monto_mxn":
            out.at[i, "monto"] = f"{out.at[i,'monto']} MXN"
        elif t == "nulls":
            col = random.choice(["canal", "geolocalizacion", "dispositivo_confianza", "horatransaccion"])
            out.at[i, col] = None
        elif t == "minusculas":
            col = random.choice(["canal", "geolocalizacion"])
            val = str(out.at[i, col])
            out.at[i, col] = "  " + val.lower() + "  "
        elif t == "si_no":
            out.at[i, "dispositivo_confianza"] = random.choice(["SI", "NO"])
        elif t == "fecha_invalida":
            out.at[i, "horatransaccion"] = random.choice(["2026-99-99 99:99:99", "NA", ""])

    # Duplicados controlados en idtransaccion
    if n >= 20:
        for _ in range(max(1, int(n * dirty_rate * 0.2))):
            a = random.randint(0, n-1)
            b = random.randint(0, n-1)
            out.at[b, "idtransaccion"] = out.at[a, "idtransaccion"]

    return out

def generar_dataset(
    n: int,
    seed: int,
    fraude_rate: float,
    clientes_con_fraude: int,
    p_cdmx_no_fraude: float,
    dirty_rate: float
) -> pd.DataFrame:
    random.seed(seed)

    perfiles = [
        PerfilCliente(idcliente=1,   monto_base=20,  horario="laboral_lunes",         dispositivo_base=False),
        PerfilCliente(idcliente=7,   monto_base=100, horario="todo_dia",              dispositivo_base=True),
        PerfilCliente(idcliente=8,   monto_base=150, horario="todo_dia",              dispositivo_base=True),
        PerfilCliente(idcliente=9,   monto_base=200, horario="tarde_noche_madrugada", dispositivo_base=True),
        PerfilCliente(idcliente=150, monto_base=150, horario="noche",                 dispositivo_base=True),
    ]

    ids = [p.idcliente for p in perfiles]
    clientes_fraude = set(random.sample(ids, k=min(clientes_con_fraude, len(ids))))

    monto_prom = {p.idcliente: round(p.monto_base * random.uniform(0.85, 1.15), 2) for p in perfiles}

    base = datetime(2026, 3, 1)
    filas = []
    idtx = 100000

    n_fraudes = max(1, int(round(n * fraude_rate)))

    # 1) Genera transacciones normales
    for _ in range(n):
        perfil = random.choice(perfiles)
        fecha = base + timedelta(days=random.randint(0, 29))
        ts = elegir_hora(perfil, fecha, forzar_riesgo=False)
        canal = elegir_canal(forzar_riesgo=False)
        geo = elegir_geo(forzar_riesgo=False, p_cdmx_no_fraude=p_cdmx_no_fraude)
        disp = api_disp(perfil, forzar_riesgo=False)
        mp = monto_prom[perfil.idcliente]
        monto = derivar_monto(perfil, mp, forzar_riesgo=False)

        filas.append({
            "idtransaccion": idtx,
            "idcliente": perfil.idcliente,
            "horatransaccion": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "monto": monto,
            "canal": canal,
            "geolocalizacion": geo,
            "dispositivo_confianza": disp,
            "monto_promedio": mp,
            "ataque": 0,
        })
        idtx += 1

    df = pd.DataFrame(filas)

    # 2) Inserta fraudes coherentes SOLO en algunos clientes
    candidatos = df[df["idcliente"].isin(clientes_fraude)].copy()
    if len(candidatos) == 0:
        return df

    # recalcular un “riesgo por reglas” y elegir los más altos como fraude
    candidatos["_hora"] = pd.to_datetime(candidatos["horatransaccion"], errors="coerce").dt.hour.fillna(0).astype(int)
    candidatos["_score"] = [
        score_reglas(int(h), str(g), bool(d), float(m), float(mp))
        for h,g,d,m,mp in zip(
            candidatos["_hora"], candidatos["geolocalizacion"], candidatos["dispositivo_confianza"],
            candidatos["monto"], candidatos["monto_promedio"]
        )
    ]
    candidatos = candidatos.sort_values("_score", ascending=False)

    fraud_idx = candidatos.head(min(n_fraudes, len(candidatos))).index.tolist()

    # Forzar el patrón fraudulento en esas filas (para que sea MUY aprendible)
    for i in fraud_idx:
        perfil = next(p for p in perfiles if p.idcliente == int(df.at[i,"idcliente"]))
        mp = float(df.at[i,"monto_promedio"])
        fecha = base + timedelta(days=random.randint(0, 29))
        ts = elegir_hora(perfil, fecha, forzar_riesgo=True)
        df.at[i, "horatransaccion"] = ts.strftime("%Y-%m-%d %H:%M:%S")
        df.at[i, "canal"] = elegir_canal(forzar_riesgo=True)
        df.at[i, "geolocalizacion"] = elegir_geo(forzar_riesgo=True, p_cdmx_no_fraude=p_cdmx_no_fraude)
        df.at[i, "dispositivo_confianza"] = api_disp(perfil, forzar_riesgo=True)
        df.at[i, "monto"] = derivar_monto(perfil, mp, forzar_riesgo=True)
        df.at[i, "ataque"] = 1

    # 3) estatus coherente, pero NO lo uses en X (está bien que exista en el dataset)
    df["estatus"] = df["ataque"].map({1: "NO_APROBADA", 0: "APROBADA"}).astype(str)

    # 4) Ensuciar opcional
    if dirty_rate and dirty_rate > 0:
        df = ensuciar(df, dirty_rate=dirty_rate, seed=seed)

    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fraude_rate", type=float, default=0.05)
    ap.add_argument("--clientes_fraude", type=int, default=3)
    ap.add_argument("--p_cdmx", type=float, default=0.80)
    ap.add_argument("--dirty_rate", type=float, default=0.03)
    ap.add_argument("--out", type=str, default="datos/dataset_oltp.csv")
    args = ap.parse_args()

    df = generar_dataset(
        n=args.n,
        seed=args.seed,
        fraude_rate=args.fraude_rate,
        clientes_con_fraude=args.clientes_fraude,
        p_cdmx_no_fraude=args.p_cdmx,
        dirty_rate=args.dirty_rate,
    )

    df.to_csv(args.out, index=False, encoding="utf-8")
    print(f"OK -> {args.out}")
    print(f"Tasa real fraude: {df['ataque'].astype(str).str.contains('1').mean():.3f}")
    # Nota: ataque puede ensuciarse si dirty_rate toca esa columna (aquí no la tocamos)

if __name__ == "__main__":
    main()