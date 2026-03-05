# modelos/entrenar.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
)
from sklearn.utils.class_weight import compute_sample_weight

import joblib


@dataclass
class ResultadoEntrenamiento:
    modelo: Any
    metricas: Dict[str, Any]
    columnas_usadas: Dict[str, list]
    umbral_recomendado: float


def _asegurar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # hora
    if "hora" not in out.columns:
        if "horatransaccion" in out.columns:
            out["horatransaccion"] = pd.to_datetime(out["horatransaccion"], errors="coerce")
            out["hora"] = out["horatransaccion"].dt.hour
        else:
            out["hora"] = np.nan

    # ratio_monto
    if "ratio_monto" not in out.columns and "monto_promedio" in out.columns:
        out["monto_promedio"] = pd.to_numeric(out["monto_promedio"], errors="coerce")
        out["monto"] = pd.to_numeric(out["monto"], errors="coerce")
        out["ratio_monto"] = out["monto"] / out["monto_promedio"]

    return out


def entrenar_mlp_antifraude(
    df_limpio: pd.DataFrame,
    random_state: int = 42,
    test_size: float = 0.25,
    max_iter: int = 400,
) -> ResultadoEntrenamiento:

    df = _asegurar_columnas(df_limpio)

    if "ataque" not in df.columns:
        raise ValueError("No existe la columna 'ataque' (etiqueta).")

    y = pd.to_numeric(df["ataque"], errors="coerce").fillna(0).astype(int).values

    num_cols = [c for c in ["monto", "monto_promedio", "ratio_monto", "hora"] if c in df.columns]
    cat_cols = [c for c in ["canal", "geolocalizacion"] if c in df.columns]
    bool_cols = [c for c in ["dispositivo_confianza"] if c in df.columns]

    X = df[num_cols + cat_cols + bool_cols].copy()

    # bool a 0/1
    for c in bool_cols:
        X[c] = X[c].astype(str).str.strip().str.upper().map({"TRUE": 1, "FALSE": 0, "1": 1, "0": 0, "SI": 1, "NO": 0})
        X[c] = pd.to_numeric(X[c], errors="coerce")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if len(np.unique(y)) > 1 else None
    )

    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
            ("bool", numeric_pipe, bool_cols),
        ],
        remainder="drop"
    )

    mlp = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        solver="adam",
        alpha=0.0005,
        learning_rate_init=0.001,
        max_iter=max_iter,
        random_state=random_state,
        early_stopping=True,
        n_iter_no_change=15,
        validation_fraction=0.15
    )

    pipe = Pipeline(steps=[
        ("pre", pre),
        ("clf", mlp),
    ])

    sample_w = compute_sample_weight(class_weight="balanced", y=y_train)
    pipe.fit(X_train, y_train, clf__sample_weight=sample_w)

    proba_test = pipe.predict_proba(X_test)[:, 1]

    precisions, recalls, thresholds = precision_recall_curve(y_test, proba_test)
    f1 = (2 * precisions * recalls) / (precisions + recalls + 1e-12)

    if len(thresholds) > 0:
        best_idx = int(np.nanargmax(f1[:-1]))
        umbral_recomendado = float(thresholds[best_idx])
    else:
        umbral_recomendado = 0.5

    y_pred_opt = (proba_test >= umbral_recomendado).astype(int)

    metricas = {
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "fraude_train": float(np.mean(y_train)),
        "fraude_test": float(np.mean(y_test)),
        "roc_auc": float(roc_auc_score(y_test, proba_test)) if len(np.unique(y_test)) > 1 else None,
        "avg_precision": float(average_precision_score(y_test, proba_test)) if len(np.unique(y_test)) > 1 else None,
        "umbral_recomendado": float(umbral_recomendado),
        "matriz_confusion_optimo": confusion_matrix(y_test, y_pred_opt).tolist(),
        "reporte_optimo": classification_report(y_test, y_pred_opt, output_dict=True, zero_division=0),
    }

    columnas_usadas = {"numericas": num_cols, "categoricas": cat_cols, "booleanas": bool_cols}

    return ResultadoEntrenamiento(
        modelo=pipe,
        metricas=metricas,
        columnas_usadas=columnas_usadas,
        umbral_recomendado=float(umbral_recomendado),
    )


def guardar_modelo(pipe: Any, ruta_salida: str = "modelos/modelo_mlp.joblib") -> str:
    joblib.dump(pipe, ruta_salida)
    return ruta_salida