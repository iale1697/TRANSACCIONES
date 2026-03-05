import streamlit as st
import pandas as pd
import numpy as np
import sys
import os


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# ----------------------------
# Configuración general
# ----------------------------
st.set_page_config(page_title="Sistema Híbrido Antifraude (Prototipo)", layout="wide")

# ----------------------------
# Funciones auxiliares
# ----------------------------
def safe_to_datetime(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")

def hour_from_ts(ts: pd.Series) -> pd.Series:
    dt = safe_to_datetime(ts)
    return dt.dt.hour

def conteo_categorias(series: pd.Series) -> pd.DataFrame:
    """Conteo de todas las categorías (sin Top N)."""
    return (
        series.value_counts(dropna=False)
        .rename_axis("valor")
        .reset_index(name="conteo")
    )

def leer_csv(uploaded_file, sep: str, encoding: str) -> pd.DataFrame:
    return pd.read_csv(uploaded_file, sep=sep, encoding=encoding)

def normalizar_texto(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.upper()

def _score_a_float(score):
    """Por si predecir_una_transaccion devolviera dict en alguna versión vieja."""
    if isinstance(score, dict):
        return float(score.get("score_riesgo", score.get("score_final", score.get("score", 0.0))))
    return float(score)

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.title("Panel de control")

st.sidebar.subheader("📁 Archivo de entrenamiento")
uploaded = st.sidebar.file_uploader(
    "Carga tu archivo (CSV) para entrenamiento",
    type=["csv"],
    help="Se utiliza para visualizar datos antes de limpiar y posteriormente entrenar el modelo."
)

# Opciones de lectura (plegable)
sep = ","
encoding = "utf-8"
with st.sidebar.expander("⚙️ Opciones de lectura (abrir/cerrar)", expanded=False):
    sep = st.text_input("Separador", value=sep)
    encoding = st.selectbox("Codificación", ["utf-8", "utf-8-sig", "latin-1"], index=0)

# Navegación
st.sidebar.subheader("🧭 Navegación")
vista = st.sidebar.radio(
    "Selecciona una sección",
    ["Datos (EDA)", "Limpieza", "Entrenamiento", "Prueba"],
    index=0
)

# ----------------------------
# Encabezado principal
# ----------------------------
st.title("Sistema híbrido de detección de fraude bancario")
st.caption("Isai Abraham Lopez Esquivel")


# ----------------------------
# Cargar datos
# ----------------------------
# if uploaded is None:
#     st.warning("Carga un archivo CSV en la barra lateral para continuar.")
#     st.stop()

# try:
#     df = leer_csv(uploaded, sep=sep, encoding=encoding)
# except Exception as e:
#     st.error(f"No pude leer el CSV: {e}")
#     st.stop()

# ----------------------------
# Cargar datos (Lógica Híbrida)
# ----------------------------
#ruta_predeterminada = "datos/dataset_oltp.csv"
# Esto le dice a Python: "sal de 'src' y busca la carpeta 'datos'"
#ruta_predeterminada = os.path.join(ROOT_DIR, "datos", "dataset_oltp.csv")
df = None

# if uploaded is not None:
#     # Caso 1: El usuario subió un archivo manualmente
#     try:
#         df = leer_csv(uploaded, sep=sep, encoding=encoding)
#         st.sidebar.success("✅ Usando archivo cargado manualmente.")
#     except Exception as e:
#         st.error(f"Error al leer el archivo subido: {e}")
#         st.stop()
# elif os.path.exists(ruta_predeterminada):
#     # Caso 2: No hay subida manual, pero existe el archivo en el repo
#     try:
#         df = pd.read_csv(ruta_predeterminada, sep=sep, encoding=encoding)
#         st.sidebar.info("ℹ️ Cargado automáticamente desde el repositorio.")
#     except Exception as e:
#         st.sidebar.error(f"Error al cargar archivo predeterminado: {e}")
#         st.stop()
# else:
#     # Caso 3: Ni subida ni archivo en repo
#     st.warning("⚠️ Carga un archivo CSV en la barra lateral para continuar.")
#     st.stop()

# Definimos la ruta usando el directorio raíz del proyecto
ruta_predeterminada = os.path.join(ROOT_DIR, "datos", "dataset_oltp.csv")

if uploaded is not None:
    df = leer_csv(uploaded, sep=sep, encoding=encoding)
    st.sidebar.success("✅ Usando archivo cargado manualmente.")
elif os.path.exists(ruta_predeterminada):
    # Aquí es donde la nube encontrará el archivo en la carpeta 'datos'
    df = pd.read_csv(ruta_predeterminada, sep=sep, encoding=encoding)
    st.sidebar.info("ℹ️ Cargado automáticamente desde el repositorio.")
else:
    st.warning("⚠️ Carga un archivo CSV en la barra lateral para continuar.")
    st.stop()


# Normalizaciones suaves (solo para visualizar)
if "canal" in df.columns:
    df["canal"] = normalizar_texto(df["canal"])
if "estatus" in df.columns:
    df["estatus"] = normalizar_texto(df["estatus"])
if "geolocalizacion" in df.columns:
    df["geolocalizacion"] = normalizar_texto(df["geolocalizacion"])

# =========================================================
# DATOS (EDA)  <-- aquí SÍ van los KPIs grandes (como pediste)
# =========================================================
if vista == "Datos (EDA)":
    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Filas", f"{len(df):,}")
    c2.metric("Columnas", f"{df.shape[1]:,}")
    c3.metric("Clientes únicos", f"{df['idcliente'].nunique():,}" if "idcliente" in df.columns else "—")
    c4.metric("Transacciones únicas", f"{df['idtransaccion'].nunique():,}" if "idtransaccion" in df.columns else "—")

    if "ataque" in df.columns:
        ataque_num = pd.to_numeric(df["ataque"], errors="coerce").fillna(0).astype(int)
        total_fraudes = int(ataque_num.sum())
        tasa_fraude = float(total_fraudes / max(1, len(df)) * 100)

        if "idcliente" in df.columns:
            fraudes_por_cliente_tmp = (
                pd.DataFrame({"idcliente": df["idcliente"], "ataque": ataque_num})
                .groupby("idcliente")["ataque"].sum()
            )
            clientes_con_fraude = int((fraudes_por_cliente_tmp > 0).sum())
        else:
            clientes_con_fraude = 0

        c5, c6, c7 = st.columns(3)
        c5.metric("Fraudes (ataque=1)", f"{total_fraudes:,}")
        c6.metric("Tasa de fraude", f"{tasa_fraude:.2f}%")
        c7.metric("Clientes con ≥1 fraude", f"{clientes_con_fraude:,}")

    st.subheader("Datos - EDA")
    st.write(
        "Aquí se revisa el archivo **tal como se cargó** (sin transformaciones de limpieza), "
        "para entender estructura, calidad inicial y distribución de fraude (si aplica)."
    )

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Vista previa", "Tipos", "Calidad rápida", "Fraude por cliente", "Resumen por cliente"]
    )

    with tab1:
        st.dataframe(df.head(50), use_container_width=True)

    with tab2:
        tipos = pd.DataFrame({"columna": df.columns, "tipo": [str(t) for t in df.dtypes]})
        st.dataframe(tipos, use_container_width=True)

    with tab3:
        st.subheader("Calidad rápida")

        nulls = df.isna().sum().sort_values(ascending=False)
        st.write("Nulos por columna")
        st.dataframe(
            nulls.rename("nulos").reset_index().rename(columns={"index": "columna"}),
            use_container_width=True
        )

        if "idtransaccion" in df.columns:
            dup = df["idtransaccion"].duplicated().sum()
            st.write(f"Duplicados en idtransaccion: **{dup:,}**")

        if "monto" in df.columns:
            monto_num = pd.to_numeric(df["monto"], errors="coerce")
            invalid = (monto_num.isna() | (monto_num <= 0)).sum()
            st.write(f"Montos inválidos (NaN o <= 0): **{invalid:,}**")

        colA, colB, colC, colD = st.columns(4)
        with colA:
            if "canal" in df.columns:
                st.write("Canal")
                st.dataframe(conteo_categorias(df["canal"]), use_container_width=True)
        with colB:
            if "estatus" in df.columns:
                st.write("Estatus")
                st.dataframe(conteo_categorias(df["estatus"]), use_container_width=True)
        with colC:
            if "geolocalizacion" in df.columns:
                st.write("Geolocalización")
                st.dataframe(conteo_categorias(df["geolocalizacion"]), use_container_width=True)
        with colD:
            if "ataque" in df.columns:
                ataque_num = pd.to_numeric(df["ataque"], errors="coerce").fillna(0).astype(int)
                st.write("Ataque (0/1)")
                st.dataframe(conteo_categorias(ataque_num), use_container_width=True)

        if "ataque" in df.columns and "geolocalizacion" in df.columns:
            ataque_num = pd.to_numeric(df["ataque"], errors="coerce").fillna(0).astype(int)
            st.divider()
            st.write("Geolocalización: comparación fraude vs no fraude")
            comp = (
                pd.crosstab(df["geolocalizacion"], ataque_num, normalize="columns")
                .rename(columns={0: "no_fraude", 1: "fraude"})
                .fillna(0)
                .round(3)
                .reset_index()
            )
            st.dataframe(comp, use_container_width=True)

    with tab4:
        st.subheader("Fraude por cliente")

        if "ataque" not in df.columns:
            st.info("Tu archivo no tiene la columna 'ataque'. Para entrenar supervisado, necesitas una etiqueta 0/1.")
        elif "idcliente" not in df.columns:
            st.warning("No existe la columna idcliente, no puedo agrupar fraude por cliente.")
        else:
            tmp = df.copy()
            tmp["_ataque"] = pd.to_numeric(tmp["ataque"], errors="coerce").fillna(0).astype(int)

            tabla = (
                tmp.groupby("idcliente")
                .agg(
                    total_transacciones=("idcliente", "size"),
                    intentos_fraude=("_ataque", "sum")
                )
                .reset_index()
            )
            tabla["tasa_fraude_cliente"] = (tabla["intentos_fraude"] / tabla["total_transacciones"] * 100).round(2)
            tabla = tabla.sort_values(["intentos_fraude", "total_transacciones"], ascending=[False, False])

            def resaltar_fila(row):
                if row["intentos_fraude"] > 0:
                    return ["background-color: rgba(255,0,0,0.22)"] * len(row)
                return [""] * len(row)

            st.dataframe(tabla.style.apply(resaltar_fila, axis=1), use_container_width=True)

            st.download_button(
                "Descargar fraude por cliente (CSV)",
                data=tabla.to_csv(index=False).encode("utf-8"),
                file_name="fraude_por_cliente.csv",
                mime="text/csv",
            )

    with tab5:
        st.subheader("Resumen por sujeto de prueba (perfil)")

        if "idcliente" not in df.columns:
            st.warning("No existe la columna idcliente, no puedo agrupar por sujeto de prueba.")
            st.stop()

        tmp = df.copy()
        tmp["_hora"] = hour_from_ts(tmp["horatransaccion"]) if "horatransaccion" in tmp.columns else np.nan
        tmp["_monto"] = pd.to_numeric(tmp["monto"], errors="coerce") if "monto" in tmp.columns else np.nan

        cliente_sel = st.selectbox(
            "Selecciona idcliente",
            sorted(tmp["idcliente"].dropna().unique().tolist())
        )

        df_c = tmp[tmp["idcliente"] == cliente_sel].copy()

        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Transacciones", f"{len(df_c):,}")
        m2.metric("Monto total", f"{df_c['_monto'].sum(skipna=True):,.2f}")
        m3.metric("Monto promedio", f"{df_c['_monto'].mean(skipna=True):,.2f}")
        m4.metric("Monto máximo", f"{df_c['_monto'].max(skipna=True):,.2f}")
        m5.metric("Hora mínima", f"{int(df_c['_hora'].min()) if df_c['_hora'].notna().any() else '—'}")
        m6.metric("Hora máxima", f"{int(df_c['_hora'].max()) if df_c['_hora'].notna().any() else '—'}")

        if "ataque" in df_c.columns:
            a = pd.to_numeric(df_c["ataque"], errors="coerce").fillna(0).astype(int)
            st.warning(f"⚠️ Intentos de fraude (ataque=1) para este cliente: **{int(a.sum())}**")

        st.divider()

        a1, a2, a3 = st.columns(3)
        with a1:
            st.write("Canal (conteo)")
            if "canal" in df_c.columns:
                st.dataframe(conteo_categorias(df_c["canal"]), use_container_width=True)
        with a2:
            st.write("Geolocalización (conteo)")
            if "geolocalizacion" in df_c.columns:
                st.dataframe(conteo_categorias(df_c["geolocalizacion"]), use_container_width=True)
        with a3:
            st.write("Estatus / Dispositivo confianza")
            if "estatus" in df_c.columns:
                st.dataframe(conteo_categorias(df_c["estatus"]), use_container_width=True)
            if "dispositivo_confianza" in df_c.columns:
                st.dataframe(conteo_categorias(df_c["dispositivo_confianza"]), use_container_width=True)

    st.stop()

# =========================================================
# LIMPIEZA
# =========================================================
if vista == "Limpieza":
    st.subheader("Limpieza de datos (ETL controlado)")

    from modelos.limpieza import limpiar_dataset

    df_limpio, df_rechazados, reporte = limpiar_dataset(df)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total original", reporte["total_original"])
    col2.metric("Registros limpios", reporte["total_limpio"])
    col3.metric("Rechazados", reporte["total_rechazado"])
    col4.metric("% rechazado", f'{reporte["porcentaje_rechazado"]}%')

    st.divider()

    st.write("### Dataset limpio")
    st.dataframe(df_limpio.head(50), use_container_width=True)

    st.write("### Registros rechazados")
    st.dataframe(df_rechazados.head(50), use_container_width=True)

    st.download_button(
        "Descargar dataset limpio",
        data=df_limpio.to_csv(index=False).encode("utf-8"),
        file_name="dataset_limpio.csv",
        mime="text/csv",
    )

    st.download_button(
        "Descargar rechazados",
        data=df_rechazados.to_csv(index=False).encode("utf-8"),
        file_name="dataset_rechazados.csv",
        mime="text/csv",
    )

    st.stop()

# =========================================================
# ENTRENAMIENTO
# =========================================================
if vista == "Entrenamiento":
    st.subheader("Entrenamiento (Perceptrón Multicapa)")

    from modelos.limpieza import limpiar_dataset
    df_limpio, df_rechazados, reporte = limpiar_dataset(df)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total original", reporte["total_original"])
    col2.metric("Registros limpios", reporte["total_limpio"])
    col3.metric("Rechazados", reporte["total_rechazado"])
    col4.metric("% rechazado", f'{reporte["porcentaje_rechazado"]}%')

    if len(df_limpio) == 0:
        st.error("No hay datos suficientes para entrenar.")
        st.stop()

    st.divider()

    st.write("## Parámetros")
    colA, colB, colC = st.columns(3)

    proporcion_prueba = colA.slider("Proporción para prueba", 0.10, 0.40, 0.25, 0.05)
    iteraciones_maximas = colB.slider("Iteraciones máximas", 200, 800, 400, 50)
    semilla = colC.number_input("Semilla (random_state)", 0, 9999, 42, 1)

    st.divider()

    from modelos.entrenar import entrenar_mlp_antifraude, guardar_modelo

    if st.button("🚀 Entrenar modelo", type="primary"):
        with st.spinner("Entrenando modelo..."):
            resultado = entrenar_mlp_antifraude(
                df_limpio=df_limpio,
                random_state=int(semilla),
                test_size=float(proporcion_prueba),
                max_iter=int(iteraciones_maximas),
            )
            ruta_modelo = guardar_modelo(resultado.modelo, "modelos/modelo_mlp.joblib")

        st.success(f"Modelo guardado en: {ruta_modelo}")

        st.write("## Métricas principales")
        m = resultado.metricas
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Registros entrenamiento", m["n_train"])
        c2.metric("Registros prueba", m["n_test"])
        c3.metric("AUC ROC", f"{m['roc_auc']:.3f}" if m["roc_auc"] is not None else "N/A")
        c4.metric("AP (Precisión-Recall)", f"{m['avg_precision']:.3f}" if m["avg_precision"] is not None else "N/A")

        st.divider()
        st.write("## Umbral recomendado (para decisión operativa)")
        st.code(f"{m['umbral_recomendado']:.6f}")
        st.caption("Se calcula maximizando F1 en el conjunto de prueba.")

        st.divider()
        st.write("## Matriz de confusión (umbral recomendado)")
        matriz = m["matriz_confusion_optimo"]
        df_matriz = pd.DataFrame(matriz, index=["Real 0", "Real 1"], columns=["Pred 0", "Pred 1"])
        st.dataframe(df_matriz, use_container_width=True)

        st.divider()
        st.write("## Reporte (umbral recomendado)")
        reporte_df = pd.DataFrame(m["reporte_optimo"]).T
        st.dataframe(reporte_df, use_container_width=True)

        st.divider()
        st.write("## Columnas usadas como entradas (X)")
        st.json(resultado.columnas_usadas)

    st.stop()

# =========================================================
# PRUEBA
# =========================================================
if vista == "Prueba":
    st.subheader("Prueba operativa (scoring + decisión híbrida)")

    from modelos.limpieza import limpiar_dataset
    df_limpio, df_rechazados, reporte = limpiar_dataset(df)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total original", reporte["total_original"])
    col2.metric("Registros limpios", reporte["total_limpio"])
    col3.metric("Rechazados", reporte["total_rechazado"])
    col4.metric("% rechazado", f'{reporte["porcentaje_rechazado"]}%')

    if len(df_limpio) == 0:
        st.error("No hay datos limpios para probar.")
        st.stop()

    st.divider()

    from modelos.prediccion import (
        cargar_modelo, preparar_entradas,
        predecir_scores, predecir_una_transaccion, decidir
    )

    try:
        modelo = cargar_modelo()
        st.success("Modelo cargado correctamente.")
    except Exception as e:
        st.error(f"No se pudo cargar el modelo: {e}")
        st.stop()

    st.write("### Bandas de decisión operativa")
    cA, cB = st.columns(2)
    umbral_revisar = cA.slider("Umbral para REVISAR", 0.05, 0.95, 0.30, 0.01)
    umbral_bloquear = cB.slider("Umbral para BLOQUEAR", 0.05, 0.95, 0.50, 0.01)

    if umbral_bloquear < umbral_revisar:
        st.warning("El umbral de BLOQUEAR debe ser ≥ que el de REVISAR.")
        st.stop()

    st.divider()

    st.write("## Simulación de nueva transacción (microservicio)")

    with st.expander("Capturar transacción nueva", expanded=True):
        c1, c2, c3 = st.columns(3)
        idtransaccion = c1.text_input("idtransaccion", "TX_DEMO_001")
        idcliente = c2.text_input("idcliente", "7")
        monto = c3.number_input("monto", min_value=1.0, value=1200.0, step=1.0)

        c4, c5, c6 = st.columns(3)
        canal = c4.selectbox("canal", ["APP", "WEB", "ATM", "SUCURSAL"])
        geolocalizacion = c5.selectbox("geolocalizacion", ["CDMX", "GDL", "MTY", "OTRA"])
        dispositivo_confianza = c6.selectbox("dispositivo_confianza", [True, False])

        c7, c8 = st.columns(2)
        monto_promedio = c7.number_input("monto_promedio", min_value=1.0, value=100.0, step=1.0)
        horatransaccion = c8.text_input("horatransaccion (YYYY-MM-DD HH:MM:SS)", "2026-03-29 02:22:10")

        payload = {
            "idtransaccion": idtransaccion,
            "idcliente": idcliente,
            "horatransaccion": horatransaccion,
            "monto": float(monto),
            "canal": canal,
            "geolocalizacion": geolocalizacion,
            "dispositivo_confianza": bool(dispositivo_confianza),
            "monto_promedio": float(monto_promedio),
        }

        st.write("### JSON (entrada)")
        st.json(payload)

        ver_debug = st.checkbox("Ver debug de entradas X (recomendado)", value=True)

        if st.button("Evaluar riesgo (score)", type="primary"):
            X_debug = preparar_entradas(pd.DataFrame([payload]))

            if ver_debug:
                st.write("### Entradas reales (X) que recibe el modelo")
                st.dataframe(X_debug, use_container_width=True)
                st.write("Nulos por columna (X)")
                st.write(X_debug.isna().sum())

            score_raw = predecir_una_transaccion(modelo, payload)
            score = _score_a_float(score_raw)

            decision = decidir(score, umbral_revisar, umbral_bloquear)

            st.write("### Respuesta del microservicio")
            st.json({
                "idtransaccion": idtransaccion,
                "score_riesgo": round(score, 6),
                "decision": decision,
                "umbral_revisar": umbral_revisar,
                "umbral_bloquear": umbral_bloquear,
                "modelo_version": "mlp_v1"
            })

            if decision == "APROBAR":
                st.success(f"APROBAR | score={score:.4f}")
            elif decision == "REVISAR":
                st.warning(f"REVISAR | score={score:.4f}")
            else:
                st.error(f"BLOQUEAR | score={score:.4f}")

    st.divider()

    #st.write("## Scoring por lote (dataset cargado)")
    #scores = predecir_scores(modelo, df_limpio)

    #df_prueba = df_limpio.copy()
    #df_prueba["score_riesgo"] = scores

    #p95 = float(df_prueba["score_riesgo"].quantile(0.95))
    #p99 = float(df_prueba["score_riesgo"].quantile(0.99))
    #st.info(f"Umbrales sugeridos por percentil → REVISAR (p95) = {p95:.4f} | BLOQUEAR (p99) = {p99:.4f}")

    #df_prueba["decision"] = df_prueba["score_riesgo"].apply(lambda s: decidir(float(s), umbral_revisar, umbral_bloquear))

    #st.write("### Vista rápida (primeras 20)")
    #st.dataframe(df_prueba.head(20), use_container_width=True)

    
    st.stop()