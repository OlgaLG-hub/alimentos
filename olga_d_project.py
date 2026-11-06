"""
EL RESCATISTA DE ALIMENTOS
Aplicación de predicción de desperdicio alimentario con Streamlit.

Uso:
    pip install pandas numpy matplotlib seaborn scikit-learn joblib streamlit
    streamlit run app.py

Coloca food_waste_dataset.csv en la misma carpeta que este archivo.
El modelo se entrena automáticamente si modelo_desperdicio.joblib no existe.
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# ---------------------------------------------------------------------
# CONFIGURACIÓN
# ---------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "food_waste_dataset.csv"
MODEL_PATH = BASE_DIR / "modelo_desperdicio.joblib"

TARGET = "Waste_Quantity"

BASE_CATEGORICAL_COLUMNS = [
    "Business_Type",
    "Food_Category",
    "Promotion",
    "Weather",
]

CATEGORICAL_COLUMNS = [
    *BASE_CATEGORICAL_COLUMNS,
    "Season",
]

NUMERIC_COLUMNS = [
    "Purchase_Quantity",
    "Units_Sold",
    "Remaining_Stock",
    "Shelf_Life_Days",
    "Storage_Temperature",
    "Daily_Demand",
]

REQUIRED_COLUMNS = [
    "Date",
    TARGET,
    *BASE_CATEGORICAL_COLUMNS,
    *NUMERIC_COLUMNS,
]

sns.set_theme(style="whitegrid")


# ---------------------------------------------------------------------
# CARGA Y LIMPIEZA DE DATOS
# ---------------------------------------------------------------------

def obtener_temporada(mes):
    """Devuelve la estación del año a partir del número del mes."""
    if pd.isna(mes):
        return "desconocido"
    if mes in (12, 1, 2):
        return "invierno"
    if mes in (3, 4, 5):
        return "primavera"
    if mes in (6, 7, 8):
        return "verano"
    return "otoño"


@st.cache_data
def cargar_y_limpiar_datos(ruta=DATA_PATH):
    """Carga y limpia el CSV para entrenamiento o predicción."""
    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encuentra el archivo {ruta.name}. "
            "Colócalo en la misma carpeta que este archivo."
        )

    datos = pd.read_csv(ruta)
    filas_iniciales = len(datos)

    faltantes = [col for col in REQUIRED_COLUMNS if col not in datos.columns]
    if faltantes:
        raise ValueError(
            "Faltan estas columnas obligatorias en el CSV: "
            + ", ".join(faltantes)
        )

    datos = datos.copy()
    datos["Date"] = pd.to_datetime(datos["Date"], errors="coerce")
    datos[TARGET] = pd.to_numeric(datos[TARGET], errors="coerce")

    duplicados = int(datos.duplicated().sum())
    datos = datos.drop_duplicates().dropna(subset=["Date", TARGET]).copy()

    for columna in datos.select_dtypes(include=["object"]).columns:
        datos[columna] = (
            datos[columna]
            .astype("string")
            .str.strip()
            .str.lower()
            .str.replace(r"\s+", " ", regex=True)
        )

    datos["Season"] = datos["Date"].dt.month.map(obtener_temporada)

    for columna in NUMERIC_COLUMNS:
        datos[columna] = pd.to_numeric(datos[columna], errors="coerce")
        medianas_categoria = datos.groupby("Food_Category")[columna].transform("median")
        datos[columna] = datos[columna].fillna(medianas_categoria)
        datos[columna] = datos[columna].fillna(datos[columna].median())
        datos[columna] = datos[columna].fillna(0)

    for columna in CATEGORICAL_COLUMNS:
        datos[columna] = datos[columna].fillna("desconocido").astype(str).str.lower()

    resumen = {
        "filas_iniciales": filas_iniciales,
        "duplicados_eliminados": duplicados,
        "filas_finales": len(datos),
    }
    return datos, resumen


# ---------------------------------------------------------------------
# ENTRENAMIENTO Y EVALUACIÓN
# ---------------------------------------------------------------------

def entrenar_modelo(datos):
    """Entrena un pipeline con preprocesamiento y Random Forest."""
    if len(datos) < 2:
        raise ValueError("Se necesitan al menos dos filas válidas para entrenar.")

    X = datos[CATEGORICAL_COLUMNS + NUMERIC_COLUMNS].copy()
    y = datos[TARGET].copy()

    preprocesador = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(drop="first", handle_unknown="ignore"),
                CATEGORICAL_COLUMNS,
            )
        ],
        remainder="passthrough",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocesamiento", preprocesador),
            (
                "regresor",
                RandomForestRegressor(
                    n_estimators=100,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    pipeline.fit(X_train, y_train)
    predicciones = pipeline.predict(X_test)

    metricas = {
        "MAE": mean_absolute_error(y_test, predicciones),
        "RMSE": np.sqrt(mean_squared_error(y_test, predicciones)),
        "R2": r2_score(y_test, predicciones) if len(y_test) > 1 else float("nan"),
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    return pipeline, metricas


@st.cache_resource
def cargar_o_entrenar_modelo():
    """Carga el modelo guardado o lo entrena si aún no existe."""
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH), None

    datos, _ = cargar_y_limpiar_datos()
    modelo, metricas = entrenar_modelo(datos)
    return modelo, metricas


# ---------------------------------------------------------------------
# VISUALIZACIONES
# ---------------------------------------------------------------------

def verificar_archivos_requeridos():
    """Comprueba que los recursos mínimos para ejecutar la app existan."""
    problemas = []

    if not DATA_PATH.exists():
        problemas.append(
            f"No se encontró el archivo de datos: {DATA_PATH.name}. "
            "Colócalo en la misma carpeta que este script."
        )

    if not MODEL_PATH.exists():
        problemas.append(
            f"No se encontró el modelo previo: {MODEL_PATH.name}. "
            "Se generará automáticamente al iniciar si los datos están disponibles."
        )

    return problemas


def mostrar_analisis(datos):
    st.header("Análisis exploratorio")

    st.subheader("Unidades vendidas por estación")
    ventas_temporada = (
        datos.groupby("Season", as_index=False)["Units_Sold"]
        .sum()
        .sort_values("Units_Sold", ascending=False)
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=ventas_temporada, x="Season", y="Units_Sold", ax=ax)
    ax.set_title("Unidades vendidas por estación")
    ax.set_xlabel("Estación")
    ax.set_ylabel("Unidades vendidas")
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Desperdicio por categoría de alimento")
    desperdicio_categoria = (
        datos.groupby("Food_Category", as_index=False)[TARGET]
        .sum()
        .sort_values(TARGET, ascending=False)
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.barplot(
        data=desperdicio_categoria,
        x=TARGET,
        y="Food_Category",
        hue="Food_Category",
        legend=False,
        ax=axes[0],
    )
    axes[0].set_title("Desperdicio total por categoría")
    axes[0].set_xlabel("Desperdicio total")
    axes[0].set_ylabel("Categoría")

    sns.boxplot(
        data=datos,
        x=TARGET,
        y="Food_Category",
        hue="Food_Category",
        legend=False,
        ax=axes[1],
    )
    axes[1].set_title("Distribución del desperdicio por categoría")
    axes[1].set_xlabel("Cantidad de desperdicio")
    axes[1].set_ylabel("Categoría")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Matriz de correlación")
    columnas_numericas = datos.select_dtypes(include=[np.number]).columns
    matriz_corr = datos[columnas_numericas].corr()

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(
        matriz_corr,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5,
        vmin=-1,
        vmax=1,
        ax=ax,
    )
    ax.set_title("Correlación entre variables numéricas")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Correlación por estación")
    estaciones = list(datos["Season"].dropna().unique())
    if estaciones:
        fig, axes = plt.subplots(2, 2, figsize=(14, 11))
        axes = axes.flatten()

        for indice, estacion in enumerate(estaciones[:4]):
            datos_estacion = datos[datos["Season"] == estacion]
            corr_estacion = datos_estacion.select_dtypes(include=[np.number]).corr()

            sns.heatmap(
                corr_estacion,
                annot=True,
                cmap="coolwarm",
                fmt=".2f",
                linewidths=0.5,
                vmin=-1,
                vmax=1,
                ax=axes[indice],
                cbar=(indice == min(len(estaciones), 4) - 1),
            )
            axes[indice].set_title(f"Estación: {estacion.capitalize()}")

        for indice in range(min(len(estaciones), 4), 4):
            axes[indice].set_visible(False)

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


# ---------------------------------------------------------------------
# APLICACIÓN STREAMLIT
# ---------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="El Rescatista de Alimentos",
        page_icon="🍏",
        layout="wide",
    )

    st.title("🍏 El Rescatista de Alimentos")
    st.write(
        "Herramienta predictiva de desperdicio alimentario "
        "basada en aprendizaje automático."
    )

    problemas = verificar_archivos_requeridos()
    for problema in problemas:
        if "modelo previo" in problema:
            st.info(problema)
        else:
            st.error(problema)

    try:
        datos, resumen = cargar_y_limpiar_datos()
    except FileNotFoundError as error:
        st.error(f"No se pudo abrir el archivo necesario: {error}")
        st.info("Coloca el CSV en la misma carpeta que este script y vuelve a intentarlo.")
        st.stop()
    except ValueError as error:
        st.error(f"La estructura del archivo no es válida: {error}")
        st.info("Revisa las columnas requeridas del dataset y vuelve a intentarlo.")
        st.stop()
    except Exception as error:
        st.error(f"No se pudo iniciar la aplicación: {error}")
        st.stop()

    try:
        modelo, metricas_entrenamiento = cargar_o_entrenar_modelo()
    except Exception as error:
        st.error(f"No se pudo cargar ni entrenar el modelo: {error}")
        st.stop()

    with st.expander("Resumen de la limpieza de datos"):
        st.write(f"Filas iniciales: {resumen['filas_iniciales']}")
        st.write(f"Duplicados detectados/eliminados: {resumen['duplicados_eliminados']}")
        st.write(f"Filas finales: {resumen['filas_finales']}")
        st.write("Valores nulos restantes:")
        st.dataframe(datos.isnull().sum().rename("Nulos"))

    if metricas_entrenamiento is not None:
        st.subheader("Evaluación del modelo")
        col1, col2, col3 = st.columns(3)
        col1.metric("MAE", f"{metricas_entrenamiento['MAE']:.4f}")
        col2.metric("RMSE", f"{metricas_entrenamiento['RMSE']:.4f}")
        col3.metric("R²", f"{metricas_entrenamiento['R2']:.4f}")

    with st.expander("Explorar visualizaciones"):
        mostrar_analisis(datos)

    st.markdown("---")
    st.header("Estimación del desperdicio de un día")
    st.write("Introduce las características del negocio y del inventario.")

    valores_unicos = {
        columna: sorted(datos[columna].dropna().astype(str).unique().tolist())
        for columna in CATEGORICAL_COLUMNS
    }

    col1, col2 = st.columns(2)

    with col1:
        business_type = st.selectbox(
            "Tipo de negocio",
            valores_unicos["Business_Type"],
        )
        food_category = st.selectbox(
            "Categoría de alimento",
            valores_unicos["Food_Category"],
        )
        season = st.selectbox(
            "Estación del año",
            ["invierno", "primavera", "verano", "otoño"],
        )
        promotion = st.selectbox(
            "¿Hay una promoción activa?",
            valores_unicos["Promotion"],
        )
        weather = st.selectbox(
            "Condiciones meteorológicas",
            valores_unicos["Weather"],
        )

    with col2:
        purchase_quantity = st.number_input(
            "Cantidad comprada",
            min_value=0.0,
            max_value=100000.0,
            value=100.0,
        )
        units_sold = st.number_input(
            "Unidades vendidas",
            min_value=0.0,
            max_value=100000.0,
            value=80.0,
        )
        remaining_stock = st.number_input(
            "Inventario restante",
            min_value=0.0,
            max_value=100000.0,
            value=20.0,
        )
        shelf_life = st.number_input(
            "Vida útil (días)",
            min_value=0.0,
            max_value=365.0,
            value=5.0,
        )
        temperature = st.number_input(
            "Temperatura de almacenamiento (°C)",
            min_value=-50.0,
            max_value=60.0,
            value=4.0,
        )
        daily_demand = st.number_input(
            "Demanda diaria estimada",
            min_value=0.0,
            max_value=100000.0,
            value=85.0,
        )

    if st.button("Calcular estimación de desperdicio", type="primary"):
        datos_entrada = pd.DataFrame(
            [
                {
                    "Business_Type": business_type,
                    "Food_Category": food_category,
                    "Season": season,
                    "Promotion": promotion,
                    "Weather": weather,
                    "Purchase_Quantity": purchase_quantity,
                    "Units_Sold": units_sold,
                    "Remaining_Stock": remaining_stock,
                    "Shelf_Life_Days": shelf_life,
                    "Storage_Temperature": temperature,
                    "Daily_Demand": daily_demand,
                }
            ]
        )

        prediccion = float(modelo.predict(datos_entrada)[0])
        prediccion = max(0.0, prediccion)

        st.success(f"Desperdicio estimado para hoy: **{prediccion:.2f} unidades**")

        if prediccion > 20:
            st.warning(
                "El desperdicio estimado es elevado. "
                "Considera ajustar las compras, aplicar ofertas o facilitar donaciones."
            )
        else:
            st.info(
                "El desperdicio estimado es relativamente bajo "
                "para los parámetros introducidos."
            )


if __name__ == "__main__":
    main()
