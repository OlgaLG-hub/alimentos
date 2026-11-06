"""
EL RESCATISTA DE ALIMENTOS
Aplicación de predicción de desperdicio alimentario con Streamlit.

Uso:
    pip install pandas numpy matplotlib seaborn scikit-learn joblib streamlit
    streamlit run app.py

Coloca food_waste_dataset.csv en la misma carpeta que este archivo.
El modelo se entrena automáticamente si modelo_desperdicio.joblib no existe.
"""

from pathlib import Path  # Importa Path para trabajar con rutas de archivos de forma portable

import joblib  # Importa joblib para guardar y cargar el modelo entrenado
import matplotlib.pyplot as plt  # Importa pyplot para crear gráficos
import numpy as np  # Importa numpy para cálculos numéricos
import pandas as pd  # Importa pandas para manejar tablas de datos
import seaborn as sns  # Importa seaborn para gráficos estadísticos más atractivos
import streamlit as st  # Importa streamlit para construir la aplicación web
from sklearn.compose import ColumnTransformer  # Permite aplicar transformaciones distintas a columnas distintas
from sklearn.ensemble import RandomForestRegressor  # Modelo de bosque aleatorio para regresión
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # Métricas para evaluar el modelo
from sklearn.model_selection import train_test_split  # Divide los datos en entrenamiento y prueba
from sklearn.pipeline import Pipeline  # Encadena preprocesamiento y modelo en un solo objeto
from sklearn.preprocessing import OneHotEncoder  # Convierte categorías de texto en columnas numéricas 0/1

# ---------------------------------------------------------------------
# CONFIGURACIÓN
# ---------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent  # Carpeta donde está guardado este script
DATA_PATH = BASE_DIR / "food_waste_dataset.csv"  # Ruta completa del archivo CSV con los datos
MODEL_PATH = BASE_DIR / "modelo_desperdicio.joblib"  # Ruta donde se guarda/carga el modelo entrenado

TARGET = "Waste_Quantity"  # Nombre de la columna que queremos predecir (cantidad desperdiciada)

BASE_CATEGORICAL_COLUMNS = [  # Lista de columnas categóricas que ya vienen en el CSV
    "Business_Type",  # Tipo de negocio
    "Food_Category",  # Categoría del alimento
    "Promotion",  # Si hay promoción activa
    "Weather",  # Condiciones meteorológicas
]  # Cierra la lista de columnas categóricas base

CATEGORICAL_COLUMNS = [  # Lista completa de columnas categóricas que usará el modelo
    *BASE_CATEGORICAL_COLUMNS,  # Incluye todas las columnas categóricas base
    "Season",  # Añade la estación del año, calculada a partir de la fecha
]  # Cierra la lista de columnas categóricas

NUMERIC_COLUMNS = [  # Lista de columnas numéricas que usará el modelo
    "Purchase_Quantity",  # Cantidad comprada
    "Units_Sold",  # Unidades vendidas
    "Remaining_Stock",  # Inventario restante
    "Shelf_Life_Days",  # Vida útil en días
    "Storage_Temperature",  # Temperatura de almacenamiento
    "Daily_Demand",  # Demanda diaria
]  # Cierra la lista de columnas numéricas

REQUIRED_COLUMNS = [  # Columnas que el CSV debe contener obligatoriamente
    "Date",  # Fecha del registro
    TARGET,  # Variable objetivo (desperdicio)
    *BASE_CATEGORICAL_COLUMNS,  # Todas las columnas categóricas base
    *NUMERIC_COLUMNS,  # Todas las columnas numéricas
]  # Cierra la lista de columnas obligatorias

sns.set_theme(style="whitegrid")  # Aplica un estilo de gráficos con fondo blanco y rejilla


# ---------------------------------------------------------------------
# CARGA Y LIMPIEZA DE DATOS
# ---------------------------------------------------------------------

def obtener_temporada(mes):  # Define una función que convierte un mes en estación del año
    """Devuelve la estación del año a partir del número del mes."""
    if pd.isna(mes):  # Si el mes es un valor vacío (NaN)
        return "desconocido"  # Devuelve "desconocido" porque no se puede calcular la estación
    if mes in (12, 1, 2):  # Si el mes es diciembre, enero o febrero
        return "invierno"  # Devuelve invierno
    if mes in (3, 4, 5):  # Si el mes es marzo, abril o mayo
        return "primavera"  # Devuelve primavera
    if mes in (6, 7, 8):  # Si el mes es junio, julio o agosto
        return "verano"  # Devuelve verano
    return "otoño"  # En cualquier otro caso (sep, oct, nov) devuelve otoño


@st.cache_data  # Guarda en caché el resultado para no recargar y limpiar el CSV en cada interacción
def cargar_y_limpiar_datos(ruta=DATA_PATH):  # Función que lee el CSV y lo limpia; usa DATA_PATH por defecto
    """Carga y limpia el CSV para entrenamiento o predicción."""
    if not ruta.exists():  # Comprueba si el archivo no existe
        raise FileNotFoundError(  # Lanza un error de archivo no encontrado
            f"No se encuentra el archivo {ruta.name}. "  # Mensaje con el nombre del archivo
            "Colócalo en la misma carpeta que este archivo."  # Indica dónde colocar el archivo
        )  # Cierra la excepción

    datos = pd.read_csv(ruta)  # Lee el archivo CSV y lo guarda en un DataFrame
    filas_iniciales = len(datos)  # Guarda el número de filas antes de limpiar

    faltantes = [col for col in REQUIRED_COLUMNS if col not in datos.columns]  # Busca columnas obligatorias que no estén en el CSV
    if faltantes:  # Si falta alguna columna obligatoria
        raise ValueError(  # Lanza un error de valor
            "Faltan estas columnas obligatorias en el CSV: "  # Inicio del mensaje de error
            + ", ".join(faltantes)  # Añade la lista de columnas que faltan separadas por comas
        )  # Cierra la excepción

    datos = datos.copy()  # Crea una copia para no modificar el DataFrame original
    datos["Date"] = pd.to_datetime(datos["Date"], errors="coerce")  # Convierte la columna Date a fecha; si falla pone NaT
    datos[TARGET] = pd.to_numeric(datos[TARGET], errors="coerce")  # Convierte el desperdicio a número; si falla pone NaN

    duplicados = int(datos.duplicated().sum())  # Cuenta cuántas filas están duplicadas
    datos = datos.drop_duplicates().dropna(subset=["Date", TARGET]).copy()  # Elimina duplicados y filas sin fecha o sin desperdicio

    for columna in datos.select_dtypes(include=["object"]).columns:  # Recorre todas las columnas de tipo texto
        datos[columna] = (  # Reemplaza la columna por su versión limpia
            datos[columna]  # Toma la columna actual
            .astype("string")  # La convierte al tipo texto de pandas
            .str.strip()  # Quita espacios al principio y al final
            .str.lower()  # Convierte todo a minúsculas
            .str.replace(r"\s+", " ", regex=True)  # Sustituye varios espacios seguidos por uno solo
        )  # Cierra la limpieza de texto

    datos["Season"] = datos["Date"].dt.month.map(obtener_temporada)  # Crea la columna Season a partir del mes de la fecha

    for columna in NUMERIC_COLUMNS:  # Recorre cada columna numérica
        datos[columna] = pd.to_numeric(datos[columna], errors="coerce")  # Convierte a número; los errores pasan a NaN
        medianas_categoria = datos.groupby("Food_Category")[columna].transform("median")  # Calcula la mediana de la columna por cada categoría de alimento
        datos[columna] = datos[columna].fillna(medianas_categoria)  # Rellena los vacíos con la mediana de su categoría
        datos[columna] = datos[columna].fillna(datos[columna].median())  # Los vacíos restantes se rellenan con la mediana global
        datos[columna] = datos[columna].fillna(0)  # Si aún quedan vacíos, se rellenan con 0

    for columna in CATEGORICAL_COLUMNS:  # Recorre cada columna categórica
        datos[columna] = datos[columna].fillna("desconocido").astype(str).str.lower()  # Rellena vacíos con "desconocido", convierte a texto y a minúsculas

    resumen = {  # Crea un diccionario con el resumen de la limpieza
        "filas_iniciales": filas_iniciales,  # Filas que había al principio
        "duplicados_eliminados": duplicados,  # Filas duplicadas detectadas
        "filas_finales": len(datos),  # Filas que quedan después de limpiar
    }  # Cierra el diccionario de resumen
    return datos, resumen  # Devuelve los datos limpios y el resumen


# ---------------------------------------------------------------------
# ENTRENAMIENTO Y EVALUACIÓN
# ---------------------------------------------------------------------

def entrenar_modelo(datos):  # Función que entrena el modelo con los datos limpios
    """Entrena un pipeline con preprocesamiento y Random Forest."""
    if len(datos) < 2:  # Si hay menos de dos filas
        raise ValueError("Se necesitan al menos dos filas válidas para entrenar.")  # Lanza error porque no se puede entrenar

    X = datos[CATEGORICAL_COLUMNS + NUMERIC_COLUMNS].copy()  # Variables de entrada: columnas categóricas y numéricas
    y = datos[TARGET].copy()  # Variable objetivo: cantidad de desperdicio

    preprocesador = ColumnTransformer(  # Crea el preprocesador de columnas
        transformers=[  # Lista de transformaciones a aplicar
            (  # Inicio de una transformación
                "cat",  # Nombre de la transformación
                OneHotEncoder(drop="first", handle_unknown="ignore"),  # Codifica categorías en 0/1, elimina la primera y ignora categorías nuevas
                CATEGORICAL_COLUMNS,  # Columnas a las que se aplica
            )  # Cierra la transformación
        ],  # Cierra la lista de transformaciones
        remainder="passthrough",  # Las columnas restantes (numéricas) pasan sin cambios
    )  # Cierra el ColumnTransformer

    pipeline = Pipeline(  # Crea el pipeline completo
        steps=[  # Lista de pasos del pipeline
            ("preprocesamiento", preprocesador),  # Paso 1: preprocesar los datos
            (  # Inicio del paso 2
                "regresor",  # Nombre del paso del modelo
                RandomForestRegressor(  # Modelo de bosque aleatorio para regresión
                    n_estimators=100,  # Usa 100 árboles
                    random_state=42,  # Semilla para que los resultados sean reproducibles
                    n_jobs=-1,  # Usa todos los núcleos del procesador
                ),  # Cierra el modelo
            ),  # Cierra el paso 2
        ]  # Cierra la lista de pasos
    )  # Cierra el Pipeline

    X_train, X_test, y_train, y_test = train_test_split(  # Divide los datos en entrenamiento y prueba
        X,  # Variables de entrada
        y,  # Variable objetivo
        test_size=0.2,  # 20% de los datos se reserva para prueba
        random_state=42,  # Semilla para que la división sea siempre igual
    )  # Cierra la división

    pipeline.fit(X_train, y_train)  # Entrena el pipeline con los datos de entrenamiento
    predicciones = pipeline.predict(X_test)  # Predice el desperdicio para los datos de prueba

    metricas = {  # Diccionario con las métricas de evaluación
        "MAE": mean_absolute_error(y_test, predicciones),  # Error absoluto medio
        "RMSE": np.sqrt(mean_squared_error(y_test, predicciones)),  # Raíz del error cuadrático medio
        "R2": r2_score(y_test, predicciones) if len(y_test) > 1 else float("nan"),  # Coeficiente R²; NaN si hay una sola muestra de prueba
    }  # Cierra el diccionario de métricas

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)  # Crea la carpeta del modelo si no existe
    joblib.dump(pipeline, MODEL_PATH)  # Guarda el modelo entrenado en disco
    return pipeline, metricas  # Devuelve el modelo y sus métricas


@st.cache_resource  # Guarda en caché el modelo para no cargarlo ni entrenarlo en cada interacción
def cargar_o_entrenar_modelo():  # Función que carga el modelo guardado o lo entrena
    """Carga el modelo guardado o lo entrena si aún no existe."""
    if MODEL_PATH.exists():  # Si el archivo del modelo ya existe
        return joblib.load(MODEL_PATH), None  # Lo carga y devuelve sin métricas (None)

    datos, _ = cargar_y_limpiar_datos()  # Si no existe, carga y limpia los datos (ignora el resumen)
    modelo, metricas = entrenar_modelo(datos)  # Entrena un modelo nuevo y obtiene sus métricas
    return modelo, metricas  # Devuelve el modelo y las métricas


# ---------------------------------------------------------------------
# VISUALIZACIONES
# ---------------------------------------------------------------------

def verificar_archivos_requeridos():  # Función que comprueba que existan los archivos necesarios
    """Comprueba que los recursos mínimos para ejecutar la app existan."""
    problemas = []  # Lista vacía donde se guardarán los avisos

    if not DATA_PATH.exists():  # Si el CSV de datos no existe
        problemas.append(  # Añade un mensaje de problema a la lista
            f"No se encontró el archivo de datos: {DATA_PATH.name}. "  # Indica qué archivo falta
            "Colócalo en la misma carpeta que este script."  # Indica dónde colocarlo
        )  # Cierra el append

    if not MODEL_PATH.exists():  # Si el archivo del modelo no existe
        problemas.append(  # Añade un mensaje informativo a la lista
            f"No se encontró el modelo previo: {MODEL_PATH.name}. "  # Indica qué modelo falta
            "Se generará automáticamente al iniciar si los datos están disponibles."  # Explica que se creará solo
        )  # Cierra el append

    return problemas  # Devuelve la lista de problemas encontrados


def mostrar_analisis(datos):  # Función que muestra los gráficos del análisis exploratorio
    st.header("Análisis exploratorio")  # Muestra un encabezado grande en la app

    st.subheader("Unidades vendidas por estación")  # Muestra un subtítulo
    ventas_temporada = (  # Calcula las ventas totales por estación
        datos.groupby("Season", as_index=False)["Units_Sold"]  # Agrupa por estación y toma las unidades vendidas
        .sum()  # Suma las unidades de cada estación
        .sort_values("Units_Sold", ascending=False)  # Ordena de mayor a menor
    )  # Cierra el cálculo
    fig, ax = plt.subplots(figsize=(8, 4))  # Crea una figura de 8x4 pulgadas con un eje
    sns.barplot(data=ventas_temporada, x="Season", y="Units_Sold", ax=ax)  # Dibuja un gráfico de barras de ventas por estación
    ax.set_title("Unidades vendidas por estación")  # Pone el título del gráfico
    ax.set_xlabel("Estación")  # Etiqueta del eje X
    ax.set_ylabel("Unidades vendidas")  # Etiqueta del eje Y
    st.pyplot(fig)  # Muestra la figura en la app de Streamlit
    plt.close(fig)  # Cierra la figura para liberar memoria

    st.subheader("Desperdicio por categoría de alimento")  # Muestra un subtítulo
    desperdicio_categoria = (  # Calcula el desperdicio total por categoría
        datos.groupby("Food_Category", as_index=False)[TARGET]  # Agrupa por categoría y toma el desperdicio
        .sum()  # Suma el desperdicio de cada categoría
        .sort_values(TARGET, ascending=False)  # Ordena de mayor a menor
    )  # Cierra el cálculo

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))  # Crea una figura con dos gráficos lado a lado
    sns.barplot(  # Dibuja un gráfico de barras en el primer eje
        data=desperdicio_categoria,  # Datos del gráfico
        x=TARGET,  # Eje X: desperdicio total
        y="Food_Category",  # Eje Y: categoría de alimento
        hue="Food_Category",  # Colorea cada barra según su categoría
        legend=False,  # Oculta la leyenda
        ax=axes[0],  # Lo dibuja en el primer eje
    )  # Cierra el barplot
    axes[0].set_title("Desperdicio total por categoría")  # Título del primer gráfico
    axes[0].set_xlabel("Desperdicio total")  # Etiqueta del eje X del primer gráfico
    axes[0].set_ylabel("Categoría")  # Etiqueta del eje Y del primer gráfico

    sns.boxplot(  # Dibuja un diagrama de caja en el segundo eje
        data=datos,  # Usa todos los datos
        x=TARGET,  # Eje X: cantidad de desperdicio
        y="Food_Category",  # Eje Y: categoría de alimento
        hue="Food_Category",  # Colorea cada caja según su categoría
        legend=False,  # Oculta la leyenda
        ax=axes[1],  # Lo dibuja en el segundo eje
    )  # Cierra el boxplot
    axes[1].set_title("Distribución del desperdicio por categoría")  # Título del segundo gráfico
    axes[1].set_xlabel("Cantidad de desperdicio")  # Etiqueta del eje X del segundo gráfico
    axes[1].set_ylabel("Categoría")  # Etiqueta del eje Y del segundo gráfico
    fig.tight_layout()  # Ajusta los espacios para que nada se solape
    st.pyplot(fig)  # Muestra la figura en la app
    plt.close(fig)  # Cierra la figura para liberar memoria

    st.subheader("Matriz de correlación")  # Muestra un subtítulo
    columnas_numericas = datos.select_dtypes(include=[np.number]).columns  # Obtiene los nombres de las columnas numéricas
    matriz_corr = datos[columnas_numericas].corr()  # Calcula la matriz de correlación entre ellas

    fig, ax = plt.subplots(figsize=(10, 7))  # Crea una figura de 10x7 pulgadas
    sns.heatmap(  # Dibuja un mapa de calor
        matriz_corr,  # Datos: la matriz de correlación
        annot=True,  # Muestra los valores dentro de cada celda
        cmap="coolwarm",  # Paleta de colores de frío a calor
        fmt=".2f",  # Formato con dos decimales
        linewidths=0.5,  # Grosor de las líneas entre celdas
        vmin=-1,  # Valor mínimo de la escala de color
        vmax=1,  # Valor máximo de la escala de color
        ax=ax,  # Lo dibuja en el eje creado
    )  # Cierra el heatmap
    ax.set_title("Correlación entre variables numéricas")  # Título del gráfico
    fig.tight_layout()  # Ajusta los espacios de la figura
    st.pyplot(fig)  # Muestra la figura en la app
    plt.close(fig)  # Cierra la figura para liberar memoria

    st.subheader("Correlación por estación")  # Muestra un subtítulo
    estaciones = list(datos["Season"].dropna().unique())  # Lista de estaciones distintas presentes en los datos
    if estaciones:  # Si hay al menos una estación
        fig, axes = plt.subplots(2, 2, figsize=(14, 11))  # Crea una cuadrícula de 2x2 gráficos
        axes = axes.flatten()  # Convierte la cuadrícula en una lista simple de ejes

        for indice, estacion in enumerate(estaciones[:4]):  # Recorre hasta 4 estaciones con su índice
            datos_estacion = datos[datos["Season"] == estacion]  # Filtra los datos de esa estación
            corr_estacion = datos_estacion.select_dtypes(include=[np.number]).corr()  # Calcula la correlación de las columnas numéricas

            sns.heatmap(  # Dibuja el mapa de calor de esa estación
                corr_estacion,  # Datos: la matriz de correlación de la estación
                annot=True,  # Muestra los valores en las celdas
                cmap="coolwarm",  # Paleta de colores
                fmt=".2f",  # Formato con dos decimales
                linewidths=0.5,  # Grosor de las líneas entre celdas
                vmin=-1,  # Valor mínimo de la escala
                vmax=1,  # Valor máximo de la escala
                ax=axes[indice],  # Eje donde se dibuja
                cbar=(indice == min(len(estaciones), 4) - 1),  # Muestra la barra de color solo en el último gráfico
            )  # Cierra el heatmap
            axes[indice].set_title(f"Estación: {estacion.capitalize()}")  # Título con el nombre de la estación en mayúscula inicial

        for indice in range(min(len(estaciones), 4), 4):  # Recorre los ejes que sobran (si hay menos de 4 estaciones)
            axes[indice].set_visible(False)  # Oculta esos ejes vacíos

        fig.tight_layout()  # Ajusta los espacios de la figura
        st.pyplot(fig)  # Muestra la figura en la app
        plt.close(fig)  # Cierra la figura para liberar memoria


# ---------------------------------------------------------------------
# APLICACIÓN STREAMLIT
# ---------------------------------------------------------------------

def main():  # Función principal que construye la aplicación
    st.set_page_config(  # Configura la página de Streamlit
        page_title="El Rescatista de Alimentos",  # Título que aparece en la pestaña del navegador
        page_icon="🍏",  # Icono de la pestaña
        layout="wide",  # Usa todo el ancho de la pantalla
    )  # Cierra la configuración

    st.title("🍏 El Rescatista de Alimentos")  # Muestra el título principal de la app
    st.write(  # Escribe un texto descriptivo en la app
        "Herramienta predictiva de desperdicio alimentario "  # Primera parte de la descripción
        "basada en aprendizaje automático."  # Segunda parte de la descripción
    )  # Cierra el write

    problemas = verificar_archivos_requeridos()  # Comprueba si faltan archivos
    for problema in problemas:  # Recorre cada problema encontrado
        if "modelo previo" in problema:  # Si el aviso trata del modelo previo
            st.info(problema)  # Lo muestra como información (no es grave)
        else:  # Si es otro problema (falta el CSV)
            st.error(problema)  # Lo muestra como error

    try:  # Intenta cargar y limpiar los datos
        datos, resumen = cargar_y_limpiar_datos()  # Obtiene los datos limpios y el resumen
    except FileNotFoundError as error:  # Si no se encuentra el archivo
        st.error(f"No se pudo abrir el archivo necesario: {error}")  # Muestra el error
        st.info("Coloca el CSV en la misma carpeta que este script y vuelve a intentarlo.")  # Da una indicación
        st.stop()  # Detiene la ejecución de la app
    except ValueError as error:  # Si faltan columnas o hay datos inválidos
        st.error(f"La estructura del archivo no es válida: {error}")  # Muestra el error
        st.info("Revisa las columnas requeridas del dataset y vuelve a intentarlo.")  # Da una indicación
        st.stop()  # Detiene la ejecución de la app
    except Exception as error:  # Cualquier otro error inesperado
        st.error(f"No se pudo iniciar la aplicación: {error}")  # Muestra el error
        st.stop()  # Detiene la ejecución de la app

    try:  # Intenta cargar o entrenar el modelo
        modelo, metricas_entrenamiento = cargar_o_entrenar_modelo()  # Obtiene el modelo y sus métricas (si se entrenó ahora)
    except Exception as error:  # Si ocurre cualquier error
        st.error(f"No se pudo cargar ni entrenar el modelo: {error}")  # Muestra el error
        st.stop()  # Detiene la ejecución de la app

    with st.expander("Resumen de la limpieza de datos"):  # Crea una sección desplegable
        st.write(f"Filas iniciales: {resumen['filas_iniciales']}")  # Muestra las filas iniciales
        st.write(f"Duplicados detectados/eliminados: {resumen['duplicados_eliminados']}")  # Muestra los duplicados eliminados
        st.write(f"Filas finales: {resumen['filas_finales']}")  # Muestra las filas finales
        st.write("Valores nulos restantes:")  # Texto introductorio de la tabla
        st.dataframe(datos.isnull().sum().rename("Nulos"))  # Muestra una tabla con los nulos por columna

    if metricas_entrenamiento is not None:  # Si el modelo se acaba de entrenar (hay métricas)
        st.subheader("Evaluación del modelo")  # Muestra un subtítulo
        col1, col2, col3 = st.columns(3)  # Crea tres columnas en la pantalla
        col1.metric("MAE", f"{metricas_entrenamiento['MAE']:.4f}")  # Muestra el MAE con 4 decimales
        col2.metric("RMSE", f"{metricas_entrenamiento['RMSE']:.4f}")  # Muestra el RMSE con 4 decimales
        col3.metric("R²", f"{metricas_entrenamiento['R2']:.4f}")  # Muestra el R² con 4 decimales

    mostrar_analisis(datos)  # Muestra todos los gráficos directamente en la página, sin sección desplegable

    st.markdown("---")  # Dibuja una línea horizontal separadora
    st.header("Estimación del desperdicio de un día")  # Encabezado de la sección de predicción
    st.write("Introduce las características del negocio y del inventario.")  # Instrucción para el usuario

    valores_unicos = {  # Diccionario con las opciones disponibles de cada columna categórica
        columna: sorted(datos[columna].dropna().astype(str).unique().tolist())  # Valores únicos, sin vacíos, como texto y ordenados
        for columna in CATEGORICAL_COLUMNS  # Para cada columna categórica
    }  # Cierra el diccionario

    col1, col2 = st.columns(2)  # Crea dos columnas para organizar los campos del formulario

    with col1:  # Todo lo siguiente se coloca en la primera columna
        business_type = st.selectbox(  # Desplegable para elegir el tipo de negocio
            "Tipo de negocio",  # Etiqueta del campo
            valores_unicos["Business_Type"],  # Opciones disponibles
        )  # Cierra el selectbox
        food_category = st.selectbox(  # Desplegable para elegir la categoría de alimento
            "Categoría de alimento",  # Etiqueta del campo
            valores_unicos["Food_Category"],  # Opciones disponibles
        )  # Cierra el selectbox
        season = st.selectbox(  # Desplegable para elegir la estación del año
            "Estación del año",  # Etiqueta del campo
            ["invierno", "primavera", "verano", "otoño"],  # Opciones fijas
        )  # Cierra el selectbox
        promotion = st.selectbox(  # Desplegable para indicar si hay promoción
            "¿Hay una promoción activa?",  # Etiqueta del campo
            valores_unicos["Promotion"],  # Opciones disponibles
        )  # Cierra el selectbox
        weather = st.selectbox(  # Desplegable para elegir el clima
            "Condiciones meteorológicas",  # Etiqueta del campo
            valores_unicos["Weather"],  # Opciones disponibles
        )  # Cierra el selectbox

    with col2:  # Todo lo siguiente se coloca en la segunda columna
        purchase_quantity = st.number_input(  # Campo numérico para la cantidad comprada
            "Cantidad comprada",  # Etiqueta del campo
            min_value=0.0,  # Valor mínimo permitido
            max_value=100000.0,  # Valor máximo permitido
            value=100.0,  # Valor inicial por defecto
        )  # Cierra el number_input
        units_sold = st.number_input(  # Campo numérico para las unidades vendidas
            "Unidades vendidas",  # Etiqueta del campo
            min_value=0.0,  # Valor mínimo permitido
            max_value=100000.0,  # Valor máximo permitido
            value=80.0,  # Valor inicial por defecto
        )  # Cierra el number_input
        remaining_stock = st.number_input(  # Campo numérico para el inventario restante
            "Inventario restante",  # Etiqueta del campo
            min_value=0.0,  # Valor mínimo permitido
            max_value=100000.0,  # Valor máximo permitido
            value=20.0,  # Valor inicial por defecto
        )  # Cierra el number_input
        shelf_life = st.number_input(  # Campo numérico para la vida útil
            "Vida útil (días)",  # Etiqueta del campo
            min_value=0.0,  # Valor mínimo permitido
            max_value=365.0,  # Valor máximo permitido (un año)
            value=5.0,  # Valor inicial por defecto
        )  # Cierra el number_input
        temperature = st.number_input(  # Campo numérico para la temperatura de almacenamiento
            "Temperatura de almacenamiento (°C)",  # Etiqueta del campo
            min_value=-50.0,  # Temperatura mínima permitida
            max_value=60.0,  # Temperatura máxima permitida
            value=4.0,  # Valor inicial por defecto
        )  # Cierra el number_input
        daily_demand = st.number_input(  # Campo numérico para la demanda diaria estimada
            "Demanda diaria estimada",  # Etiqueta del campo
            min_value=0.0,  # Valor mínimo permitido
            max_value=100000.0,  # Valor máximo permitido
            value=85.0,  # Valor inicial por defecto
        )  # Cierra el number_input

    if st.button("Calcular estimación de desperdicio", type="primary"):  # Si el usuario pulsa el botón principal
        datos_entrada = pd.DataFrame(  # Crea un DataFrame con los datos introducidos
            [  # Lista con una sola fila
                {  # Diccionario con los valores de esa fila
                    "Business_Type": business_type,  # Tipo de negocio elegido
                    "Food_Category": food_category,  # Categoría de alimento elegida
                    "Season": season,  # Estación elegida
                    "Promotion": promotion,  # Promoción elegida
                    "Weather": weather,  # Clima elegido
                    "Purchase_Quantity": purchase_quantity,  # Cantidad comprada introducida
                    "Units_Sold": units_sold,  # Unidades vendidas introducidas
                    "Remaining_Stock": remaining_stock,  # Inventario restante introducido
                    "Shelf_Life_Days": shelf_life,  # Vida útil introducida
                    "Storage_Temperature": temperature,  # Temperatura introducida
                    "Daily_Demand": daily_demand,  # Demanda diaria introducida
                }  # Cierra el diccionario de la fila
            ]  # Cierra la lista de filas
        )  # Cierra el DataFrame

        prediccion = float(modelo.predict(datos_entrada)[0])  # Predice el desperdicio y toma el primer (único) resultado como número decimal
        prediccion = max(0.0, prediccion)  # Evita valores negativos: el mínimo es 0

        st.success(f"Desperdicio estimado para hoy: **{prediccion:.2f} unidades**")  # Muestra el resultado con 2 decimales en un cuadro verde

        if prediccion > 20:  # Si el desperdicio estimado supera 20 unidades
            st.warning(  # Muestra un aviso amarillo
                "El desperdicio estimado es elevado. "  # Primera parte del aviso
                "Considera ajustar las compras, aplicar ofertas o facilitar donaciones."  # Recomendaciones
            )  # Cierra el warning
        else:  # Si el desperdicio es 20 o menos
            st.info(  # Muestra un mensaje informativo
                "El desperdicio estimado es relativamente bajo "  # Primera parte del mensaje
                "para los parámetros introducidos."  # Segunda parte del mensaje
            )  # Cierra el info


if __name__ == "__main__":  # Comprueba que el script se ejecuta directamente (no importado)
    main()  # Llama a la función principal para iniciar la app
