"""
Módulo de Preparación, Separación y Análisis Estadístico de Datos
Proyecto: Clasificación de Calidad de Vinos (PRISM, Apriori, Lógica Difusa y Algoritmo Genético)
Submódulo: Discretización y Particionado
"""

import os
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.datos.carga.cargador import cargar_dataset

# =============================================================================
# CONFIGURACIÓN Y PARÁMETROS DE PREPARACIÓN
# =============================================================================
NOMBRE_VARIABLE_OBJETIVO: str = "quality"
NOMBRE_CALIDAD_CATEGORICA: str = "calidad_categoria"

# Lista oficial de las 11 variables fisicoquímicas predictoras
VARIABLES_FISICOQUIMICAS: List[str] = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]

# Reglas de agrupación para la variable objetivo según especificación
MAPEO_CALIDAD: Dict[int, str] = {
    3: "Baja",
    4: "Baja",
    5: "Baja",
    6: "Media",
    7: "Alta",
    8: "Alta",
}

# Parámetros de partición y discretización
PORCENTAJE_TEST: float = 0.20           # 20% para evaluación en conjunto de prueba
SEED_ALEATORIA: int = 42                # Semilla para reproducibilidad
NUMERO_INTERVALOS: int = 3              # 3 rangos por variable (Bajo, Medio, Alto)
ETIQUETAS_INTERVALOS: List[str] = ["Bajo", "Medio", "Alto"]
# =============================================================================


def separar_caracteristicas_objetivo(
    dataframe: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separa las 11 características fisicoquímicas de la variable objetivo 'quality'.

    Args:
        dataframe (pd.DataFrame): Dataset completo.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: (X con variables predictoras, y con calidad original).
    """
    columnas_disponibles = [col for col in VARIABLES_FISICOQUIMICAS if col in dataframe.columns]
    caracteristicas_x = dataframe[columnas_disponibles].copy()
    objetivo_y = dataframe[NOMBRE_VARIABLE_OBJETIVO].copy()
    return caracteristicas_x, objetivo_y


def calcular_estadisticas_descriptivas(dataframe_x: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula un resumen estadístico detallado de las 11 variables fisicoquímicas.

    Args:
        dataframe_x (pd.DataFrame): DataFrame con las características continuas.

    Returns:
        pd.DataFrame: Tabla resumen con min, p25, mediana (p50), p75, max, media, std y asimetría.
    """
    resumen = pd.DataFrame(index=dataframe_x.columns)
    resumen["Min"] = dataframe_x.min()
    resumen["P25 (Q1)"] = dataframe_x.quantile(0.25)
    resumen["Mediana (P50)"] = dataframe_x.median()
    resumen["P75 (Q3)"] = dataframe_x.quantile(0.75)
    resumen["Max"] = dataframe_x.max()
    resumen["Media"] = dataframe_x.mean()
    resumen["Desv. Est."] = dataframe_x.std()
    resumen["Asimetria"] = dataframe_x.skew()
    return resumen


def categorizar_calidad(serie_calidad: pd.Series) -> pd.Series:
    """
    Transforma la puntuación discreta (3 a 8) en categorías: 'Baja', 'Media', 'Alta'.

    Args:
        serie_calidad (pd.Series): Serie numérica original con la calidad del vino.

    Returns:
        pd.Series: Serie categorizada con valores ['Baja', 'Media', 'Alta'].
    """
    serie_categorizada = serie_calidad.map(MAPEO_CALIDAD)
    serie_categorizada.name = NOMBRE_CALIDAD_CATEGORICA
    return serie_categorizada


def discretizar_por_cuantiles(
    dataframe_x: pd.DataFrame,
    puntos_corte_precalculados: Optional[Dict[str, List[float]]] = None,
) -> Tuple[pd.DataFrame, Dict[str, List[float]]]:
    """
    Discretiza variables continuas en categorías de igual frecuencia (Equal-Frequency: 33% Bajo, Medio, Alto).

    Args:
        dataframe_x (pd.DataFrame): Características continuas a discretizar.
        puntos_corte_precalculados (Optional[Dict]): Cortes calculados en entrenamiento si se aplica a test.

    Returns:
        Tuple[pd.DataFrame, Dict[str, List[float]]]: (DataFrame discretizado, Diccionario de puntos de corte por variable).
    """
    dataframe_discreto = pd.DataFrame(index=dataframe_x.index)
    cortes_variables: Dict[str, List[float]] = {}

    for columna in dataframe_x.columns:
        valores = dataframe_x[columna]
        if puntos_corte_precalculados is None or columna not in puntos_corte_precalculados:
            q33 = float(valores.quantile(1.0 / 3.0))
            q66 = float(valores.quantile(2.0 / 3.0))
            bins = [-np.inf, q33, q66, np.inf]
            cortes_variables[columna] = [float(valores.min()), q33, q66, float(valores.max())]
        else:
            puntos = puntos_corte_precalculados[columna]
            bins = [-np.inf, puntos[1], puntos[2], np.inf]
            cortes_variables[columna] = puntos

        dataframe_discreto[columna] = pd.cut(
            valores,
            bins=bins,
            labels=ETIQUETAS_INTERVALOS,
            include_lowest=True,
        ).astype(str)

    return dataframe_discreto, cortes_variables


def preparar_conjuntos_entrenamiento_prueba(
    dataframe: Optional[pd.DataFrame] = None,
    test_size: float = PORCENTAJE_TEST,
    seed: int = SEED_ALEATORIA,
) -> Dict[str, Any]:
    """
    Ejecuta el pipeline completo de preparación de datos generando conjuntos numéricos
    y discretizados para Train y Test, manteniendo ambas representaciones.

    Args:
        dataframe (Optional[pd.DataFrame]): Dataset original. Si es None, se carga automáticamente.
        test_size (float): Proporción de datos para el conjunto de test.
        seed (int): Semilla para reproducibilidad del particionado.

    Returns:
        Dict[str, Any]: Diccionario con conjuntos Train/Test numéricos y discretizados.
    """
    if dataframe is None:
        dataframe = cargar_dataset()

    X_num, y_num = separar_caracteristicas_objetivo(dataframe)
    y_cat = categorizar_calidad(y_num)

    X_train_num, X_test_num, y_train_cat, y_test_cat = train_test_split(
        X_num,
        y_cat,
        test_size=test_size,
        random_state=seed,
        stratify=y_cat,
    )

    X_train_disc, cortes_entrenamiento = discretizar_por_cuantiles(X_train_num)
    X_test_disc, _ = discretizar_por_cuantiles(
        X_test_num, puntos_corte_precalculados=cortes_entrenamiento
    )

    return {
        "X_train_num": X_train_num,
        "X_test_num": X_test_num,
        "X_train_disc": X_train_disc,
        "X_test_disc": X_test_disc,
        "y_train": y_train_cat,
        "y_test": y_test_cat,
        "y_train_cat": y_train_cat,
        "y_test_cat": y_test_cat,
        "cortes_discretizacion": cortes_entrenamiento,
        "puntos_corte": cortes_entrenamiento,
    }


def mostrar_resumen_preparacion(dataframe: pd.DataFrame) -> None:
    """
    Imprime en consola un informe detallado sobre la separación, estadísticas descriptivas
    y balance de clases del dataset.
    """
    X_num, y_num = separar_caracteristicas_objetivo(dataframe)
    y_cat = categorizar_calidad(y_num)
    estadisticas = calcular_estadisticas_descriptivas(X_num)

    print("\n" + "=" * 95)
    print(" [FASE 2] RESUMEN DE ESTADISTICAS DESCRIPTIVAS (11 VARIABLES FISICOQUIMICAS)")
    print("=" * 95)
    print(
        f"{'Variable':<22} | {'Min':<7} | {'Q1(25%)':<7} | {'Mediana':<7} | {'Q3(75%)':<7} | {'Max':<7} | {'Media':<7} | {'Asimetria':<7}"
    )
    print("-" * 95)
    for variable, fila in estadisticas.iterrows():
        print(
            f"{variable:<22} | {fila['Min']:<7.2f} | {fila['P25 (Q1)']:<7.2f} | {fila['Mediana (P50)']:<7.2f} | {fila['P75 (Q3)']:<7.2f} | {fila['Max']:<7.2f} | {fila['Media']:<7.2f} | {fila['Asimetria']:<7.2f}"
        )
    print("=" * 95)

    print("\n" + "=" * 95)
    print(" DISTRIBUCION DE CLASES DE CALIDAD (VARIABLE OBJETIVO TRANSFORMADA)")
    print("=" * 95)
    distribucion = y_cat.value_counts()
    porcentajes = y_cat.value_counts(normalize=True) * 100
    for clase in ["Baja", "Media", "Alta"]:
        conteo = distribucion.get(clase, 0)
        pct = porcentajes.get(clase, 0.0)
        print(f" * Calidad {clase:<6} : {conteo:>4} muestras ({pct:>5.2f}%)")
    print("=" * 95 + "\n")
