"""
Módulo de Carga y Exploración Inicial del Dataset
Proyecto: Clasificación de Calidad de Vinos (PRISM, Apriori, Lógica Difusa y Algoritmo Genético)
Submódulo: Carga de Datos
"""

import os
from typing import Optional, Tuple
import pandas as pd

# =============================================================================
# CONFIGURACIÓN Y PARÁMETROS DE CARGA
# =============================================================================
DIRECTORIO_BASE: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RUTA_DATASET_DEFAULT: str = os.path.join(DIRECTORIO_BASE, "wine+quality", "winequality-red.csv")
SEPARADOR_CSV: str = ";"  # El dataset original de UCI Wine Quality utiliza punto y coma como delimitador
ENCODING_CSV: str = "utf-8"
# =============================================================================


def cargar_dataset(ruta_archivo: Optional[str] = None) -> pd.DataFrame:
    """
    Carga el dataset de vinos tintos desde un archivo CSV.

    Args:
        ruta_archivo (Optional[str]): Ruta personalizada al archivo CSV. Si es None,
                                      utiliza la ruta configurada por defecto.

    Returns:
        pd.DataFrame: DataFrame con los datos cargados.

    Raises:
        FileNotFoundError: Si el archivo no existe en la ruta indicada.
    """
    ruta_final = ruta_archivo if ruta_archivo is not None else RUTA_DATASET_DEFAULT

    if not os.path.exists(ruta_final):
        # Intento de búsqueda alternativa en carpeta data/
        ruta_alternativa = os.path.join(DIRECTORIO_BASE, "data", "winequality-red.csv")
        if os.path.exists(ruta_alternativa):
            ruta_final = ruta_alternativa
        else:
            raise FileNotFoundError(
                f"No se encontró el archivo de datos en: '{ruta_final}' ni en '{ruta_alternativa}'."
            )

    dataframe = pd.read_csv(ruta_final, sep=SEPARADOR_CSV, encoding=ENCODING_CSV)
    return dataframe


def verificar_integridad(dataframe: pd.DataFrame) -> Tuple[int, int, int]:
    """
    Comprueba las dimensiones, cantidad de valores nulos y registros duplicados en el DataFrame.

    Args:
        dataframe (pd.DataFrame): Dataset cargado.

    Returns:
        Tuple[int, int, int]: Tupla con (total_filas, total_columnas, total_duplicados).
    """
    total_filas, total_columnas = dataframe.shape
    total_nulos = int(dataframe.isnull().sum().sum())
    total_duplicados = int(dataframe.duplicated().sum())

    return total_filas, total_columnas, total_duplicados


def mostrar_resumen_carga(dataframe: pd.DataFrame) -> None:
    """
    Imprime en consola un informe visual y legible sobre la estructura del dataset cargado.

    Args:
        dataframe (pd.DataFrame): Dataset a resumir.
    """
    total_filas, total_columnas, total_duplicados = verificar_integridad(dataframe)
    nulos_por_columna = dataframe.isnull().sum()

    print("\n" + "=" * 75)
    print(" [FASE 1] INFORME DE CARGA Y EXPLORACION INICIAL DEL DATASET")
    print("=" * 75)
    print(f" * Total de Registros (Filas)    : {total_filas:,}")
    print(f" * Total de Variables (Columnas) : {total_columnas}")
    print(f" * Registros Duplicados          : {total_duplicados} ({(total_duplicados / total_filas) * 100:.2f}%)")
    print(f" * Valores Nulos Globales        : {dataframe.isnull().sum().sum()}")
    print("-" * 75)
    print(" DETALLE DE VARIABLES Y VALORES FALTANTES:")
    for columna, nulos in nulos_por_columna.items():
        tipo_dato = dataframe[columna].dtype
        print(f"   - {columna:<22} | Tipo: {str(tipo_dato):<8} | Nulos: {nulos}")
    print("=" * 75 + "\n")
