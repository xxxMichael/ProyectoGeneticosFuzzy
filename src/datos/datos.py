"""
Módulo Integrador de Carga y Preparación de Datos
Proyecto: Clasificación de Calidad de Vinos (PRISM, Apriori, Lógica Difusa y Algoritmo Genético)
"""

from typing import Any, Dict
import pandas as pd

from src.datos.carga.cargador import (
    DIRECTORIO_BASE,
    ENCODING_CSV,
    RUTA_DATASET_DEFAULT,
    SEPARADOR_CSV,
    cargar_dataset,
    mostrar_resumen_carga,
    verificar_integridad,
)
from src.datos.preparacion.discretizacion import (
    ETIQUETAS_INTERVALOS,
    MAPEO_CALIDAD,
    NOMBRE_CALIDAD_CATEGORICA,
    NOMBRE_VARIABLE_OBJETIVO,
    NUMERO_INTERVALOS,
    PORCENTAJE_TEST,
    SEED_ALEATORIA,
    VARIABLES_FISICOQUIMICAS,
    calcular_estadisticas_descriptivas,
    categorizar_calidad,
    discretizar_por_cuantiles,
    preparar_conjuntos_entrenamiento_prueba,
    separar_caracteristicas_objetivo,
)


def ejecutar_pipeline_datos() -> Dict[str, Any]:
    """
    Ejecuta el flujo completo de carga, verificación y preparación de datos.
    """
    print("\n" + "=" * 80)
    print(" [DATOS] CARGA, VERIFICACIÓN Y PREPARACIÓN DEL CONJUNTO DE DATOS")
    print("=" * 80)

    df = cargar_dataset()
    mostrar_resumen_carga(df)

    particiones = preparar_conjuntos_entrenamiento_prueba(df)
    print(f" * Instancias en Train : {len(particiones['X_train_num'])}")
    print(f" * Instancias en Test  : {len(particiones['X_test_num'])}")
    print(f" * Variables Continuas : {len(VARIABLES_FISICOQUIMICAS)}")
    print("=" * 80 + "\n")

    return particiones


if __name__ == "__main__":
    ejecutar_pipeline_datos()
