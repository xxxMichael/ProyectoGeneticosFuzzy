"""
Módulo de Preparación, Separación y Análisis Estadístico de Datos (Shim de Compatibilidad hacia src.datos)
Proyecto: Clasificación de Calidad de Vinos (PRISM, Apriori, Lógica Difusa y Algoritmo Genético)
"""

from src.datos import (
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
    mostrar_resumen_preparacion,
    preparar_conjuntos_entrenamiento_prueba,
    separar_caracteristicas_objetivo,
)

__all__ = [
    "NOMBRE_VARIABLE_OBJETIVO",
    "NOMBRE_CALIDAD_CATEGORICA",
    "VARIABLES_FISICOQUIMICAS",
    "MAPEO_CALIDAD",
    "PORCENTAJE_TEST",
    "SEED_ALEATORIA",
    "NUMERO_INTERVALOS",
    "ETIQUETAS_INTERVALOS",
    "separar_caracteristicas_objetivo",
    "calcular_estadisticas_descriptivas",
    "categorizar_calidad",
    "discretizar_por_cuantiles",
    "preparar_conjuntos_entrenamiento_prueba",
    "mostrar_resumen_preparacion",
]

if __name__ == "__main__":
    from src.datos import cargar_dataset
    df = cargar_dataset()
    mostrar_resumen_preparacion(df)
