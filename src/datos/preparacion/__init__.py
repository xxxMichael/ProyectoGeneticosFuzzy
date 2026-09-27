"""
Submódulo de Preparación de Datos y Discretización
"""

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
]
