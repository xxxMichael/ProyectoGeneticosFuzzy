"""
Paquete del Algoritmo PRISM
"""

from src.prism.reglas.regla import ReglaPRISM
from src.prism.induccion.selector import buscar_mejor_termino_voraz
from src.prism.prism import (
    AlgoritmoPRISM,
    CLASES_OBJETIVO,
    CLASE_POR_DEFECTO,
    MAX_CONDICIONES_POR_REGLA,
    MIN_INSTANCIAS_CUBIERTAS,
    MIN_PRECISION_REGLA,
    RUTA_SALIDA_REGLAS,
    ejecutar_induccion_prism,
)

__all__ = [
    "ReglaPRISM",
    "buscar_mejor_termino_voraz",
    "AlgoritmoPRISM",
    "MIN_PRECISION_REGLA",
    "MIN_INSTANCIAS_CUBIERTAS",
    "MAX_CONDICIONES_POR_REGLA",
    "CLASES_OBJETIVO",
    "CLASE_POR_DEFECTO",
    "RUTA_SALIDA_REGLAS",
    "ejecutar_induccion_prism",
]
