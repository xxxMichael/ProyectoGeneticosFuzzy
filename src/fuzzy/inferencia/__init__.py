"""
Submódulo de Inferencia Difusa
"""

from src.fuzzy.inferencia.motor import (
    UMBRAL_ACTIVACION_MINIMA,
    agregar_salida_difusa,
    evaluar_activaciones_reglas,
)

__all__ = [
    "UMBRAL_ACTIVACION_MINIMA",
    "evaluar_activaciones_reglas",
    "agregar_salida_difusa",
]
