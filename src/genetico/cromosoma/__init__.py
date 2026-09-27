"""
Submódulo de Cromosoma y Reparación
"""

from src.genetico.cromosoma.estructura import (
    LONGITUD_CROMOSOMA_MFS,
    LONGITUD_CROMOSOMA_REGLAS,
    LONGITUD_TOTAL_CROMOSOMA,
    UMBRAL_ACTIVACION_REGLA,
    decodificar_cromosoma,
)
from src.genetico.cromosoma.reparacion import reparar_individuo

__all__ = [
    "LONGITUD_CROMOSOMA_MFS",
    "LONGITUD_CROMOSOMA_REGLAS",
    "LONGITUD_TOTAL_CROMOSOMA",
    "UMBRAL_ACTIVACION_REGLA",
    "decodificar_cromosoma",
    "reparar_individuo",
]
