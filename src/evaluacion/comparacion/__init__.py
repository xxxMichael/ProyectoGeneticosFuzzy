"""
Submódulo de Comparación de Fases
"""

from src.evaluacion.comparacion.comparador import (
    UMBRAL_PESO_REGLA_ACTIVA,
    construir_dataframe_comparativo,
    evaluar_fase_1_discreto,
    evaluar_fase_2_pre_ga,
    evaluar_fase_3_post_ga,
)

__all__ = [
    "UMBRAL_PESO_REGLA_ACTIVA",
    "evaluar_fase_1_discreto",
    "evaluar_fase_2_pre_ga",
    "evaluar_fase_3_post_ga",
    "construir_dataframe_comparativo",
]
