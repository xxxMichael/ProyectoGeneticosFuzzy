"""
Submódulo de Evaluación de Aptitud (Fitness Multi-criterio)
Proyecto: Clasificación de Calidad de Vinos (PRISM, Apriori, Lógica Difusa y Algoritmo Genético)
"""

from typing import List, Tuple
import numpy as np

from src.fuzzy.fuzzy import ClasificadorFuzzyMamdani
from src.genetico.cromosoma.estructura import (
    UMBRAL_ACTIVACION_REGLA,
    decodificar_cromosoma,
)
from src.reglas.regla_unificada import ReglaUnificada

# Ponderaciones por defecto de la función de aptitud
PESO_F1_MACRO: float = 0.70
PESO_COBERTURA: float = 0.20
PESO_PENALIZACION_REGLAS: float = 0.10


def evaluar_individuo_fitness(
    individuo: List[float],
    reglas_base: List[ReglaUnificada],
    X_matriz: np.ndarray,
    y_vector: np.ndarray,
    peso_f1: float = PESO_F1_MACRO,
    peso_cobertura: float = PESO_COBERTURA,
    peso_penalizacion: float = PESO_PENALIZACION_REGLAS,
    umbral_activacion: float = UMBRAL_ACTIVACION_REGLA,
) -> Tuple[float]:
    """
    Función de Fitness Multi-criterio:
    Fitness = 0.70 * F1_Macro + 0.20 * Cobertura - 0.10 * (Reglas_Activas / Reglas_Totales)
    """
    config_mfs, reglas_eval = decodificar_cromosoma(individuo, reglas_base, umbral_activacion)

    clasificador = ClasificadorFuzzyMamdani(
        config_mfs=config_mfs,
        reglas=reglas_eval,
        umbral_peso_activa=umbral_activacion,
    )

    metricas = clasificador.evaluar(X_matriz, y_vector)

    f1_macro = metricas["f1_macro"]
    cobertura = metricas["cobertura"]
    reglas_activas = metricas["reglas_activas"]
    total_reglas = len(reglas_base)

    tasa_reglas_activas = float(reglas_activas / total_reglas) if total_reglas > 0 else 0.0
    fitness = (peso_f1 * f1_macro) + (peso_cobertura * cobertura) - (peso_penalizacion * tasa_reglas_activas)

    return (float(fitness),)
