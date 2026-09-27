"""
Submódulo de Estructura y Decodificación del Cromosoma Mixto
Proyecto: Clasificación de Calidad de Vinos (PRISM, Apriori, Lógica Difusa y Algoritmo Genético)
"""

import copy
from typing import List, Tuple
from src.fuzzy.pertenencia.funciones import ConfiguracionMFs
from src.reglas.regla_unificada import ReglaUnificada

# Estructura del cromosoma mixto
LONGITUD_CROMOSOMA_MFS: int = 33      # 11 variables x 3 puntos de corte (a, b, c)
LONGITUD_CROMOSOMA_REGLAS: int = 26   # 26 reglas unificadas (pesos en [0.0, 1.0])
LONGITUD_TOTAL_CROMOSOMA: int = LONGITUD_CROMOSOMA_MFS + LONGITUD_CROMOSOMA_REGLAS
UMBRAL_ACTIVACION_REGLA: float = 0.05


def decodificar_cromosoma(
    individuo: List[float],
    reglas_base: List[ReglaUnificada],
    umbral_activacion: float = UMBRAL_ACTIVACION_REGLA,
) -> Tuple[ConfiguracionMFs, List[ReglaUnificada]]:
    """
    Decodifica el cromosoma mixto en una ConfiguracionMFs y una lista de ReglaUnificada con pesos actualizados.
    """
    genes_mfs = individuo[:LONGITUD_CROMOSOMA_MFS]
    genes_reglas = individuo[LONGITUD_CROMOSOMA_MFS:]

    config_mfs = ConfiguracionMFs.desde_cromosoma(genes_mfs)

    reglas_decodificadas: List[ReglaUnificada] = []
    for idx, r_base in enumerate(reglas_base):
        peso = float(genes_reglas[idx])
        activa = bool(peso >= umbral_activacion)
        regla_clon = copy.deepcopy(r_base)
        regla_clon.peso = peso
        regla_clon.activa = activa
        reglas_decodificadas.append(regla_clon)

    return config_mfs, reglas_decodificadas
