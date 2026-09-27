"""
Submódulo de Reparación y Factibilidad Estricta de Cromosomas
Proyecto: Clasificación de Calidad de Vinos (PRISM, Apriori, Lógica Difusa y Algoritmo Genético)
"""

from typing import List
import numpy as np

from src.datos.preparacion.discretizacion import VARIABLES_FISICOQUIMICAS
from src.fuzzy.pertenencia.funciones import RANGOS_VARIABLES_DEFAULT

LONGITUD_CROMOSOMA_MFS: int = 33      # 11 variables x 3 puntos de corte (a, b, c)


def reparar_individuo(individuo: List[float]) -> List[float]:
    """
    Operador de Reparación y Factibilidad Estricta:
    1. Para las 11 variables continuas, garantiza min_i <= a_i < b_i < c_i <= max_i.
    2. Para los 26 pesos de reglas, garantiza acotamiento en [0.0, 1.0].
    """
    # 1. Reparar genes de Funciones de Pertenencia (0..32)
    for i, var in enumerate(VARIABLES_FISICOQUIMICAS):
        offset = i * 3
        min_val, max_val = RANGOS_VARIABLES_DEFAULT.get(var, (0.0, 100.0))
        rango_total = max_val - min_val
        eps = max(1e-4, 1e-3 * rango_total)

        puntos = sorted([float(individuo[offset]), float(individuo[offset + 1]), float(individuo[offset + 2])])
        a, b, c = puntos[0], puntos[1], puntos[2]

        a = float(np.clip(a, min_val, max_val - 2 * eps))
        b = float(np.clip(b, a + eps, max_val - eps))
        c = float(np.clip(c, b + eps, max_val))

        individuo[offset] = a
        individuo[offset + 1] = b
        individuo[offset + 2] = c

    # 2. Reparar genes de Pesos de Reglas (33..58)
    for j in range(LONGITUD_CROMOSOMA_MFS, len(individuo)):
        individuo[j] = float(np.clip(individuo[j], 0.0, 1.0))

    return individuo
