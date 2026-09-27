"""
Submódulo de Operadores de Mutación del Algoritmo Genético
Proyecto: Clasificación de Calidad de Vinos (PRISM, Apriori, Lógica Difusa y Algoritmo Genético)
"""

import random
from typing import List, Tuple

from src.datos.preparacion.discretizacion import VARIABLES_FISICOQUIMICAS
from src.fuzzy.pertenencia.funciones import RANGOS_VARIABLES_DEFAULT
from src.genetico.cromosoma.estructura import LONGITUD_CROMOSOMA_MFS
from src.genetico.cromosoma.reparacion import reparar_individuo


def mutar_individuo_mixto(
    individuo: List[float],
    indpb_mf: float = 0.20,
    sigma_mf_relativo: float = 0.08,
    indpb_regla: float = 0.20,
) -> Tuple[List[float]]:
    """
    Operador de Mutación Mixta Acotada:
    - MFs: Perturbación gaussiana proporcional al rango de cada variable.
    - Reglas: Mutación gaussiana de pesos con posibilidad de desactivación/reactivación.
    """
    # 1. Mutar genes de MFs
    for i, var in enumerate(VARIABLES_FISICOQUIMICAS):
        offset = i * 3
        min_val, max_val = RANGOS_VARIABLES_DEFAULT.get(var, (0.0, 100.0))
        rango = max_val - min_val
        sigma = sigma_mf_relativo * rango

        for k in range(3):
            idx = offset + k
            if random.random() < indpb_mf:
                individuo[idx] += random.gauss(0.0, sigma)

    # 2. Mutar genes de Reglas
    for j in range(LONGITUD_CROMOSOMA_MFS, len(individuo)):
        if random.random() < indpb_regla:
            # 20% de probabilidad de apagar/encender la regla completamente
            if random.random() < 0.20:
                individuo[j] = 0.0 if individuo[j] > 0.5 else 1.0
            else:
                individuo[j] += random.gauss(0.0, 0.15)

    reparar_individuo(individuo)
    return (individuo,)
