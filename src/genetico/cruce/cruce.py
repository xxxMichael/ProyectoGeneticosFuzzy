"""
Submódulo de Operadores de Cruce del Algoritmo Genético
Proyecto: Clasificación de Calidad de Vinos (PRISM, Apriori, Lógica Difusa y Algoritmo Genético)
"""

import random
from typing import List, Tuple

from src.datos.preparacion.discretizacion import VARIABLES_FISICOQUIMICAS
from src.genetico.cromosoma.estructura import LONGITUD_CROMOSOMA_MFS
from src.genetico.cromosoma.reparacion import reparar_individuo


def cruzar_individuos_mixtos(
    ind1: List[float],
    ind2: List[float],
    eta_sbx: float = 15.0,
    indpb_uniform: float = 0.5,
) -> Tuple[List[float], List[float]]:
    """
    Operador de Cruce Híbrido Mixto:
    - Parte 1 (Genes reales de MFs): Cruce SBX (Simulated Binary Crossover).
    - Parte 2 (Pesos de reglas): Cruce Uniforme.
    """
    hijo1 = list(ind1)
    hijo2 = list(ind2)

    # 1. Cruce SBX por variable para los genes de MFs
    for i in range(len(VARIABLES_FISICOQUIMICAS)):
        offset = i * 3
        if random.random() < 0.7:
            for k in range(3):
                idx = offset + k
                p1, p2 = hijo1[idx], hijo2[idx]
                if abs(p1 - p2) > 1e-6:
                    u = random.random()
                    if u <= 0.5:
                        beta = (2.0 * u) ** (1.0 / (eta_sbx + 1.0))
                    else:
                        beta = (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta_sbx + 1.0))
                    hijo1[idx] = 0.5 * ((1.0 + beta) * p1 + (1.0 - beta) * p2)
                    hijo2[idx] = 0.5 * ((1.0 - beta) * p1 + (1.0 + beta) * p2)

    # 2. Cruce Uniforme para los pesos de las 26 reglas
    for j in range(LONGITUD_CROMOSOMA_MFS, len(hijo1)):
        if random.random() < indpb_uniform:
            hijo1[j], hijo2[j] = hijo2[j], hijo1[j]

    reparar_individuo(hijo1)
    reparar_individuo(hijo2)

    for idx in range(len(hijo1)):
        ind1[idx] = hijo1[idx]
        ind2[idx] = hijo2[idx]

    return ind1, ind2
