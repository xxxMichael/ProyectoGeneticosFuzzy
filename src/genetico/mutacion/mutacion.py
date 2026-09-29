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
    delta_mf_relativo: float = 0.10,
    indpb_regla: float = 0.20,
) -> Tuple[List[float]]:
    """
    Operador de Mutación Uniforme Estándar (Nativa de Algoritmos Genéticos):
    - MFs: Mutación uniforme acotada en [-delta, delta] proporcional al rango de la variable.
    - Reglas: Mutación uniforme de pesos [-0.15, 0.15] con probabilidad de encendido/apagado (bit-flip).
    """
    # 1. Mutar genes de MFs (Mutación Uniforme Clásica)
    for i, var in enumerate(VARIABLES_FISICOQUIMICAS):
        offset = i * 3
        min_val, max_val = RANGOS_VARIABLES_DEFAULT.get(var, (0.0, 100.0))
        rango = max_val - min_val
        delta = delta_mf_relativo * rango

        for k in range(3):
            idx = offset + k
            if random.random() < indpb_mf:
                individuo[idx] += random.uniform(-delta, delta)

    # 2. Mutar genes de Reglas (Mutación Uniforme / Flip)
    for j in range(LONGITUD_CROMOSOMA_MFS, len(individuo)):
        if random.random() < indpb_regla:
            # 20% de probabilidad de apagar/encender la regla completamente (Flip clásico)
            if random.random() < 0.20:
                individuo[j] = 0.0 if individuo[j] > 0.5 else 1.0
            else:
                individuo[j] += random.uniform(-0.15, 0.15)

    reparar_individuo(individuo)
    return (individuo,)
