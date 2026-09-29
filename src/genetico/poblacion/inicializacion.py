"""
Submódulo de Inicialización de Población del Algoritmo Genético
Proyecto: Clasificación de Calidad de Vinos (PRISM, Apriori, Lógica Difusa y Algoritmo Genético)
"""

import random
from typing import List

from src.datos.preparacion.discretizacion import VARIABLES_FISICOQUIMICAS
from src.fuzzy.pertenencia.funciones import (
    RANGOS_VARIABLES_DEFAULT,
    ConfiguracionMFs,
)
from src.genetico.cromosoma.estructura import (
    LONGITUD_CROMOSOMA_MFS,
    LONGITUD_CROMOSOMA_REGLAS,
)
from src.genetico.cromosoma.reparacion import reparar_individuo


def crear_individuo_aleatorio(num_reglas: int = LONGITUD_CROMOSOMA_REGLAS) -> List[float]:
    """
    Genera un individuo 100% aleatorio dentro de los límites físicos del dataset:
    - 33 genes de MFs: 3 puntos continuos aleatorios en [min_val, max_val] para cada una de las 11 variables.
    - 26 genes de reglas: pesos continuos aleatorios en [0.0, 1.0].
    - Se aplica reparación estricta para garantizar que min_i <= a_i < b_i < c_i <= max_i.
    """
    cromosoma_mfs: List[float] = []
    for var in VARIABLES_FISICOQUIMICAS:
        min_val, max_val = RANGOS_VARIABLES_DEFAULT.get(var, (0.0, 100.0))
        puntos = [random.uniform(min_val, max_val) for _ in range(3)]
        cromosoma_mfs.extend(puntos)

    cromosoma_reglas = [random.uniform(0.0, 1.0) for _ in range(num_reglas)]
    individuo = cromosoma_mfs + cromosoma_reglas
    return reparar_individuo(individuo)


def crear_individuo_semilla(
    config_mfs_inicial: ConfiguracionMFs,
    num_reglas: int = LONGITUD_CROMOSOMA_REGLAS,
) -> List[float]:
    """
    Construye el cromosoma semilla canónico:
    - Parte 1 (33 genes): Puntos de corte empíricos de las 11 variables (a, b, c).
    - Parte 2 (26 genes): Pesos iniciales de las reglas fijados en 1.0 (todas activas).
    """
    cromosoma_mfs = config_mfs_inicial.a_cromosoma()
    cromosoma_reglas = [1.0] * num_reglas
    return cromosoma_mfs + cromosoma_reglas


def crear_individuo_perturbado(
    individuo_semilla: List[float],
    escala_ruido_mfs: float = 0.08,
) -> List[float]:
    """
    Genera un individuo exploratorio aplicando perturbaciones aleatorias uniformes sobre el individuo semilla.
    """
    nuevo_ind = list(individuo_semilla)

    for i, var in enumerate(VARIABLES_FISICOQUIMICAS):
        offset = i * 3
        min_val, max_val = RANGOS_VARIABLES_DEFAULT.get(var, (0.0, 100.0))
        rango = max_val - min_val
        delta = escala_ruido_mfs * rango

        for k in range(3):
            nuevo_ind[offset + k] += random.uniform(-delta, delta)

    for j in range(LONGITUD_CROMOSOMA_MFS, len(nuevo_ind)):
        nuevo_ind[j] = random.uniform(0.2, 1.0)

    return reparar_individuo(nuevo_ind)
