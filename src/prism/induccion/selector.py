"""
Submódulo de Inducción y Selección Voraz de Términos para PRISM
Proyecto: Clasificación de Calidad de Vinos (PRISM, Apriori, Lógica Difusa y Algoritmo Genético)
"""

from typing import List, Optional, Tuple
import numpy as np
import pandas as pd


def buscar_mejor_termino_voraz(
    conjunto_s: pd.DataFrame,
    variables_disponibles: List[str],
    clase_actual: str,
    columna_target: str = "__target__",
) -> Tuple[Optional[Tuple[str, str]], float, int, int]:
    """
    Busca de manera voraz el término (variable = valor) con mayor precisión local
    p(clase_actual | variable = valor) sobre el subconjunto S.

    Criterio de desempate de PRISM:
    1. Mayor precisión local.
    2. En empate, mayor soporte positivo (número de instancias de la clase).
    3. En empate, mayor cobertura total.

    Args:
        conjunto_s (pd.DataFrame): Subconjunto activo de instancias.
        variables_disponibles (List[str]): Variables aún no incluidas en la regla actual.
        clase_actual (str): Clase objetivo que se está induciendo.
        columna_target (str): Nombre de la columna que contiene la etiqueta de clase.

    Returns:
        Tuple[Optional[Tuple[str, str]], float, int, int]:
            - Par (variable, valor) seleccionado (o None si no hay candidatos positivos).
            - Mejor precisión alcanzada.
            - Cantidad de positivos locales.
            - Total de instancias locales cubiertas.
    """
    mejor_condicion: Optional[Tuple[str, str]] = None
    mejor_precision: float = -1.0
    mejor_soporte_positivo: int = -1
    mejor_cobertura_local: int = -1

    for variable in variables_disponibles:
        valores_presentes = conjunto_s[variable].unique()
        for valor in valores_presentes:
            subconjunto_condicion = conjunto_s[conjunto_s[variable] == valor]
            total_locales = len(subconjunto_condicion)
            if total_locales == 0:
                continue

            positivos_locales = len(
                subconjunto_condicion[subconjunto_condicion[columna_target] == clase_actual]
            )
            precision_condicion = positivos_locales / total_locales

            es_optima = False
            if precision_condicion > mejor_precision:
                es_optima = True
            elif np.isclose(precision_condicion, mejor_precision):
                if positivos_locales > mejor_soporte_positivo:
                    es_optima = True
                elif (
                    positivos_locales == mejor_soporte_positivo
                    and total_locales > mejor_cobertura_local
                ):
                    es_optima = True

            if es_optima:
                mejor_precision = precision_condicion
                mejor_soporte_positivo = positivos_locales
                mejor_cobertura_local = total_locales
                mejor_condicion = (variable, str(valor))

    return mejor_condicion, mejor_precision, mejor_soporte_positivo, mejor_cobertura_local
