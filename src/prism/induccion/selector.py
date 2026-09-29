"""
Submódulo de Inducción y Selección Voraz de Términos para PRISM
Proyecto: Clasificación de Calidad de Vinos (PRISM, Apriori, Lógica Difusa y Algoritmo Genético)
"""

import random
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


def buscar_mejor_termino_voraz(
    conjunto_s: pd.DataFrame,
    variables_disponibles: List[str],
    clase_actual: str,
    columna_target: str = "__target__",
    semilla_aleatoria: Optional[int] = 42,
) -> Tuple[Optional[Tuple[str, str]], float, int, int]:
    """
    Busca de manera voraz el término (variable = valor) con mayor poder discriminante
    sobre el subconjunto S siguiendo la cascada jerárquica de desempate formal:

    Cascada de Desempate de PRISM:
    1. Confianza: Precisión local p(clase_actual | variable = valor).
    2. Cobertura: Total de instancias en S que satisfacen la condición.
    3. Soporte: Cantidad absoluta de instancias positivas de la clase objetivo.
    4. Sustentación (Lift): Relación de confianza frente a la probabilidad a priori de la clase.
    5. Aleatoriamente: Selección estocástica equiprobable para neutralizar sesgos posicionales.

    Args:
        conjunto_s (pd.DataFrame): Subconjunto activo de instancias.
        variables_disponibles (List[str]): Variables aún no incluidas en la regla actual.
        clase_actual (str): Clase objetivo que se está induciendo.
        columna_target (str): Nombre de la columna que contiene la etiqueta de clase.
        semilla_aleatoria (Optional[int]): Semilla para reproducibilidad del desempate aleatorio.

    Returns:
        Tuple[Optional[Tuple[str, str]], float, int, int]:
            - Par (variable, valor) seleccionado (o None si no hay candidatos positivos).
            - Mejor precisión (confianza) alcanzada.
            - Cantidad de positivos locales (soporte).
            - Total de instancias locales cubiertas (cobertura).
    """
    if len(conjunto_s) == 0:
        return None, -1.0, -1, -1

    total_s = len(conjunto_s)
    total_positivos_s = len(conjunto_s[conjunto_s[columna_target] == clase_actual])
    if total_positivos_s == 0:
        return None, -1.0, -1, -1

    prob_clase_global = total_positivos_s / total_s

    # 1. Evaluar todos los términos candidatos válidos
    candidatos: List[Dict[str, Any]] = []

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
            if positivos_locales == 0:
                continue

            confianza = positivos_locales / total_locales
            cobertura = total_locales
            soporte = positivos_locales
            lift = (confianza / prob_clase_global) if prob_clase_global > 0 else 0.0

            candidatos.append({
                "condicion": (variable, str(valor)),
                "confianza": float(confianza),
                "cobertura": int(cobertura),
                "soporte": int(soporte),
                "lift": float(lift),
            })

    if not candidatos:
        return None, -1.0, -1, -1

    # Nivel 1: Mayor Confianza
    max_confianza = max(c["confianza"] for c in candidatos)
    cand_nivel_1 = [c for c in candidatos if np.isclose(c["confianza"], max_confianza)]

    if len(cand_nivel_1) == 1:
        ganador = cand_nivel_1[0]
        return ganador["condicion"], ganador["confianza"], ganador["soporte"], ganador["cobertura"]

    # Nivel 2: Mayor Cobertura (Desempate 1)
    max_cobertura = max(c["cobertura"] for c in cand_nivel_1)
    cand_nivel_2 = [c for c in cand_nivel_1 if c["cobertura"] == max_cobertura]

    if len(cand_nivel_2) == 1:
        ganador = cand_nivel_2[0]
        return ganador["condicion"], ganador["confianza"], ganador["soporte"], ganador["cobertura"]

    # Nivel 3: Mayor Soporte (Desempate 2)
    max_soporte = max(c["soporte"] for c in cand_nivel_2)
    cand_nivel_3 = [c for c in cand_nivel_2 if c["soporte"] == max_soporte]

    if len(cand_nivel_3) == 1:
        ganador = cand_nivel_3[0]
        return ganador["condicion"], ganador["confianza"], ganador["soporte"], ganador["cobertura"]

    # Nivel 4: Mayor Lift (Desempate 3)
    max_lift = max(c["lift"] for c in cand_nivel_3)
    cand_nivel_4 = [c for c in cand_nivel_3 if np.isclose(c["lift"], max_lift)]

    if len(cand_nivel_4) == 1:
        ganador = cand_nivel_4[0]
        return ganador["condicion"], ganador["confianza"], ganador["soporte"], ganador["cobertura"]

    # Nivel 5: Desempate Aleatorio (Rompe cualquier sesgo posicional de lectura)
    rng = random.Random(semilla_aleatoria) if semilla_aleatoria is not None else random
    ganador = rng.choice(cand_nivel_4)

    return ganador["condicion"], ganador["confianza"], ganador["soporte"], ganador["cobertura"]
