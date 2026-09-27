"""
Submódulo de Minería de Itemsets Frecuentes — Algoritmo Apriori
Proyecto: Clasificación de Calidad de Vinos (PRISM, Apriori, Lógica Difusa y Algoritmo Genético)
"""

from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple
import pandas as pd


def construir_transacciones(
    dataframe_x_disc: pd.DataFrame,
    serie_y_cat: pd.Series,
    nombre_objetivo: str = "calidad_categoria",
) -> List[Set[str]]:
    """
    Transforma DataFrames tabulares discretizados en una lista transaccional.
    Cada fila se convierte en un conjunto de items ('variable=valor') más el target.
    """
    transacciones: List[Set[str]] = []
    columnas = list(dataframe_x_disc.columns)

    for idx, fila in dataframe_x_disc.iterrows():
        items_fila = {f"{col}={fila[col]}" for col in columnas}
        etiqueta_calidad = str(serie_y_cat.loc[idx])
        items_fila.add(f"{nombre_objetivo}={etiqueta_calidad}")
        transacciones.append(items_fila)

    return transacciones


def generar_candidatos_c1(
    transacciones: List[Set[str]],
    min_soporte: float,
) -> Tuple[List[Tuple[str, ...]], Dict[Tuple[str, ...], float]]:
    """
    Calcula los itemsets frecuentes de tamaño 1 (L1) y sus soportes.
    """
    conteos: Dict[str, int] = defaultdict(int)
    total = len(transacciones)

    for transaccion in transacciones:
        for item in transaccion:
            conteos[item] += 1

    l1: List[Tuple[str, ...]] = []
    soporte_l1: Dict[Tuple[str, ...], float] = {}

    for item, conteo in conteos.items():
        soporte = conteo / total
        if soporte >= min_soporte:
            itemset = (item,)
            l1.append(itemset)
            soporte_l1[itemset] = soporte

    l1.sort()
    return l1, soporte_l1


def generar_candidatos_ck(
    itemsets_previos: List[Tuple[str, ...]],
    k: int,
) -> List[Tuple[str, ...]]:
    """
    Genera candidatos C_k a partir de L_{k-1} usando unión por prefijo y poda de subconjuntos.
    """
    candidatos: List[Tuple[str, ...]] = []
    num_itemsets = len(itemsets_previos)
    conjunto_previos: Set[Tuple[str, ...]] = set(itemsets_previos)

    for i in range(num_itemsets):
        for j in range(i + 1, num_itemsets):
            itemset_1 = itemsets_previos[i]
            itemset_2 = itemsets_previos[j]

            # Unión por prefijo
            if itemset_1[:k - 2] == itemset_2[:k - 2]:
                if itemset_1[k - 2] < itemset_2[k - 2]:
                    candidato = itemset_1 + (itemset_2[k - 2],)

                    # Poda (Prune): Verificar que todos los subconjuntos de tamaño k-1 estén en L_{k-1}
                    subconjuntos_validos = True
                    for m in range(k):
                        subconjunto = candidato[:m] + candidato[m + 1:]
                        if subconjunto not in conjunto_previos:
                            subconjuntos_validos = False
                            break

                    if subconjuntos_validos:
                        candidatos.append(candidato)
            else:
                break

    return candidatos


def contar_soporte_candidatos(
    candidatos: List[Tuple[str, ...]],
    transacciones: List[Set[str]],
    min_soporte: float,
) -> Dict[Tuple[str, ...], float]:
    """
    Calcula el soporte de cada candidato escaneando las transacciones.
    """
    if not candidatos:
        return {}

    conteos: Dict[Tuple[str, ...], int] = defaultdict(int)
    total = len(transacciones)

    candidatos_set = [(cand, set(cand)) for cand in candidatos]

    for transaccion in transacciones:
        for cand_tuple, cand_set in candidatos_set:
            if cand_set.issubset(transaccion):
                conteos[cand_tuple] += 1

    soporte_ck: Dict[Tuple[str, ...], float] = {}
    for cand, conteo in conteos.items():
        soporte = conteo / total
        if soporte >= min_soporte:
            soporte_ck[cand] = soporte

    return soporte_ck


def encontrar_itemsets_frecuentes(
    transacciones: List[Set[str]],
    min_soporte: float,
    max_longitud: int,
) -> Dict[Tuple[str, ...], float]:
    """
    Ejecuta el ciclo principal de Apriori para encontrar todos los itemsets frecuentes.
    """
    todos_los_frecuentes: Dict[Tuple[str, ...], float] = {}

    l1, soporte_l1 = generar_candidatos_c1(transacciones, min_soporte)
    todos_los_frecuentes.update(soporte_l1)
    actual_l = l1

    for k in range(2, max_longitud + 1):
        if not actual_l:
            break

        candidatos_ck = generar_candidatos_ck(actual_l, k)
        if not candidatos_ck:
            break

        soporte_lk = contar_soporte_candidatos(candidatos_ck, transacciones, min_soporte)
        if not soporte_lk:
            break

        actual_l = sorted(soporte_lk.keys())
        todos_los_frecuentes.update(soporte_lk)

    return todos_los_frecuentes
