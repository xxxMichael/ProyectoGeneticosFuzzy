"""
Submódulo del Motor de Inferencia Mamdani
Proyecto: Clasificación de Calidad de Vinos (PRISM, Apriori, Lógica Difusa y Algoritmo Genético)
"""

from typing import Dict, List, Optional, Tuple
import numpy as np

from src.reglas.regla_unificada import ReglaUnificada

UMBRAL_ACTIVACION_MINIMA: float = 1e-4


def evaluar_activaciones_reglas(
    pertenencias_muestra: Dict[str, Dict[str, float]],
    reglas: List[ReglaUnificada],
    umbral_minimo: float = UMBRAL_ACTIVACION_MINIMA,
) -> Dict[int, float]:
    """
    Calcula el grado de activación alpha_k = w_k * min(mu_{var, val}) para cada regla.
    """
    activaciones: Dict[int, float] = {}

    for regla in reglas:
        if not regla.activa or regla.peso <= 0.0:
            continue

        grados_condiciones = []
        for var, etiqueta in regla.antecedentes.items():
            if var in pertenencias_muestra and etiqueta in pertenencias_muestra[var]:
                grados_condiciones.append(pertenencias_muestra[var][etiqueta])
            else:
                grados_condiciones.append(0.0)

        if grados_condiciones:
            alpha_k = float(regla.peso * min(grados_condiciones))
        else:
            alpha_k = 0.0

        if alpha_k >= umbral_minimo:
            activaciones[regla.id_regla] = alpha_k

    return activaciones


def agregar_salida_difusa(
    activaciones_reglas: Dict[int, float],
    reglas: List[ReglaUnificada],
    mf_salida_baja: np.ndarray,
    mf_salida_media: np.ndarray,
    mf_salida_alta: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Agrega las salidas difusas de las reglas activadas por clase (Mamdani MAX) y
    recorta los conjuntos de salida (Mamdani MIN).
    """
    mapa_reglas = {r.id_regla: r for r in reglas}
    alphas_por_clase: Dict[str, float] = {"Baja": 0.0, "Media": 0.0, "Alta": 0.0}

    for id_r, alpha_k in activaciones_reglas.items():
        if id_r in mapa_reglas:
            clase = mapa_reglas[id_r].consecuente
            if clase in alphas_por_clase:
                alphas_por_clase[clase] = max(alphas_por_clase[clase], alpha_k)

    corte_baja = np.minimum(alphas_por_clase["Baja"], mf_salida_baja)
    corte_media = np.minimum(alphas_por_clase["Media"], mf_salida_media)
    corte_alta = np.minimum(alphas_por_clase["Alta"], mf_salida_alta)

    salida_agregada = np.maximum(corte_baja, np.maximum(corte_media, corte_alta))
    return salida_agregada, alphas_por_clase
