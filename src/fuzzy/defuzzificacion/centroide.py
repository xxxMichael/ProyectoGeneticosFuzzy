"""
Submódulo de Defuzzificación por Centroide
Proyecto: Clasificación de Calidad de Vinos (PRISM, Apriori, Lógica Difusa y Algoritmo Genético)
"""

from typing import Dict, List
import numpy as np

# Universo de discurso para la variable objetivo de salida (Calidad del vino)
MIN_UNIVERSO_CALIDAD: float = 0.0
MAX_UNIVERSO_CALIDAD: float = 10.0
PASO_UNIVERSO_CALIDAD: float = 0.1
UNIVERSO_CALIDAD: np.ndarray = np.arange(
    MIN_UNIVERSO_CALIDAD, MAX_UNIVERSO_CALIDAD + PASO_UNIVERSO_CALIDAD, PASO_UNIVERSO_CALIDAD
)

# Centros y funciones de pertenencia de los conjuntos de salida (Calidad)
PARAMS_SALIDA_CALIDAD: Dict[str, List[float]] = {
    "Baja": [0.0, 0.0, 4.0, 5.5],        # Trapezoidal hombro izquierdo
    "Media": [4.5, 6.0, 7.2],            # Triangular simétrica
    "Alta": [6.5, 7.8, 10.0, 10.0],      # Trapezoidal hombro derecho
}

# Umbrales lingüísticos
UMBRAL_CORTE_BAJA_MEDIA: float = 5.30    # Centroide < 5.30 -> 'Baja'
UMBRAL_CORTE_MEDIA_ALTA: float = 6.60    # Centroide >= 6.60 -> 'Alta', resto -> 'Media'

CENTROIDE_NEUTRO_FALLBACK: float = 5.80  # Centroide neutro en caso de disparo nulo
CLASE_NEUTRA_FALLBACK: str = "Media"     # Clase asignada por defecto ante fallo de activación


def defuzzificar_centroide(
    universo: np.ndarray,
    mu_agregado: np.ndarray,
    centroide_fallback: float = CENTROIDE_NEUTRO_FALLBACK,
) -> float:
    """
    Calcula el centroide cuantitativo (Centro de Gravedad) de la curva difusa agregada.
    """
    area_total = float(np.sum(mu_agregado))
    if area_total <= 1e-6:
        return centroide_fallback

    momento = float(np.sum(mu_agregado * universo))
    try:
        centroide = momento / area_total
        if np.isnan(centroide) or np.isinf(centroide):
            return centroide_fallback
        return float(centroide)
    except ZeroDivisionError:
        return centroide_fallback


def clasificar_centroide(
    valor_centroide: float,
    umbral_baja_media: float = UMBRAL_CORTE_BAJA_MEDIA,
    umbral_media_alta: float = UMBRAL_CORTE_MEDIA_ALTA,
) -> str:
    """
    Aplica los umbrales cuantitativos sobre el centroide defuzzificado:
    - Centroide < 5.30 -> 'Baja'
    - Centroide >= 6.60 -> 'Alta'
    - Resto -> 'Media'
    """
    if valor_centroide < umbral_baja_media:
        return "Baja"
    if valor_centroide >= umbral_media_alta:
        return "Alta"
    return "Media"
