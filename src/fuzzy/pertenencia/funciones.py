"""
Submódulo de Funciones de Pertenencia y Configuración Difusa
Proyecto: Clasificación de Calidad de Vinos (PRISM, Apriori, Lógica Difusa y Algoritmo Genético)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.datos.preparacion.discretizacion import VARIABLES_FISICOQUIMICAS

# Rango empírico mínimo y máximo de las 11 variables continuas del dataset (con margen de seguridad)
RANGOS_VARIABLES_DEFAULT: Dict[str, Tuple[float, float]] = {
    "fixed acidity": (4.0, 16.5),
    "volatile acidity": (0.10, 1.65),
    "citric acid": (0.0, 1.05),
    "residual sugar": (0.8, 16.0),
    "chlorides": (0.01, 0.65),
    "free sulfur dioxide": (1.0, 75.0),
    "total sulfur dioxide": (5.0, 295.0),
    "density": (0.989, 1.005),
    "pH": (2.70, 4.05),
    "sulphates": (0.30, 2.05),
    "alcohol": (8.0, 15.5),
}


def calcular_pertenencia_trapezoidal(
    x: Union[float, np.ndarray],
    a: float,
    b: float,
    c: float,
    d: float,
) -> Union[float, np.ndarray]:
    """
    Calcula el grado de pertenencia mu(x) para una función trapezoidal [a, b, c, d].
    Soporta escalares y arrays vectorizados de NumPy.
    """
    if isinstance(x, (int, float)):
        if x <= a:
            return 1.0 if a == b else 0.0
        if a < x < b:
            return float((x - a) / (b - a)) if b > a else 1.0
        if b <= x <= c:
            return 1.0
        if c < x < d:
            return float((d - x) / (d - c)) if d > c else 1.0
        return 1.0 if c == d and x >= d else 0.0

    mu = np.zeros_like(x, dtype=float)
    if a == b:
        mu[x <= a] = 1.0
    else:
        idx_subida = (x > a) & (x < b)
        mu[idx_subida] = (x[idx_subida] - a) / (b - a)

    idx_tope = (x >= b) & (x <= c)
    mu[idx_tope] = 1.0

    if c == d:
        mu[x >= d] = 1.0
    else:
        idx_bajada = (x > c) & (x < d)
        mu[idx_bajada] = (d - x[idx_bajada]) / (d - c)

    return np.clip(mu, 0.0, 1.0)


def calcular_pertenencia_triangular(
    x: Union[float, np.ndarray],
    a: float,
    b: float,
    c: float,
) -> Union[float, np.ndarray]:
    """
    Calcula el grado de pertenencia mu(x) para una función triangular [a, b, c].
    """
    if isinstance(x, (int, float)):
        if x <= a or x >= c:
            return 0.0
        if a < x <= b:
            return float((x - a) / (b - a)) if b > a else 1.0
        if b < x < c:
            return float((c - x) / (c - b)) if c > b else 1.0
        return 0.0

    mu = np.zeros_like(x, dtype=float)
    idx_subida = (x > a) & (x <= b)
    if b > a:
        mu[idx_subida] = (x[idx_subida] - a) / (b - a)
    else:
        mu[idx_subida] = 1.0

    idx_bajada = (x > b) & (x < c)
    if c > b:
        mu[idx_bajada] = (c - x[idx_bajada]) / (c - b)
    else:
        mu[idx_bajada] = 1.0

    return np.clip(mu, 0.0, 1.0)


@dataclass
class ConfiguracionMFs:
    """
    Representa la configuración de los parámetros (a, b, c) de las funciones de pertenencia
    híbridas para las 11 variables fisicoquímicas, garantizando a < b < c.
    """

    parametros: Dict[str, List[float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.parametros:
            self.parametros = self._inicializar_desde_estadisticas_default()

    @classmethod
    def desde_dataframe(cls, dataframe_x: pd.DataFrame) -> "ConfiguracionMFs":
        params: Dict[str, List[float]] = {}
        for col in dataframe_x.columns:
            valores = dataframe_x[col]
            q25 = float(valores.quantile(0.25))
            q50 = float(valores.median())
            q75 = float(valores.quantile(0.75))

            eps = 1e-4 * (float(valores.max()) - float(valores.min()) + 1.0)
            if q50 <= q25:
                q50 = q25 + eps
            if q75 <= q50:
                q75 = q50 + eps

            params[col] = [q25, q50, q75]

        return cls(parametros=params)

    @classmethod
    def desde_cromosoma(
        cls,
        cromosoma: List[float],
        variables: Optional[List[str]] = None,
    ) -> "ConfiguracionMFs":
        vars_list = variables if variables is not None else VARIABLES_FISICOQUIMICAS
        params: Dict[str, List[float]] = {}

        for i, var in enumerate(vars_list):
            offset = i * 3
            puntos = sorted(cromosoma[offset : offset + 3])
            a, b, c = puntos[0], puntos[1], puntos[2]
            if b <= a:
                b = a + 1e-4
            if c <= b:
                c = b + 1e-4
            params[var] = [a, b, c]

        return cls(parametros=params)

    def a_cromosoma(self, variables: Optional[List[str]] = None) -> List[float]:
        vars_list = variables if variables is not None else VARIABLES_FISICOQUIMICAS
        cromosoma: List[float] = []
        for var in vars_list:
            a, b, c = self.parametros.get(var, [0.0, 1.0, 2.0])
            cromosoma.extend([a, b, c])
        return cromosoma

    def copiar(self) -> "ConfiguracionMFs":
        return ConfiguracionMFs(parametros={k: list(v) for k, v in self.parametros.items()})

    def _inicializar_desde_estadisticas_default(self) -> Dict[str, List[float]]:
        return {
            "fixed acidity": [7.10, 7.90, 9.20],
            "volatile acidity": [0.39, 0.52, 0.64],
            "citric acid": [0.09, 0.26, 0.42],
            "residual sugar": [1.90, 2.20, 2.60],
            "chlorides": [0.07, 0.08, 0.09],
            "free sulfur dioxide": [7.0, 14.0, 21.0],
            "total sulfur dioxide": [22.0, 38.0, 62.0],
            "density": [0.9956, 0.9968, 0.9978],
            "pH": [3.21, 3.31, 3.40],
            "sulphates": [0.55, 0.62, 0.73],
            "alcohol": [9.50, 10.20, 11.10],
        }

    def calcular_pertenencias(
        self,
        variable: str,
        valor: float,
    ) -> Dict[str, float]:
        min_lim, max_lim = RANGOS_VARIABLES_DEFAULT.get(variable, (0.0, 100.0))
        a, b, c = self.parametros[variable]

        mu_bajo = calcular_pertenencia_trapezoidal(valor, min_lim - 1.0, min_lim - 1.0, a, b)
        mu_medio = calcular_pertenencia_triangular(valor, a, b, c)
        mu_alto = calcular_pertenencia_trapezoidal(valor, b, c, max_lim + 1.0, max_lim + 1.0)

        return {
            "Bajo": float(np.clip(mu_bajo, 0.0, 1.0)),
            "Medio": float(np.clip(mu_medio, 0.0, 1.0)),
            "Alto": float(np.clip(mu_alto, 0.0, 1.0)),
        }


def validar_cobertura_difusa(
    dataframe_x: pd.DataFrame,
    config_mfs: ConfiguracionMFs,
) -> Tuple[bool, Dict[str, Any]]:
    inconsistencias_orden: List[str] = []
    muestras_con_pertenencia_nula: int = 0
    detalles_por_variable: Dict[str, Dict[str, float]] = {}

    for var, (a, b, c) in config_mfs.parametros.items():
        if not (a < b < c):
            inconsistencias_orden.append(f"{var}: a={a}, b={b}, c={c} no cumple a < b < c")

    for col in dataframe_x.columns:
        if col not in config_mfs.parametros:
            continue

        suma_minima = 1.0
        suma_maxima = 0.0
        nulos_var = 0

        for val in dataframe_x[col]:
            perts = config_mfs.calcular_pertenencias(col, float(val))
            suma = sum(perts.values())
            if suma < suma_minima:
                suma_minima = suma
            if suma > suma_maxima:
                suma_maxima = suma
            if max(perts.values()) == 0.0:
                nulos_var += 1

        detalles_por_variable[col] = {
            "suma_minima_pertenencia": suma_minima,
            "suma_maxima_pertenencia": suma_maxima,
            "muestras_con_cero": nulos_var,
        }
        if nulos_var > 0:
            muestras_con_pertenencia_nula += nulos_var

    es_valido = (len(inconsistencias_orden) == 0) and (muestras_con_pertenencia_nula == 0)

    resultado = {
        "es_valido": es_valido,
        "inconsistencias_orden": inconsistencias_orden,
        "muestras_con_pertenencia_nula": muestras_con_pertenencia_nula,
        "detalles_por_variable": detalles_por_variable,
    }
    return es_valido, resultado
