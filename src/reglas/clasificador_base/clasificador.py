"""
Submódulo del Clasificador de Reglas Discretas (Modelo Base — Baseline)
Proyecto: Clasificación de Calidad de Vinos (PRISM, Apriori, Lógica Difusa y Algoritmo Genético)
"""

from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from src.reglas.regla_unificada import ReglaUnificada

CLASE_POR_DEFECTO: str = "Media"


class ClasificadorReglasDiscretas:
    """
    Clasificador de inferencia discreta (Modelo Base - Baseline) que evalúa
    la base de reglas integradas mediante votación ponderada por confianza.
    """

    def __init__(
        self,
        reglas: List[ReglaUnificada],
        clase_por_defecto: str = CLASE_POR_DEFECTO,
    ) -> None:
        self.reglas = [r for r in reglas if r.activa]
        self.clase_por_defecto = clase_por_defecto

    def predecir_muestra(self, fila_discreta: Dict[str, Any]) -> Tuple[str, List[int]]:
        """
        Predice la calidad de un vino evaluando las reglas activadas.

        Returns:
            Tuple[str, List[int]]: (Clase predicha, IDs de reglas disparadas).
        """
        votos_por_clase: Dict[str, float] = {"Baja": 0.0, "Media": 0.0, "Alta": 0.0}
        reglas_disparadas: List[int] = []

        for regla in self.reglas:
            if regla.coincide_discreto(fila_discreta):
                reglas_disparadas.append(regla.id_regla)
                votos_por_clase[regla.consecuente] += regla.peso * regla.confianza

        if not reglas_disparadas or sum(votos_por_clase.values()) == 0.0:
            return self.clase_por_defecto, []

        clase_ganadora = max(votos_por_clase.items(), key=lambda x: x[1])[0]
        return clase_ganadora, reglas_disparadas

    def predecir(self, dataframe_disc: pd.DataFrame) -> Tuple[np.ndarray, List[List[int]]]:
        """Realiza predicciones sobre un conjunto completo de datos discretos."""
        predicciones = []
        todas_reglas_disparadas = []

        for _, fila in dataframe_disc.iterrows():
            dict_fila = fila.to_dict()
            pred, activadas = self.predecir_muestra(dict_fila)
            predicciones.append(pred)
            todas_reglas_disparadas.append(activadas)

        return np.array(predicciones), todas_reglas_disparadas

    def evaluar(
        self,
        dataframe_disc: pd.DataFrame,
        y_real: pd.Series,
    ) -> Dict[str, Any]:
        """Calcula métricas completas de clasificación para el conjunto dado."""
        y_pred, reglas_activadas = self.predecir(dataframe_disc)
        y_true = np.array(y_real)

        exactitud = float(np.mean(y_pred == y_true))
        f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        f1_ponderado = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

        muestras_cubiertas = sum(1 for reg in reglas_activadas if len(reg) > 0)
        cobertura = float(muestras_cubiertas / len(dataframe_disc))

        reporte = classification_report(
            y_true, y_pred, target_names=["Alta", "Baja", "Media"], output_dict=True, zero_division=0
        )
        matriz = confusion_matrix(y_true, y_pred, labels=["Alta", "Baja", "Media"])

        return {
            "exactitud": exactitud,
            "f1_macro": f1_macro,
            "f1_ponderado": f1_ponderado,
            "cobertura": cobertura,
            "matriz_confusion": matriz,
            "reporte_clasificacion": reporte,
            "predicciones": y_pred,
        }
