"""
Submódulo de Cálculo de Métricas de Rendimiento
Proyecto: Clasificación de Calidad de Vinos (PRISM, Apriori, Lógica Difusa y Algoritmo Genético)
"""

from typing import Any, Dict, List, Union
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

ORDEN_CLASES: List[str] = ["Alta", "Baja", "Media"]


def calcular_metricas_clasificacion(
    y_true: Union[pd.Series, np.ndarray, List[str]],
    y_pred: Union[pd.Series, np.ndarray, List[str]],
    muestras_cubiertas: int,
    total_muestras: int,
    reglas_activas: int,
    total_reglas: int,
    orden_clases: List[str] = ORDEN_CLASES,
) -> Dict[str, Any]:
    """
    Calcula de forma unificada el conjunto completo de métricas de desempeño:
    Accuracy, Precision (Macro/Weighted), Recall (Macro/Weighted), F1 (Macro/Weighted),
    Cobertura, Reglas Activas, Matriz de Confusión y Reporte por clase.
    """
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)

    exactitud = float(accuracy_score(y_t, y_p))
    prec_macro = float(precision_score(y_t, y_p, average="macro", zero_division=0))
    prec_weighted = float(precision_score(y_t, y_p, average="weighted", zero_division=0))
    rec_macro = float(recall_score(y_t, y_p, average="macro", zero_division=0))
    rec_weighted = float(recall_score(y_t, y_p, average="weighted", zero_division=0))
    f1_macro = float(f1_score(y_t, y_p, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_t, y_p, average="weighted", zero_division=0))

    cobertura_pct = float((muestras_cubiertas / total_muestras) * 100.0) if total_muestras > 0 else 0.0

    matriz = confusion_matrix(y_t, y_p, labels=orden_clases)
    reporte_dict = classification_report(
        y_t,
        y_p,
        labels=orden_clases,
        target_names=orden_clases,
        output_dict=True,
        zero_division=0,
    )

    return {
        "exactitud": exactitud,
        "precision_macro": prec_macro,
        "precision_weighted": prec_weighted,
        "recall_macro": rec_macro,
        "recall_weighted": rec_weighted,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "cobertura_pct": cobertura_pct,
        "muestras_cubiertas": int(muestras_cubiertas),
        "total_muestras": int(total_muestras),
        "reglas_activas": int(reglas_activas),
        "total_reglas": int(total_reglas),
        "matriz_confusion": matriz.tolist(),
        "reporte_por_clase": reporte_dict,
        "predicciones": y_p.tolist(),
    }
