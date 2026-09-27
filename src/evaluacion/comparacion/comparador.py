"""
Submódulo de Comparación Progresiva de las 3 Fases
Proyecto: Clasificación de Calidad de Vinos (PRISM, Apriori, Lógica Difusa y Algoritmo Genético)
"""

from typing import Any, Dict, List
import numpy as np
import pandas as pd

from src.evaluacion.metricas.calculo import (
    ORDEN_CLASES,
    calcular_metricas_clasificacion,
)
from src.fuzzy.fuzzy import (
    ConfiguracionMFs,
    SistemaDifusoMamdani,
    UMBRAL_ACTIVACION_MINIMA,
)
from src.reglas.regla_unificada import ReglaUnificada
from src.reglas.clasificador_base.clasificador import (
    CLASE_POR_DEFECTO,
    ClasificadorReglasDiscretas,
)

UMBRAL_PESO_REGLA_ACTIVA: float = 0.05


def evaluar_fase_1_discreto(
    reglas: List[ReglaUnificada],
    particiones: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """
    Evalúa la Fase 1 (Modelo Baseline Discreto - PRISM + Apriori) sobre Train y Test.
    """
    clasificador = ClasificadorReglasDiscretas(reglas, clase_por_defecto=CLASE_POR_DEFECTO)
    resultados: Dict[str, Dict[str, Any]] = {}

    for split, x_key, y_key in [("train", "X_train_disc", "y_train"), ("test", "X_test_disc", "y_test")]:
        x_df = particiones[x_key]
        y_real = particiones[y_key]

        y_pred, reglas_activadas = clasificador.predecir(x_df)
        cubiertas = sum(1 for reg_list in reglas_activadas if len(reg_list) > 0)
        activas_count = sum(1 for r in reglas if r.activa)

        metricas = calcular_metricas_clasificacion(
            y_true=y_real,
            y_pred=y_pred,
            muestras_cubiertas=cubiertas,
            total_muestras=len(y_real),
            reglas_activas=activas_count,
            total_reglas=len(reglas),
        )
        resultados[split] = metricas

    return resultados


def evaluar_fase_2_pre_ga(
    config_mfs: ConfiguracionMFs,
    reglas: List[ReglaUnificada],
    particiones: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """
    Evalúa la Fase 2 (Sistema Difuso Mamdani Inicial - Pre-AG) sobre Train y Test.
    """
    sistema_mamdani = SistemaDifusoMamdani(
        config_mfs=config_mfs,
        reglas=reglas,
        umbral_peso_activa=0.0,
    )
    resultados: Dict[str, Dict[str, Any]] = {}

    for split, x_key, y_key in [("train", "X_train_num", "y_train"), ("test", "X_test_num", "y_test")]:
        x_df = particiones[x_key]
        y_real = particiones[y_key]

        y_pred, _, matriz_act = sistema_mamdani.predecir_vectorizado(x_df)
        cubiertas = int(np.sum(np.any(matriz_act >= UMBRAL_ACTIVACION_MINIMA, axis=1)))
        activas_count = sum(1 for r in reglas if r.activa and r.peso > 0.0)

        metricas = calcular_metricas_clasificacion(
            y_true=y_real,
            y_pred=y_pred,
            muestras_cubiertas=cubiertas,
            total_muestras=len(y_real),
            reglas_activas=activas_count,
            total_reglas=len(reglas),
        )
        resultados[split] = metricas

    return resultados


def evaluar_fase_3_post_ga(
    config_mfs: ConfiguracionMFs,
    reglas: List[ReglaUnificada],
    particiones: Dict[str, Any],
    umbral_peso_activa: float = UMBRAL_PESO_REGLA_ACTIVA,
) -> Dict[str, Dict[str, Any]]:
    """
    Evalúa la Fase 3 (Sistema Difuso Mamdani Optimizado - Post-AG) sobre Train y Test.
    """
    sistema_mamdani = SistemaDifusoMamdani(
        config_mfs=config_mfs,
        reglas=reglas,
        umbral_peso_activa=umbral_peso_activa,
    )
    resultados: Dict[str, Dict[str, Any]] = {}

    for split, x_key, y_key in [("train", "X_train_num", "y_train"), ("test", "X_test_num", "y_test")]:
        x_df = particiones[x_key]
        y_real = particiones[y_key]

        y_pred, _, matriz_act = sistema_mamdani.predecir_vectorizado(x_df)
        cubiertas = int(np.sum(np.any(matriz_act >= UMBRAL_ACTIVACION_MINIMA, axis=1)))
        activas_count = sum(1 for r in reglas if r.activa and r.peso >= umbral_peso_activa)

        metricas = calcular_metricas_clasificacion(
            y_true=y_real,
            y_pred=y_pred,
            muestras_cubiertas=cubiertas,
            total_muestras=len(y_real),
            reglas_activas=activas_count,
            total_reglas=len(reglas),
        )
        resultados[split] = metricas

    return resultados


def construir_dataframe_comparativo(
    metricas_fase1: Dict[str, Dict[str, Any]],
    metricas_fase2: Dict[str, Dict[str, Any]],
    metricas_fase3: Dict[str, Dict[str, Any]],
) -> pd.DataFrame:
    """Construye un DataFrame consolidado con las métricas de las 3 fases en Train y Test."""
    filas = []

    fases_data = [
        ("Fase 1 (Baseline - Reglas Discretas)", metricas_fase1),
        ("Fase 2 (Pre-GA - Difuso Inicial)", metricas_fase2),
        ("Fase 3 (Post-GA - Difuso Optimizado)", metricas_fase3),
    ]

    for nombre_fase, res_fase in fases_data:
        for split in ["train", "test"]:
            met = res_fase[split]
            filas.append(
                {
                    "Fase": nombre_fase,
                    "Conjunto": split.upper(),
                    "Exactitud (%)": round(met["exactitud"] * 100.0, 2),
                    "Precision Macro": round(met["precision_macro"], 4),
                    "Recall Macro": round(met["recall_macro"], 4),
                    "F1-Macro": round(met["f1_macro"], 4),
                    "F1-Weighted": round(met["f1_weighted"], 4),
                    "Cobertura (%)": round(met["cobertura_pct"], 2),
                    "Reglas Activas": f"{met['reglas_activas']} / {met['total_reglas']}",
                }
            )

    return pd.DataFrame(filas)
