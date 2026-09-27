"""
Módulo de Selección, Integración y Evaluación de la Base de Reglas
Proyecto: Clasificación de Calidad de Vinos (PRISM, Apriori, Lógica Difusa y Algoritmo Genético)
"""

import os
import sys
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

from src.datos.carga.cargador import cargar_dataset
from src.datos.preparacion.discretizacion import preparar_conjuntos_entrenamiento_prueba
from src.reglas.regla_unificada import ReglaUnificada
from src.reglas.integracion.unificador import (
    MAX_REGLAS_APRIORI_POR_CLASE,
    MIN_CONFIANZA_APRIORI_FILTRO,
    MIN_LIFT_APRIORI_FILTRO,
    ORDEN_CLASES,
    cargar_reglas_apriori_csv,
    cargar_reglas_prism_csv,
    guardar_base_integrada,
    integrar_base_reglas,
)
from src.reglas.clasificador_base.clasificador import (
    CLASE_POR_DEFECTO,
    ClasificadorReglasDiscretas,
)

# =============================================================================
# CONFIGURACIÓN Y RUTAS DE ARTEFACTOS
# =============================================================================
DIRECTORIO_BASE: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUTA_REGLAS_PRISM: str = os.path.join(DIRECTORIO_BASE, "results", "reglas_prism.csv")
RUTA_REGLAS_APRIORI: str = os.path.join(DIRECTORIO_BASE, "results", "reglas_apriori.csv")
RUTA_SALIDA_INTEGRADA_CSV: str = os.path.join(DIRECTORIO_BASE, "results", "reglas_base_integrada.csv")
RUTA_SALIDA_INTEGRADA_JSON: str = os.path.join(DIRECTORIO_BASE, "results", "reglas_base_integrada.json")
# =============================================================================


def ejecutar_integracion_y_evaluacion() -> Tuple[List[ReglaUnificada], Dict[str, Any], Dict[str, Any]]:
    """
    Ejecuta el pipeline completo de la Fase 7: carga de PRISM y Apriori,
    integración, exportación y evaluación del Modelo Base Discreto.
    """
    print("\n" + "=" * 80)
    print(" [FASE 7] SELECCION, INTEGRACION Y EVALUACION DE LA BASE DE REGLAS")
    print("=" * 80)

    reglas_prism = cargar_reglas_prism_csv(RUTA_REGLAS_PRISM)
    reglas_apriori = cargar_reglas_apriori_csv(RUTA_REGLAS_APRIORI)
    print(f" * Reglas cargadas de PRISM   : {len(reglas_prism)}")
    print(f" * Reglas cargadas de Apriori : {len(reglas_apriori)}")

    base_integrada = integrar_base_reglas(
        reglas_prism=reglas_prism,
        reglas_apriori=reglas_apriori,
        max_apriori_por_clase=MAX_REGLAS_APRIORI_POR_CLASE,
        min_confianza=MIN_CONFIANZA_APRIORI_FILTRO,
        min_lift=MIN_LIFT_APRIORI_FILTRO,
    )
    print(f" * Total Reglas Integradas    : {len(base_integrada)}")

    conteo_origen = pd.Series([r.origen for r in base_integrada]).value_counts().to_dict()
    conteo_clase = pd.Series([r.consecuente for r in base_integrada]).value_counts().to_dict()
    print(f"   - Por Algoritmo : {conteo_origen}")
    print(f"   - Por Calidad   : {conteo_clase}")

    guardar_base_integrada(
        base_integrada,
        ruta_csv=RUTA_SALIDA_INTEGRADA_CSV,
        ruta_json=RUTA_SALIDA_INTEGRADA_JSON,
    )
    print(f" [+] Base integrada guardada en: {RUTA_SALIDA_INTEGRADA_CSV}")

    df_vinos = cargar_dataset()
    datos_particionados = preparar_conjuntos_entrenamiento_prueba(df_vinos)

    X_train_disc = datos_particionados["X_train_disc"]
    X_test_disc = datos_particionados["X_test_disc"]
    y_train = datos_particionados["y_train"]
    y_test = datos_particionados["y_test"]

    clasificador_base = ClasificadorReglasDiscretas(base_integrada)
    metricas_train = clasificador_base.evaluar(X_train_disc, y_train)
    metricas_test = clasificador_base.evaluar(X_test_disc, y_test)

    print("\n" + "-" * 80)
    print(" [MODELO BASE DISCRETO] RENDIMIENTO DEL CLASIFICADOR INICIAL DE REGLAS:")
    print("-" * 80)
    print(f" * Exactitud en Train (Entrenamiento) : {metricas_train['exactitud'] * 100:.2f}%")
    print(f" * Exactitud en Test (Prueba)         : {metricas_test['exactitud'] * 100:.2f}%")
    print(f" * F1-Macro en Train                  : {metricas_train['f1_macro']:.4f}")
    print(f" * F1-Macro en Test                   : {metricas_test['f1_macro']:.4f}")
    print(f" * Cobertura de Reglas en Test        : {metricas_test['cobertura'] * 100:.2f}%")
    print("-" * 80)
    print(" Matriz de Confusion en Test (Filas: Real [Alta, Baja, Media], Col: Pred):")
    print(metricas_test["matriz_confusion"])
    print("=" * 80 + "\n")

    return base_integrada, metricas_train, metricas_test


if __name__ == "__main__":
    ejecutar_integracion_y_evaluacion()
