"""
Módulo de Evaluación Comparativa Progresiva (Fase 12)
Proyecto: Clasificación de Calidad de Vinos (PRISM, Apriori, Lógica Difusa y Algoritmo Genético)
"""

import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.datos.carga.cargador import cargar_dataset
from src.datos.preparacion.discretizacion import preparar_conjuntos_entrenamiento_prueba
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
from src.evaluacion.metricas.calculo import (
    ORDEN_CLASES,
    calcular_metricas_clasificacion,
)
from src.evaluacion.comparacion.comparador import (
    UMBRAL_PESO_REGLA_ACTIVA,
    construir_dataframe_comparativo,
    evaluar_fase_1_discreto,
    evaluar_fase_2_pre_ga,
    evaluar_fase_3_post_ga,
)

# =============================================================================
# CONSTANTES Y CONFIGURACIÓN DE EVALUACIÓN COMPARATIVA
# =============================================================================
DIRECTORIO_BASE: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUTA_REGLAS_BASE_JSON: str = os.path.join(DIRECTORIO_BASE, "results", "reglas_base_integrada.json")
RUTA_PARAMS_MFS_INICIALES_JSON: str = os.path.join(DIRECTORIO_BASE, "results", "params_mfs_iniciales.json")
RUTA_PARAMS_MFS_OPTIMIZADAS_JSON: str = os.path.join(DIRECTORIO_BASE, "results", "mejores_params.json")
RUTA_REGLAS_OPTIMIZADAS_JSON: str = os.path.join(DIRECTORIO_BASE, "results", "reglas_optimizadas_ga.json")

RUTA_SALIDA_EVALUACION_CSV: str = os.path.join(DIRECTORIO_BASE, "results", "resultados_evaluacion.csv")
RUTA_SALIDA_METRICAS_JSON: str = os.path.join(DIRECTORIO_BASE, "results", "metricas_completas.json")
# =============================================================================


def cargar_reglas_desde_json(ruta_json: str) -> List[ReglaUnificada]:
    if not os.path.exists(ruta_json):
        raise FileNotFoundError(f"No se encontro el archivo de reglas: {ruta_json}")
    with open(ruta_json, "r", encoding="utf-8") as f:
        datos = json.load(f)
    return [ReglaUnificada(**r) for r in datos]


def cargar_configuracion_mfs(ruta_json: str) -> ConfiguracionMFs:
    if not os.path.exists(ruta_json):
        raise FileNotFoundError(f"No se encontro el archivo de parametros MFs: {ruta_json}")
    with open(ruta_json, "r", encoding="utf-8") as f:
        datos = json.load(f)
    return ConfiguracionMFs(parametros=datos)


def guardar_resultados_evaluacion(
    df_comparativo: pd.DataFrame,
    metricas_completas: Dict[str, Any],
    ruta_csv: str = RUTA_SALIDA_EVALUACION_CSV,
    ruta_json: str = RUTA_SALIDA_METRICAS_JSON,
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(ruta_csv)), exist_ok=True)
    df_comparativo.to_csv(ruta_csv, index=False, encoding="utf-8")

    with open(ruta_json, "w", encoding="utf-8") as f:
        json.dump(metricas_completas, f, indent=2, ensure_ascii=False)


def mostrar_tabla_comparativa(df_comparativo: pd.DataFrame) -> None:
    print("\n" + "=" * 105)
    print(" [FASE 12] TABLA COMPARATIVA PROGRESIVA DE RENDIMIENTO (TRAIN Y TEST)")
    print("=" * 105)
    print(
        f"{'Fase Evaluada':<38} | {'Set':<5} | {'Exactitud':<10} | {'F1-Macro':<9} | {'F1-Weig.':<9} | {'Cobertura':<10} | {'Reglas'}"
    )
    print("-" * 105)
    for _, fila in df_comparativo.iterrows():
        print(
            f"{fila['Fase']:<38} | {fila['Conjunto']:<5} | {fila['Exactitud (%)']:>8.2f}% | "
            f"{fila['F1-Macro']:>9.4f} | {fila['F1-Weighted']:>9.4f} | {fila['Cobertura (%)']:>8.2f}% | "
            f"{fila['Reglas Activas']}"
        )
    print("=" * 105 + "\n")


def ejecutar_evaluacion_comparativa_completa(
    ruta_reglas_base: str = RUTA_REGLAS_BASE_JSON,
    ruta_mfs_iniciales: str = RUTA_PARAMS_MFS_INICIALES_JSON,
    ruta_mfs_optimizadas: str = RUTA_PARAMS_MFS_OPTIMIZADAS_JSON,
    ruta_reglas_optimizadas: str = RUTA_REGLAS_OPTIMIZADAS_JSON,
    ruta_csv_salida: str = RUTA_SALIDA_EVALUACION_CSV,
    ruta_json_salida: str = RUTA_SALIDA_METRICAS_JSON,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    print("\n" + "=" * 95)
    print(" [FASE 12] INICIO DE LA EVALUACION COMPARATIVA PROGRESIVA")
    print("=" * 95)

    df_vinos = cargar_dataset()
    particiones = preparar_conjuntos_entrenamiento_prueba(df_vinos)

    reglas_base = cargar_reglas_desde_json(ruta_reglas_base)
    config_mfs_ini = cargar_configuracion_mfs(ruta_mfs_iniciales)

    if not os.path.exists(ruta_mfs_optimizadas):
        ruta_alt = os.path.join(DIRECTORIO_BASE, "results", "params_mfs_optimizadas_ga.json")
        if os.path.exists(ruta_alt):
            ruta_mfs_optimizadas = ruta_alt

    config_mfs_opt = cargar_configuracion_mfs(ruta_mfs_optimizadas)
    reglas_opt = cargar_reglas_desde_json(ruta_reglas_optimizadas)

    print(" [1/3] Evaluando Fase 1: Baseline Discreto...")
    res_fase1 = evaluar_fase_1_discreto(reglas_base, particiones)

    print(" [2/3] Evaluando Fase 2: Pre-GA Difuso Inicial...")
    res_fase2 = evaluar_fase_2_pre_ga(config_mfs_ini, reglas_base, particiones)

    print(" [3/3] Evaluando Fase 3: Post-GA Difuso Optimizado...")
    res_fase3 = evaluar_fase_3_post_ga(config_mfs_opt, reglas_opt, particiones)

    df_comparativo = construir_dataframe_comparativo(res_fase1, res_fase2, res_fase3)
    mostrar_tabla_comparativa(df_comparativo)

    metricas_completas = {
        "fase_1_baseline_discreto": res_fase1,
        "fase_2_pre_ga_difuso_inicial": res_fase2,
        "fase_3_post_ga_difuso_optimizado": res_fase3,
    }

    guardar_resultados_evaluacion(
        df_comparativo=df_comparativo,
        metricas_completas=metricas_completas,
        ruta_csv=ruta_csv_salida,
        ruta_json=ruta_json_salida,
    )
    print(f" [+] Evaluacion completada. Resultados guardados en: {ruta_csv_salida}")

    return df_comparativo, metricas_completas


if __name__ == "__main__":
    ejecutar_evaluacion_comparativa_completa()
