"""
Pipeline Maestro de Orquestación — Clasificación de Calidad de Vinos
Proyecto: Minería de Reglas (PRISM + Apriori), Lógica Difusa Mamdani y Optimización con Algoritmos Genéticos
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# Añadir directorio base al path
DIRECTORIO_BASE: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(DIRECTORIO_BASE)

from src.apriori import ejecutar_apriori_completo
from src.carga_datos import cargar_dataset
from src.fuzzy import (
    ConfiguracionMFs,
    SistemaDifusoMamdani,
    cargar_configuracion_mfs_json,
    cargar_reglas_integradas_json,
    ejecutar_smoke_test_fase_9,
    guardar_configuracion_mfs_json,
)
from src.genetico import (
    OptimizadorGeneticoFuzzy,
    ejecutar_smoke_test_fase_10,
)
from src.preparacion import (
    VARIABLES_FISICOQUIMICAS,
    categorizar_calidad,
    discretizar_por_cuantiles,
    preparar_conjuntos_entrenamiento_prueba,
)
from src.prism import ejecutar_induccion_prism
from src.reglas import (
    ClasificadorReglasDiscretas,
    ReglaUnificada,
    cargar_reglas_apriori_csv,
    cargar_reglas_prism_csv,
    guardar_base_integrada,
    integrar_base_reglas,
)
from src.visualizacion import generar_todos_los_graficos

# Rutas de artefactos generados
RUTA_RESULTS: str = os.path.join(DIRECTORIO_BASE, "results")
RUTA_PLOTS: str = os.path.join(RUTA_RESULTS, "plots")
RUTA_REGLAS_PRISM: str = os.path.join(RUTA_RESULTS, "reglas_prism.csv")
RUTA_REGLAS_APRIORI: str = os.path.join(RUTA_RESULTS, "reglas_apriori.csv")
RUTA_REGLAS_BASE_JSON: str = os.path.join(RUTA_RESULTS, "reglas_base_integrada.json")
RUTA_PARAMS_MFS_INICIALES_JSON: str = os.path.join(RUTA_RESULTS, "params_mfs_iniciales.json")
RUTA_PARAMS_MFS_OPTIMIZADAS_JSON: str = os.path.join(RUTA_RESULTS, "params_mfs_optimizadas_ga.json")
RUTA_REGLAS_OPTIMIZADAS_JSON: str = os.path.join(RUTA_RESULTS, "reglas_optimizadas_ga.json")
RUTA_HISTORIAL_GA_CSV: str = os.path.join(RUTA_RESULTS, "historial_convergencia_ga.csv")


def imprimir_banner() -> None:
    """Imprime el banner inicial del proyecto."""
    print("\n" + "=" * 90)
    print(" " * 15 + "PROYECTO: CLASIFICACION DE CALIDAD DE VINOS (RED WINE)")
    print(" " * 8 + "PRISM | APRIORI | LOGICA DIFUSA MAMDANI | ALGORITMOS GENETICOS (DEAP)")
    print("=" * 90)


def fase_1_a_4_preparacion_datos() -> Dict[str, Any]:
    """Ejecuta las Fases 1 a 4: Carga, Limpieza, Categorización y Discretización."""
    print("\n" + "=" * 80)
    print(" [FASE 1-4] CARGA, PREPARACION Y DISCRETIZACION DE DATOS (EQUAL-FREQUENCY 33%)")
    print("=" * 80)
    inicio = time.time()

    df_vinos = cargar_dataset()
    print(f" [+] Dataset cargado con éxito: {len(df_vinos)} filas x {len(df_vinos.columns)} columnas.")

    particiones = preparar_conjuntos_entrenamiento_prueba(df_vinos)
    print(f" [+] Partición Train: {len(particiones['X_train_num'])} muestras (80%)")
    print(f" [+] Partición Test : {len(particiones['X_test_num'])} muestras (20%)")
    print(f" [+] Distribución de Clases (Train): {particiones['y_train'].value_counts().to_dict()}")
    print(f" [+] Distribución de Clases (Test) : {particiones['y_test'].value_counts().to_dict()}")
    print(f" [OK] Tiempo de ejecución Fases 1-4: {time.time() - inicio:.2f} s")
    return particiones


def fase_5_induccion_prism() -> List[ReglaUnificada]:
    """Ejecuta la Fase 5: Inducción de Reglas con PRISM."""
    print("\n" + "=" * 80)
    print(" [FASE 5] INDUCCION DE REGLAS MODULARES CON ALGORITMO PRISM (DESDE CERO)")
    print("=" * 80)
    inicio = time.time()
    _, metricas = ejecutar_induccion_prism()
    reglas_prism = cargar_reglas_prism_csv(RUTA_REGLAS_PRISM)
    print(f" [+] PRISM indujo {len(reglas_prism)} reglas (Accuracy Test: {metricas['acc_test']*100:.2f}%).")
    print(f" [OK] Tiempo de ejecución Fase 5: {time.time() - inicio:.2f} s")
    return reglas_prism


def fase_6_mineria_apriori() -> List[ReglaUnificada]:
    """Ejecuta la Fase 6: Minería de Reglas Asociativas con Apriori."""
    print("\n" + "=" * 80)
    print(" [FASE 6] MINERIA DE REGLAS ASOCIATIVAS CON ALGORITMO APRIORI (DESDE CERO)")
    print("=" * 80)
    inicio = time.time()
    _, df_reglas = ejecutar_apriori_completo()
    reglas_apriori = cargar_reglas_apriori_csv(RUTA_REGLAS_APRIORI)
    print(f" [+] Apriori generó {len(df_reglas)} reglas asociativas filtradas ({len(reglas_apriori)} candidatas unificadas).")
    print(f" [OK] Tiempo de ejecución Fase 6: {time.time() - inicio:.2f} s")
    return reglas_apriori


def fase_7_integracion_reglas(
    reglas_prism: Optional[List[ReglaUnificada]] = None,
    reglas_apriori: Optional[List[ReglaUnificada]] = None,
) -> Tuple[List[ReglaUnificada], Dict[str, Any]]:
    """Ejecuta la Fase 7: Integración y Selección del Modelo Base Discreto."""
    print("\n" + "=" * 80)
    print(" [FASE 7] INTEGRACION DE REGLAS Y EVALUACION DEL MODELO BASE DISCRETO")
    print("=" * 80)
    inicio = time.time()

    if reglas_prism is None:
        reglas_prism = cargar_reglas_prism_csv(RUTA_REGLAS_PRISM)
    if reglas_apriori is None:
        reglas_apriori = cargar_reglas_apriori_csv(RUTA_REGLAS_APRIORI)

    base_integrada = integrar_base_reglas(reglas_prism, reglas_apriori)
    guardar_base_integrada(base_integrada)
    print(f" [+] Base Integrada generada: {len(base_integrada)} reglas unificadas.")

    # Evaluación rápida del baseline
    df_vinos = cargar_dataset()
    datos = preparar_conjuntos_entrenamiento_prueba(df_vinos)
    clf_disc = ClasificadorReglasDiscretas(base_integrada)
    met_test = clf_disc.evaluar(datos["X_test_disc"], datos["y_test"])

    print(f" [+] Baseline Discreto Test Accuracy : {met_test['exactitud']*100:.2f}%")
    print(f" [+] Baseline Discreto Test F1-Macro : {met_test['f1_macro']:.4f}")
    print(f" [+] Baseline Discreto Test Cobertura: {met_test['cobertura']*100:.2f}%")
    print(f" [OK] Tiempo de ejecución Fase 7: {time.time() - inicio:.2f} s")
    return base_integrada, met_test


def fase_8_9_fuzzy_mamdani(
    reglas_base: Optional[List[ReglaUnificada]] = None,
    particiones: Optional[Dict[str, Any]] = None,
) -> Tuple[SistemaDifusoMamdani, Dict[str, Any]]:
    """Ejecuta las Fases 8 y 9: Diseño difuso y evaluación Pre-AG."""
    print("\n" + "=" * 80)
    print(" [FASE 8-9] SISTEMA DE INFERENCIA DIFUSA MAMDANI Y EVALUACION PRE-AG")
    print("=" * 80)
    inicio = time.time()

    if particiones is None:
        df_vinos = cargar_dataset()
        particiones = preparar_conjuntos_entrenamiento_prueba(df_vinos)

    if reglas_base is None:
        reglas_base = cargar_reglas_integradas_json(RUTA_REGLAS_BASE_JSON)

    config_mfs_inicial = ConfiguracionMFs.desde_dataframe(particiones["X_train_num"])
    guardar_configuracion_mfs_json(config_mfs_inicial, RUTA_PARAMS_MFS_INICIALES_JSON)

    sistema_mamdani = SistemaDifusoMamdani(config_mfs=config_mfs_inicial, reglas=reglas_base)
    met_test = sistema_mamdani.evaluar(particiones["X_test_num"], particiones["y_test"])

    print(f" [+] Pre-AG Difuso Test Accuracy : {met_test['exactitud']*100:.2f}%")
    print(f" [+] Pre-AG Difuso Test F1-Macro : {met_test['f1_macro']:.4f}")
    print(f" [+] Pre-AG Difuso Test Cobertura: {met_test['cobertura']*100:.2f}%")
    print(f" [OK] Tiempo de ejecución Fases 8-9: {time.time() - inicio:.2f} s")
    return sistema_mamdani, met_test


def fase_10_11_optimizacion_genetica(
    particiones: Optional[Dict[str, Any]] = None,
    reglas_base: Optional[List[ReglaUnificada]] = None,
    num_generaciones: int = 30,
    tamano_poblacion: int = 50,
) -> Dict[str, Any]:
    """Ejecuta las Fases 10 y 11: Optimización con Algoritmo Genético (DEAP)."""
    print("\n" + "=" * 80)
    print(" [FASE 10-11] OPTIMIZACION CON ALGORITMO GENETICO (DEAP / NUMPY)")
    print(f" Hiperparámetros: Población={tamano_poblacion}, Generaciones={num_generaciones}, Cruce=0.80, Mutación=0.25")
    print("=" * 80)
    inicio = time.time()

    if particiones is None:
        df_vinos = cargar_dataset()
        particiones = preparar_conjuntos_entrenamiento_prueba(df_vinos)

    if reglas_base is None:
        reglas_base = cargar_reglas_integradas_json(RUTA_REGLAS_BASE_JSON)

    optimizador = OptimizadorGeneticoFuzzy(
        X_train=particiones["X_train_num"],
        y_train=particiones["y_train"],
        reglas_base=reglas_base,
        seed=42,
    )

    resultado = optimizador.optimizar(
        num_generaciones=num_generaciones,
        tamano_poblacion=tamano_poblacion,
        prob_cruce=0.80,
        prob_mutacion=0.25,
        tasa_elitismo=0.10,
        verbose=True,
    )

    optimizador.guardar_artefactos(
        resultado_optimizacion=resultado,
        ruta_mfs_json=RUTA_PARAMS_MFS_OPTIMIZADAS_JSON,
        ruta_reglas_json=RUTA_REGLAS_OPTIMIZADAS_JSON,
        ruta_historial_csv=RUTA_HISTORIAL_GA_CSV,
    )

    print(f" [+] Mejor Fitness alcanzado (Train): {resultado['mejor_fitness']:.4f}")
    print(f" [OK] Tiempo de ejecución Fases 10-11: {time.time() - inicio:.2f} s")
    return resultado


def fase_12_evaluacion_comparativa() -> Dict[str, Dict[str, Any]]:
    """
    Ejecuta la Fase 12: Evaluación comparativa rigurosa en 3 fases:
    - Fase 1: Baseline Discreto (PRISM + Apriori)
    - Fase 2: Pre-GA Difuso Mamdani
    - Fase 3: Post-GA Difuso Optimizado Mamdani
    """
    print("\n" + "=" * 90)
    print(" [FASE 12] EVALUACION COMPARATIVA RIGUROSA EN 3 FASES (TEST SET, N=320)")
    print("=" * 90)

    df_vinos = cargar_dataset()
    part = preparar_conjuntos_entrenamiento_prueba(df_vinos)
    X_train_num, X_test_num = part["X_train_num"], part["X_test_num"]
    X_train_disc, X_test_disc = part["X_train_disc"], part["X_test_disc"]
    y_train, y_test = part["y_train"], part["y_test"]

    # 1. Baseline Discreto
    reglas_base_raw = json.load(open(RUTA_REGLAS_BASE_JSON, "r", encoding="utf-8"))
    reglas_base = [ReglaUnificada(**r) for r in reglas_base_raw]
    clf_disc = ClasificadorReglasDiscretas(reglas_base)
    met_disc_train = clf_disc.evaluar(X_train_disc, y_train)
    met_disc_test = clf_disc.evaluar(X_test_disc, y_test)

    # 2. Pre-GA Difuso
    cfg_pre = cargar_configuracion_mfs_json(RUTA_PARAMS_MFS_INICIALES_JSON)
    sis_pre = SistemaDifusoMamdani(config_mfs=cfg_pre, reglas=reglas_base)
    met_pre_train = sis_pre.evaluar(X_train_num, y_train)
    met_pre_test = sis_pre.evaluar(X_test_num, y_test)

    # 3. Post-GA Difuso
    cfg_post = cargar_configuracion_mfs_json(RUTA_PARAMS_MFS_OPTIMIZADAS_JSON)
    reglas_post_raw = json.load(open(RUTA_REGLAS_OPTIMIZADAS_JSON, "r", encoding="utf-8"))
    reglas_post = [ReglaUnificada(**r) for r in reglas_post_raw]
    sis_post = SistemaDifusoMamdani(config_mfs=cfg_post, reglas=reglas_post)
    met_post_train = sis_post.evaluar(X_train_num, y_train)
    met_post_test = sis_post.evaluar(X_test_num, y_test)

    # Imprimir tabla comparativa en consola
    print("\n" + "-" * 90)
    print(f"{'FASE DEL SISTEMA':<32} | {'ACCURACY (TEST)':<16} | {'F1-MACRO (TEST)':<16} | {'COBERTURA':<12}")
    print("-" * 90)
    print(
        f"{'1. Baseline Discreto (PRISM+Apriori)':<32} | "
        f"{met_disc_test['exactitud']*100:>13.2f}% | "
        f"{met_disc_test['f1_macro']:>14.4f} | "
        f"{met_disc_test['cobertura']*100:>10.2f}%"
    )
    print(
        f"{'2. Sistema Difuso Pre-AG (Mamdani)':<32} | "
        f"{met_pre_test['exactitud']*100:>13.2f}% | "
        f"{met_pre_test['f1_macro']:>14.4f} | "
        f"{met_pre_test['cobertura']*100:>10.2f}%"
    )
    print(
        f"{'3. Sistema Difuso Post-AG (Optimizado)':<32} | "
        f"{met_post_test['exactitud']*100:>13.2f}% | "
        f"{met_post_test['f1_macro']:>14.4f} | "
        f"{met_post_test['cobertura']*100:>10.2f}%"
    )
    print("-" * 90)

    # Ganancias porcentuales
    ganancia_f1_baseline = (met_post_test["f1_macro"] - met_disc_test["f1_macro"]) * 100
    ganancia_acc_baseline = (met_post_test["exactitud"] - met_disc_test["exactitud"]) * 100
    ganancia_cob = (met_post_test["cobertura"] - met_disc_test["cobertura"]) * 100
    print(
        f" [*] Impacto del AG frente a Baseline Discreto: "
        f"F1-Macro: +{ganancia_f1_baseline:.2f} pts | "
        f"Accuracy: +{ganancia_acc_baseline:.2f} pts | "
        f"Cobertura: +{ganancia_cob:.2f} pts"
    )
    print("=" * 90 + "\n")

    return {
        "baseline_discreto": {"train": met_disc_train, "test": met_disc_test},
        "pre_ga": {"train": met_pre_train, "test": met_pre_test},
        "post_ga": {"train": met_post_train, "test": met_post_test},
    }


def fase_13_visualizacion_final() -> Dict[str, str]:
    """Ejecuta la Fase 13: Generación de todos los gráficos."""
    return generar_todos_los_graficos()


def ejecutar_pipeline_completo(skip_ga: bool = False, generaciones: int = 30, poblacion: int = 50) -> None:
    """Ejecuta el pipeline maestro de punta a punta."""
    imprimir_banner()
    tiempo_total_inicio = time.time()

    # Fases 1 a 4
    particiones = fase_1_a_4_preparacion_datos()

    # Fase 5
    reglas_prism = fase_5_induccion_prism()

    # Fase 6
    reglas_apriori = fase_6_mineria_apriori()

    # Fase 7
    reglas_base, _ = fase_7_integracion_reglas(reglas_prism, reglas_apriori)

    # Fases 8 y 9
    sistema_mamdani, _ = fase_8_9_fuzzy_mamdani(reglas_base, particiones)

    # Fases 10 y 11
    if not skip_ga or not os.path.exists(RUTA_PARAMS_MFS_OPTIMIZADAS_JSON):
        fase_10_11_optimizacion_genetica(
            particiones=particiones,
            reglas_base=reglas_base,
            num_generaciones=generaciones,
            tamano_poblacion=poblacion,
        )
    else:
        print("\n" + "=" * 80)
        print(" [INFO] Se omitió la re-ejecución del AG (--skip-ga). Usando artefactos existentes.")
        print("=" * 80)

    # Fase 12
    fase_12_evaluacion_comparativa()

    # Fase 13
    fase_13_visualizacion_final()

    duracion_total = time.time() - tiempo_total_inicio
    print("\n" + "=" * 90)
    print(f" [FIN] PIPELINE MAESTRO COMPLETADO EXITOSAMENTE EN {duracion_total:.2f} SEGUNDOS")
    print(" Todos los artefactos y figuras están listos en 'results/' y 'results/plots/'.")
    print("=" * 90 + "\n")


def parsear_argumentos() -> argparse.Namespace:
    """Configura los argumentos de la línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Pipeline Maestro de Clasificación de Calidad de Vinos (PRISM + Apriori + Fuzzy + GA)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Ejecuta de punta a punta todo el proyecto (Fases 1 a 13).",
    )
    parser.add_argument(
        "--fase",
        type=str,
        default=None,
        choices=["1-4", "prism", "apriori", "reglas", "fuzzy", "genetico", "evaluacion", "visualizacion"],
        help="Ejecuta una fase individual específica.",
    )
    parser.add_argument(
        "--skip-ga",
        action="store_true",
        help="Omite el entrenamiento del AG si ya existen artefactos optimizados en results/.",
    )
    parser.add_argument(
        "--generaciones",
        type=int,
        default=30,
        help="Número de generaciones evolutivas para el AG (default: 30).",
    )
    parser.add_argument(
        "--poblacion",
        type=int,
        default=50,
        help="Tamaño de la población para el AG (default: 50).",
    )
    return parser.parse_args()


def main() -> None:
    """Punto de entrada principal."""
    args = parsear_argumentos()

    if args.fase:
        imprimir_banner()
        if args.fase == "1-4":
            fase_1_a_4_preparacion_datos()
        elif args.fase == "prism":
            fase_5_induccion_prism()
        elif args.fase == "apriori":
            fase_6_mineria_apriori()
        elif args.fase == "reglas":
            fase_7_integracion_reglas()
        elif args.fase == "fuzzy":
            fase_8_9_fuzzy_mamdani()
        elif args.fase == "genetico":
            fase_10_11_optimizacion_genetica(
                num_generaciones=args.generaciones,
                tamano_poblacion=args.poblacion,
            )
        elif args.fase == "evaluacion":
            fase_12_evaluacion_comparativa()
        elif args.fase == "visualizacion":
            fase_13_visualizacion_final()
    else:
        ejecutar_pipeline_completo(
            skip_ga=args.skip_ga,
            generaciones=args.generaciones,
            poblacion=args.poblacion,
        )


if __name__ == "__main__":
    main()
