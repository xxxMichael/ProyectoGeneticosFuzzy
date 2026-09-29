"""
Módulo del Algoritmo Genético (DEAP / NumPy) para Optimización de Sistemas Difusos
Proyecto: Clasificación de Calidad de Vinos (PRISM, Apriori, Lógica Difusa y Algoritmo Genético)
"""

import json
import os
import random
import sys
import time
from dataclasses import asdict
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from deap import base, creator, tools

from src.datos.carga.cargador import cargar_dataset
from src.datos.preparacion.discretizacion import (
    VARIABLES_FISICOQUIMICAS,
    preparar_conjuntos_entrenamiento_prueba,
)
from src.fuzzy.fuzzy import (
    ConfiguracionMFs,
    ClasificadorFuzzyMamdani,
    RANGOS_VARIABLES_DEFAULT,
    UNIVERSO_CALIDAD,
    PARAMS_SALIDA_CALIDAD,
    UMBRAL_CORTE_BAJA_MEDIA,
    UMBRAL_CORTE_MEDIA_ALTA,
)
from src.reglas.regla_unificada import ReglaUnificada
from src.reglas.integracion.unificador import (
    cargar_reglas_prism_csv,
    cargar_reglas_apriori_csv,
    integrar_base_reglas,
)
from src.genetico.cromosoma.estructura import (
    LONGITUD_CROMOSOMA_MFS,
    LONGITUD_CROMOSOMA_REGLAS,
    LONGITUD_TOTAL_CROMOSOMA,
    UMBRAL_ACTIVACION_REGLA,
    decodificar_cromosoma,
)
from src.genetico.cromosoma.reparacion import reparar_individuo
from src.genetico.poblacion.inicializacion import (
    crear_individuo_semilla,
    crear_individuo_perturbado,
)
from src.genetico.cruce.cruce import cruzar_individuos_mixtos
from src.genetico.mutacion.mutacion import mutar_individuo_mixto
from src.genetico.fitness.evaluacion import (
    PESO_COBERTURA,
    PESO_F1_MACRO,
    PESO_PENALIZACION_REGLAS,
    evaluar_individuo_fitness,
)

# =============================================================================
# CONSTANTES Y CONFIGURACIÓN DEL ALGORITMO GENÉTICO
# =============================================================================
DIRECTORIO_BASE: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUTA_REGLAS_INTEGRADAS_JSON: str = os.path.join(DIRECTORIO_BASE, "results", "reglas_base_integrada.json")
RUTA_PARAMS_MFS_INICIALES_JSON: str = os.path.join(DIRECTORIO_BASE, "results", "params_mfs_iniciales.json")
RUTA_SALIDA_MFS_OPTIMIZADAS_JSON: str = os.path.join(DIRECTORIO_BASE, "results", "params_mfs_optimizadas_ga.json")
RUTA_SALIDA_REGLAS_OPTIMIZADAS_JSON: str = os.path.join(DIRECTORIO_BASE, "results", "reglas_optimizadas_ga.json")
RUTA_SALIDA_HISTORIAL_CONVERGENCIA_CSV: str = os.path.join(DIRECTORIO_BASE, "results", "historial_convergencia_ga.csv")

# Hiperparámetros evolutivos por defecto
TAMANO_POBLACION: int = 50
NUM_GENERACIONES: int = 30
PROBABILIDAD_CRUCE: float = 0.80
PROBABILIDAD_MUTACION: float = 0.25
TASA_ELITISMO: float = 0.10
METODO_SELECCION: str = "ruleta"
SEED_ALEATORIA: int = 42
# =============================================================================


class OptimizadorGeneticoFuzzy:
    """
    Motor evolutivo modular basado en DEAP y NumPy para la calibración simultánea de
    funciones de pertenencia y selección/ponderación de reglas difusas.
    """

    def __init__(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        reglas_base: List[ReglaUnificada],
        config_mfs_inicial: Optional[ConfiguracionMFs] = None,
        seed: int = SEED_ALEATORIA,
        peso_f1: float = PESO_F1_MACRO,
        peso_cobertura: float = PESO_COBERTURA,
        peso_penalizacion: float = PESO_PENALIZACION_REGLAS,
        umbral_activacion: float = UMBRAL_ACTIVACION_REGLA,
    ) -> None:
        self.X_train = X_train
        self.y_train = y_train
        self.reglas_base = reglas_base
        self.seed = seed
        self.peso_f1 = peso_f1
        self.peso_cobertura = peso_cobertura
        self.peso_penalizacion = peso_penalizacion
        self.umbral_activacion = umbral_activacion

        self.X_mat = self.X_train[VARIABLES_FISICOQUIMICAS].to_numpy(dtype=float)
        self.y_vec = np.asarray(self.y_train)

        if config_mfs_inicial is None:
            self.config_mfs_inicial = ConfiguracionMFs.desde_dataframe(self.X_train)
        else:
            self.config_mfs_inicial = config_mfs_inicial

        random.seed(self.seed)
        np.random.seed(self.seed)

        self.toolbox = self._configurar_deap()

    def _configurar_deap(self) -> base.Toolbox:
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "IndividuoFuzzy"):
            creator.create("IndividuoFuzzy", list, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()
        individuo_semilla = crear_individuo_semilla(self.config_mfs_inicial, len(self.reglas_base))

        def generar_individuo() -> creator.IndividuoFuzzy:
            ind_pert = crear_individuo_perturbado(individuo_semilla)
            return creator.IndividuoFuzzy(ind_pert)

        toolbox.register("individual", generar_individuo)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        toolbox.register("mate", cruzar_individuos_mixtos)
        toolbox.register("mutate", mutar_individuo_mixto)
        toolbox.register("select", tools.selRoulette)

        def evaluador(ind: List[float]) -> Tuple[float]:
            return evaluar_individuo_fitness(
                individuo=ind,
                reglas_base=self.reglas_base,
                X_matriz=self.X_mat,
                y_vector=self.y_vec,
                peso_f1=self.peso_f1,
                peso_cobertura=self.peso_cobertura,
                peso_penalizacion=self.peso_penalizacion,
                umbral_activacion=self.umbral_activacion,
            )

        toolbox.register("evaluate", evaluador)
        return toolbox

    def optimizar(
        self,
        num_generaciones: int = NUM_GENERACIONES,
        tamano_poblacion: int = TAMANO_POBLACION,
        prob_cruce: float = PROBABILIDAD_CRUCE,
        prob_mutacion: float = PROBABILIDAD_MUTACION,
        tasa_elitismo: float = TASA_ELITISMO,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        inicio_tiempo = time.time()
        n_elites = max(1, int(tasa_elitismo * tamano_poblacion))

        if verbose:
            print(f"\n" + "=" * 80)
            print(f" [ALGORITMO GENETICO] INICIANDO OPTIMIZACION DIFUSA EVOLUTIVA")
            print(f"=" * 80)
            print(f" * Poblacion: {tamano_poblacion} | Generaciones: {num_generaciones} | Elitismo: {n_elites} individuos | Seleccion: Ruleta")
            print(f" * Prob. Cruce: {prob_cruce * 100:.1f}% | Prob. Mutacion: {prob_mutacion * 100:.1f}%")
            print(f" * Ponderaciones Fitness -> F1: {self.peso_f1:.2f} | Cob: {self.peso_cobertura:.2f} | Pen: {self.peso_penalizacion:.2f}")
            print(f" * Cromosoma: {LONGITUD_TOTAL_CROMOSOMA} genes ({LONGITUD_CROMOSOMA_MFS} MFs + {LONGITUD_CROMOSOMA_REGLAS} Reglas)")
            print("-" * 80)
            print(f"{'Gen':<5} | {'Mejor Fit':<11} | {'Fit Prom':<11} | {'F1-Macro':<10} | {'Cobertura':<10} | {'Reglas Act':<10} | {'Tiempo':<7}")
            print("-" * 80)

        poblacion = self.toolbox.population(n=tamano_poblacion)
        ind_semilla = creator.IndividuoFuzzy(crear_individuo_semilla(self.config_mfs_inicial, len(self.reglas_base)))
        poblacion[0] = ind_semilla

        fitnesses = list(map(self.toolbox.evaluate, poblacion))
        for ind, fit in zip(poblacion, fitnesses):
            ind.fitness.values = fit

        historial_progreso: List[Dict[str, Any]] = []

        mejor_ind_gen0 = tools.selBest(poblacion, 1)[0]
        cfg_mfs0, reglas0 = decodificar_cromosoma(mejor_ind_gen0, self.reglas_base, self.umbral_activacion)
        clf0 = ClasificadorFuzzyMamdani(cfg_mfs0, reglas0, umbral_peso_activa=self.umbral_activacion)
        met0 = clf0.evaluar(self.X_mat, self.y_vec)

        fits_gen0 = [ind.fitness.values[0] for ind in poblacion]
        reg_gen0 = {
            "generacion": 0,
            "mejor_fitness": float(max(fits_gen0)),
            "fitness_promedio": float(np.mean(fits_gen0)),
            "f1_macro": met0["f1_macro"],
            "cobertura": met0["cobertura"],
            "reglas_activas": met0["reglas_activas"],
            "tiempo_seg": float(time.time() - inicio_tiempo),
        }
        historial_progreso.append(reg_gen0)

        if verbose:
            print(
                f"{0:<5} | {reg_gen0['mejor_fitness']:<11.4f} | {reg_gen0['fitness_promedio']:<11.4f} | "
                f"{reg_gen0['f1_macro']:<10.4f} | {reg_gen0['cobertura'] * 100:<9.2f}% | "
                f"{reg_gen0['reglas_activas']:<10} | {reg_gen0['tiempo_seg']:<6.2f}s"
            )

        for gen in range(1, num_generaciones + 1):
            elites = [self.toolbox.clone(ind) for ind in tools.selBest(poblacion, n_elites)]
            n_descendientes = tamano_poblacion - n_elites
            seleccionados = self.toolbox.select(poblacion, n_descendientes)
            descendencia = [self.toolbox.clone(ind) for ind in seleccionados]

            for i in range(0, len(descendencia) - 1, 2):
                if random.random() < prob_cruce:
                    self.toolbox.mate(descendencia[i], descendencia[i + 1])
                    del descendencia[i].fitness.values
                    del descendencia[i + 1].fitness.values

            for ind in descendencia:
                if random.random() < prob_mutacion:
                    self.toolbox.mutate(ind)
                    del ind.fitness.values

            invalidos = [ind for ind in descendencia if not ind.fitness.valid]
            fitnesses_inv = list(map(self.toolbox.evaluate, invalidos))
            for ind, fit in zip(invalidos, fitnesses_inv):
                ind.fitness.values = fit

            poblacion[:] = elites + descendencia

            fits_gen = [ind.fitness.values[0] for ind in poblacion]
            mejor_ind_gen = tools.selBest(poblacion, 1)[0]
            cfg_mfs_g, reglas_g = decodificar_cromosoma(mejor_ind_gen, self.reglas_base, self.umbral_activacion)
            clf_g = ClasificadorFuzzyMamdani(cfg_mfs_g, reglas_g, umbral_peso_activa=self.umbral_activacion)
            met_g = clf_g.evaluar(self.X_mat, self.y_vec)

            reg_gen = {
                "generacion": gen,
                "mejor_fitness": float(max(fits_gen)),
                "fitness_promedio": float(np.mean(fits_gen)),
                "f1_macro": met_g["f1_macro"],
                "cobertura": met_g["cobertura"],
                "reglas_activas": met_g["reglas_activas"],
                "tiempo_seg": float(time.time() - inicio_tiempo),
            }
            historial_progreso.append(reg_gen)

            if verbose and (gen % max(1, num_generaciones // 10) == 0 or gen == num_generaciones):
                print(
                    f"{gen:<5} | {reg_gen['mejor_fitness']:<11.4f} | {reg_gen['fitness_promedio']:<11.4f} | "
                    f"{reg_gen['f1_macro']:<10.4f} | {reg_gen['cobertura'] * 100:<9.2f}% | "
                    f"{reg_gen['reglas_activas']:<10} | {reg_gen['tiempo_seg']:<6.2f}s"
                )

        mejor_individuo = tools.selBest(poblacion, 1)[0]
        config_mfs_opt, reglas_opt = decodificar_cromosoma(mejor_individuo, self.reglas_base, self.umbral_activacion)
        clasificador_opt = ClasificadorFuzzyMamdani(config_mfs_opt, reglas_opt, umbral_peso_activa=self.umbral_activacion)
        metricas_finales_train = clasificador_opt.evaluar(self.X_mat, self.y_vec)

        tiempo_total = time.time() - inicio_tiempo
        if verbose:
            print("-" * 80)
            print(f" [+] OPTIMIZACION COMPLETADA EN {tiempo_total:.2f} segundos.")
            print(f" * Mejor Fitness Alcanzado : {mejor_individuo.fitness.values[0]:.4f}")
            print(f" * F1-Macro en Train       : {metricas_finales_train['f1_macro']:.4f}")
            print(f" * Cobertura en Train      : {metricas_finales_train['cobertura'] * 100:.2f}%")
            print(f" * Reglas Activas Finales  : {metricas_finales_train['reglas_activas']} / {len(self.reglas_base)}")
            print("=" * 80 + "\n")

        resultado = {
            "mejor_individuo": list(mejor_individuo),
            "mejor_fitness": float(mejor_individuo.fitness.values[0]),
            "historial_progreso": historial_progreso,
            "config_mfs_optimizada": config_mfs_opt,
            "reglas_optimizadas": reglas_opt,
            "clasificador_opt": clasificador_opt,
            "metricas_train": metricas_finales_train,
            "tiempo_total_seg": tiempo_total,
        }

        return resultado

    def guardar_artefactos(
        self,
        resultado_optimizacion: Optional[Dict[str, Any]] = None,
        ruta_mfs_json: str = RUTA_SALIDA_MFS_OPTIMIZADAS_JSON,
        ruta_reglas_json: str = RUTA_SALIDA_REGLAS_OPTIMIZADAS_JSON,
        ruta_historial_csv: str = RUTA_SALIDA_HISTORIAL_CONVERGENCIA_CSV,
        **kwargs: Any,
    ) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(ruta_mfs_json)), exist_ok=True)

        res = resultado_optimizacion if resultado_optimizacion is not None else kwargs.get("resultado")

        if res is not None:
            config_mfs: ConfiguracionMFs = res["config_mfs_optimizada"]
            reglas: List[ReglaUnificada] = res["reglas_optimizadas"]
            historial = res["historial_progreso"]
        else:
            mejor_ind = kwargs.get("mejor_individuo")
            if mejor_ind is None:
                raise ValueError("Se requiere 'resultado_optimizacion' o 'mejor_individuo' para guardar artefactos.")
            config_mfs, reglas = decodificar_cromosoma(mejor_ind, self.reglas_base, self.umbral_activacion)
            historial = kwargs.get("historial", [])

        with open(ruta_mfs_json, "w", encoding="utf-8") as f:
            json.dump(config_mfs.parametros, f, indent=2)

        ruta_mejores = os.path.join(DIRECTORIO_BASE, "results", "mejores_params.json")
        with open(ruta_mejores, "w", encoding="utf-8") as f:
            json.dump(config_mfs.parametros, f, indent=2)

        with open(ruta_reglas_json, "w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in reglas], f, indent=2, ensure_ascii=False)

        if historial:
            df_hist = pd.DataFrame(historial)
            df_hist.to_csv(ruta_historial_csv, index=False, encoding="utf-8")

        print(f" [+] Artefactos exportados exitosamente:")
        print(f"   - MFs Optimizadas        : {ruta_mfs_json}")
        print(f"   - Mejores Params (Sync)  : {ruta_mejores}")
        print(f"   - Reglas Optimizadas     : {ruta_reglas_json}")
        print(f"   - Historial Convergencia : {ruta_historial_csv}")


def ejecutar_smoke_test_fase_10() -> bool:
    print("\n" + "=" * 80)
    print(" [FASE 10] SMOKE TEST DE VALIDACION DEL ALGORITMO GENETICO (DEAP / NUMPY)")
    print("=" * 80)

    df_vinos = cargar_dataset()
    particiones = preparar_conjuntos_entrenamiento_prueba(df_vinos)
    X_train_num = particiones["X_train_num"]
    y_train = particiones["y_train"]

    if os.path.exists(RUTA_REGLAS_INTEGRADAS_JSON):
        with open(RUTA_REGLAS_INTEGRADAS_JSON, "r", encoding="utf-8") as f:
            datos_reglas = json.load(f)
        reglas_base = [ReglaUnificada(**r) for r in datos_reglas]
    else:
        reglas_prism = cargar_reglas_prism_csv(os.path.join(DIRECTORIO_BASE, "results", "reglas_prism.csv"))
        reglas_apriori = cargar_reglas_apriori_csv(os.path.join(DIRECTORIO_BASE, "results", "reglas_apriori.csv"))
        reglas_base = integrar_base_reglas(reglas_prism, reglas_apriori)

    optimizador = OptimizadorGeneticoFuzzy(
        X_train=X_train_num,
        y_train=y_train,
        reglas_base=reglas_base,
        seed=42,
    )

    resultado = optimizador.optimizar(
        num_generaciones=2,
        tamano_poblacion=10,
        prob_cruce=0.80,
        prob_mutacion=0.25,
        tasa_elitismo=0.10,
        verbose=True,
    )

    mejor_ind = resultado["mejor_individuo"]
    historial = resultado["historial_progreso"]

    dim_correcta = len(mejor_ind) == LONGITUD_TOTAL_CROMOSOMA
    num_generaciones_ejecutadas = len(historial) == 3
    fitness_valido = not np.isnan(resultado["mejor_fitness"]) and resultado["mejor_fitness"] > 0.0

    exito = dim_correcta and num_generaciones_ejecutadas and fitness_valido
    if exito:
        print(" [OK] Smoke Test de Fase 10 COMPLETADO CON EXITO.\n")
    else:
        print(" [ERROR] El Smoke Test de Fase 10 ha fallado.\n")

    return exito


if __name__ == "__main__":
    ejecutar_smoke_test_fase_10()
