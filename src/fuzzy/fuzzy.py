"""
Módulo del Sistema de Inferencia Difusa Mamdani — Variables, Funciones de Pertenencia y Clasificación
Proyecto: Clasificación de Calidad de Vinos (PRISM, Apriori, Lógica Difusa y Algoritmo Genético)
"""

import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from src.datos.carga.cargador import cargar_dataset
from src.datos.preparacion.discretizacion import (
    VARIABLES_FISICOQUIMICAS,
    categorizar_calidad,
    preparar_conjuntos_entrenamiento_prueba,
)
from src.reglas.regla_unificada import ReglaUnificada
from src.fuzzy.pertenencia.funciones import (
    RANGOS_VARIABLES_DEFAULT,
    ConfiguracionMFs,
    calcular_pertenencia_trapezoidal,
    calcular_pertenencia_triangular,
    validar_cobertura_difusa,
)
from src.fuzzy.inferencia.motor import (
    UMBRAL_ACTIVACION_MINIMA,
    agregar_salida_difusa,
    evaluar_activaciones_reglas,
)
from src.fuzzy.defuzzificacion.centroide import (
    CENTROIDE_NEUTRO_FALLBACK,
    CLASE_NEUTRA_FALLBACK,
    PARAMS_SALIDA_CALIDAD,
    UMBRAL_CORTE_BAJA_MEDIA,
    UMBRAL_CORTE_MEDIA_ALTA,
    UNIVERSO_CALIDAD,
    clasificar_centroide,
    defuzzificar_centroide,
)

# =============================================================================
# CONSTANTES Y CONFIGURACIÓN DEL SISTEMA DIFUSO
# =============================================================================
DIRECTORIO_BASE: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUTA_PARAMETROS_DEFAULT_JSON: str = os.path.join(DIRECTORIO_BASE, "results", "params_mfs_iniciales.json")
RUTA_REGLAS_INTEGRADAS_JSON: str = os.path.join(DIRECTORIO_BASE, "results", "reglas_base_integrada.json")
ORDEN_CLASES_EVALUACION: List[str] = ["Alta", "Baja", "Media"]
# =============================================================================


def cargar_reglas_integradas_json(ruta_json: str = RUTA_REGLAS_INTEGRADAS_JSON) -> List[ReglaUnificada]:
    """Carga la base de reglas integradas desde un archivo JSON."""
    if not os.path.exists(ruta_json):
        return []
    with open(ruta_json, "r", encoding="utf-8") as f:
        datos = json.load(f)
    return [ReglaUnificada(**d) for d in datos]


def cargar_configuracion_mfs_json(ruta_json: str = RUTA_PARAMETROS_DEFAULT_JSON) -> ConfiguracionMFs:
    """Carga los parámetros de las funciones de pertenencia desde un archivo JSON."""
    if not os.path.exists(ruta_json):
        return ConfiguracionMFs()
    with open(ruta_json, "r", encoding="utf-8") as f:
        parametros = json.load(f)
    return ConfiguracionMFs(parametros=parametros)


def guardar_configuracion_mfs_json(
    config_mfs: ConfiguracionMFs,
    ruta_json: str = RUTA_PARAMETROS_DEFAULT_JSON,
) -> None:
    """Exporta los parámetros de las funciones de pertenencia a un archivo JSON."""
    os.makedirs(os.path.dirname(os.path.abspath(ruta_json)), exist_ok=True)
    with open(ruta_json, "w", encoding="utf-8") as f:
        json.dump(config_mfs.parametros, f, indent=2)


class SistemaDifusoMamdani:
    """
    Motor de Inferencia Difusa Mamdani con Defuzzificación por Centroide Cuantitativo.
    """

    def __init__(
        self,
        config_mfs: Optional[ConfiguracionMFs] = None,
        reglas: Optional[List[ReglaUnificada]] = None,
        universo_calidad: np.ndarray = UNIVERSO_CALIDAD,
        params_salida: Dict[str, List[float]] = PARAMS_SALIDA_CALIDAD,
        umbral_baja_media: float = UMBRAL_CORTE_BAJA_MEDIA,
        umbral_media_alta: float = UMBRAL_CORTE_MEDIA_ALTA,
        centroide_fallback: float = CENTROIDE_NEUTRO_FALLBACK,
        clase_fallback: str = CLASE_NEUTRA_FALLBACK,
        umbral_peso_activa: float = 0.05,
    ) -> None:
        self.config_mfs = config_mfs if config_mfs is not None else cargar_configuracion_mfs_json()
        self.reglas = reglas if reglas is not None else cargar_reglas_integradas_json()
        self.universo_calidad = universo_calidad
        self.params_salida = params_salida
        self.umbral_baja_media = umbral_baja_media
        self.umbral_media_alta = umbral_media_alta
        self.centroide_fallback = centroide_fallback
        self.clase_fallback = clase_fallback
        self.umbral_peso_activa = umbral_peso_activa

        p_baja = self.params_salida["Baja"]
        p_med = self.params_salida["Media"]
        p_alta = self.params_salida["Alta"]

        self.mf_salida_baja = calcular_pertenencia_trapezoidal(
            self.universo_calidad, p_baja[0], p_baja[1], p_baja[2], p_baja[3]
        )
        self.mf_salida_media = calcular_pertenencia_triangular(
            self.universo_calidad, p_med[0], p_med[1], p_med[2]
        )
        self.mf_salida_alta = calcular_pertenencia_trapezoidal(
            self.universo_calidad, p_alta[0], p_alta[1], p_alta[2], p_alta[3]
        )

        self.var_nombres = VARIABLES_FISICOQUIMICAS
        self.var_a_idx = {var: i for i, var in enumerate(self.var_nombres)}

    def _resolver_config_y_reglas(
        self,
        config_mfs: Optional[ConfiguracionMFs],
        reglas: Optional[List[ReglaUnificada]],
    ) -> Tuple[ConfiguracionMFs, List[ReglaUnificada]]:
        cfg = config_mfs if config_mfs is not None else self.config_mfs
        rgl = reglas if reglas is not None else self.reglas
        return cfg, rgl

    def fuzzificar_muestra(
        self,
        muestra_num: Union[Dict[str, float], pd.Series, np.ndarray, List[float]],
        config_mfs: Optional[ConfiguracionMFs] = None,
    ) -> Dict[str, Dict[str, float]]:
        cfg, _ = self._resolver_config_y_reglas(config_mfs, None)
        pertenencias: Dict[str, Dict[str, float]] = {}

        if isinstance(muestra_num, dict):
            for var in self.var_nombres:
                val = float(muestra_num[var])
                pertenencias[var] = cfg.calcular_pertenencias(var, val)
        elif isinstance(muestra_num, pd.Series):
            for var in self.var_nombres:
                val = float(muestra_num[var])
                pertenencias[var] = cfg.calcular_pertenencias(var, val)
        else:
            arr = np.asarray(muestra_num, dtype=float).ravel()
            for idx, var in enumerate(self.var_nombres):
                val = float(arr[idx])
                pertenencias[var] = cfg.calcular_pertenencias(var, val)

        return pertenencias

    def evaluar_activaciones_reglas(
        self,
        pertenencias_muestra: Dict[str, Dict[str, float]],
        reglas: Optional[List[ReglaUnificada]] = None,
    ) -> Dict[int, float]:
        _, rgl = self._resolver_config_y_reglas(None, reglas)
        return evaluar_activaciones_reglas(pertenencias_muestra, rgl, UMBRAL_ACTIVACION_MINIMA)

    def agregar_salida_difusa(
        self,
        activaciones_reglas: Dict[int, float],
        reglas: Optional[List[ReglaUnificada]] = None,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        _, rgl = self._resolver_config_y_reglas(None, reglas)
        return agregar_salida_difusa(
            activaciones_reglas=activaciones_reglas,
            reglas=rgl,
            mf_salida_baja=self.mf_salida_baja,
            mf_salida_media=self.mf_salida_media,
            mf_salida_alta=self.mf_salida_alta,
        )

    def defuzzificar_centroide(self, mu_agregado: np.ndarray) -> float:
        return defuzzificar_centroide(self.universo_calidad, mu_agregado, self.centroide_fallback)

    def clasificar_centroide(self, valor_centroide: float) -> str:
        return clasificar_centroide(valor_centroide, self.umbral_baja_media, self.umbral_media_alta)

    def predecir_muestra(
        self,
        muestra_num: Union[Dict[str, float], pd.Series, np.ndarray, List[float]],
        config_mfs: Optional[ConfiguracionMFs] = None,
        reglas: Optional[List[ReglaUnificada]] = None,
    ) -> Tuple[str, float, Dict[int, float]]:
        cfg, rgl = self._resolver_config_y_reglas(config_mfs, reglas)
        # [FASE 1: FUZZIFICACIÓN]

        pertenencias = self.fuzzificar_muestra(muestra_num, cfg)

        # [FASE 2: INFERENCIA]
        activaciones = self.evaluar_activaciones_reglas(pertenencias, rgl)

        if not activaciones:
            return self.clase_fallback, self.centroide_fallback, {}

        # [FASE 3: AGREGACIÓN]
        mu_agregado, _ = self.agregar_salida_difusa(activaciones, rgl)

        # [FASE 4: DEFUZZIFICACIÓN]
        centroide = self.defuzzificar_centroide(mu_agregado)
        clase_predicha = self.clasificar_centroide(centroide)

        return clase_predicha, centroide, activaciones

    def predecir_vectorizado(
        self,
        X_num: Union[pd.DataFrame, np.ndarray],
        config_mfs: Optional[ConfiguracionMFs] = None,
        reglas: Optional[List[ReglaUnificada]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        cfg, rgl = self._resolver_config_y_reglas(config_mfs, reglas)

        if isinstance(X_num, pd.DataFrame):
            X_mat = X_num[self.var_nombres].to_numpy(dtype=float)
        else:
            X_mat = np.asarray(X_num, dtype=float)

        n_muestras = X_mat.shape[0]

        pertenencias: Dict[Tuple[str, str], np.ndarray] = {}
        for var in self.var_nombres:
            col_idx = self.var_a_idx[var]
            vals = X_mat[:, col_idx]
            min_lim, max_lim = RANGOS_VARIABLES_DEFAULT.get(var, (0.0, 100.0))
            a, b, c = cfg.parametros[var]

            pertenencias[(var, "Bajo")] = calcular_pertenencia_trapezoidal(
                vals, min_lim - 1.0, min_lim - 1.0, a, b
            )
            pertenencias[(var, "Medio")] = calcular_pertenencia_triangular(vals, a, b, c)
            pertenencias[(var, "Alto")] = calcular_pertenencia_trapezoidal(
                vals, b, c, max_lim + 1.0, max_lim + 1.0
            )

        n_reglas = len(rgl)
        matriz_activaciones = np.zeros((n_muestras, n_reglas), dtype=float)
        act_baja = np.zeros(n_muestras, dtype=float)
        act_media = np.zeros(n_muestras, dtype=float)
        act_alta = np.zeros(n_muestras, dtype=float)

        for r_idx, regla in enumerate(rgl):
            if not regla.activa or regla.peso <= 0.0:
                continue

            fuerza = np.ones(n_muestras, dtype=float)
            for var_req, val_req in regla.antecedentes.items():
                if (var_req, val_req) in pertenencias:
                    fuerza = np.minimum(fuerza, pertenencias[(var_req, val_req)])
                else:
                    fuerza = np.zeros(n_muestras, dtype=float)

            fuerza_ponderada = fuerza * regla.peso
            matriz_activaciones[:, r_idx] = fuerza_ponderada

            if regla.consecuente == "Baja":
                act_baja = np.maximum(act_baja, fuerza_ponderada)
            elif regla.consecuente == "Media":
                act_media = np.maximum(act_media, fuerza_ponderada)
            elif regla.consecuente == "Alta":
                act_alta = np.maximum(act_alta, fuerza_ponderada)

        corte_baja = np.minimum(act_baja[:, None], self.mf_salida_baja[None, :])
        corte_media = np.minimum(act_media[:, None], self.mf_salida_media[None, :])
        corte_alta = np.minimum(act_alta[:, None], self.mf_salida_alta[None, :])

        salida_agregada = np.maximum(corte_baja, np.maximum(corte_media, corte_alta))
        area_total = np.sum(salida_agregada, axis=1)
        momento = np.sum(salida_agregada * self.universo_calidad[None, :], axis=1)

        centroides = np.where(
            area_total > 1e-6,
            momento / np.maximum(area_total, 1e-6),
            self.centroide_fallback,
        )

        y_pred = np.full(n_muestras, self.clase_fallback, dtype=object)
        y_pred[centroides < self.umbral_baja_media] = "Baja"
        y_pred[centroides >= self.umbral_media_alta] = "Alta"

        return y_pred, centroides, matriz_activaciones

    def predecir(
        self,
        X_num: Union[pd.DataFrame, np.ndarray],
        config_mfs: Optional[ConfiguracionMFs] = None,
        reglas: Optional[List[ReglaUnificada]] = None,
    ) -> np.ndarray:
        y_pred, _, _ = self.predecir_vectorizado(X_num, config_mfs, reglas)
        return y_pred

    def predecir_con_detalles(
        self,
        X_num: Union[pd.DataFrame, np.ndarray],
        config_mfs: Optional[ConfiguracionMFs] = None,
        reglas: Optional[List[ReglaUnificada]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict[int, float]]]:
        cfg, rgl = self._resolver_config_y_reglas(config_mfs, reglas)
        y_pred, centroides, matriz_act = self.predecir_vectorizado(X_num, cfg, rgl)

        lista_activadas: List[Dict[int, float]] = []
        for row_idx in range(len(y_pred)):
            fila_acts = {}
            for r_idx, r in enumerate(rgl):
                val = float(matriz_act[row_idx, r_idx])
                if val >= UMBRAL_ACTIVACION_MINIMA:
                    fila_acts[r.id_regla] = val
            lista_activadas.append(fila_acts)

        return y_pred, centroides, lista_activadas

    def evaluar(
        self,
        X_num: Union[pd.DataFrame, np.ndarray],
        y_real: Union[pd.Series, np.ndarray, List[str]],
        config_mfs: Optional[ConfiguracionMFs] = None,
        reglas: Optional[List[ReglaUnificada]] = None,
    ) -> Dict[str, Any]:
        cfg, rgl = self._resolver_config_y_reglas(config_mfs, reglas)
        y_true = np.asarray(y_real)
        y_pred, centroides, matriz_act = self.predecir_vectorizado(X_num, cfg, rgl)

        exactitud = float(np.mean(y_pred == y_true))
        f1_macro_val = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        f1_weighted_val = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

        muestras_activas = int(np.sum(np.any(matriz_act >= UMBRAL_ACTIVACION_MINIMA, axis=1)))
        cobertura = float(muestras_activas / len(y_true))
        reglas_activas_count = sum(1 for r in rgl if r.activa and r.peso > 0.0)

        reporte_dict = classification_report(
            y_true,
            y_pred,
            labels=ORDEN_CLASES_EVALUACION,
            target_names=ORDEN_CLASES_EVALUACION,
            output_dict=True,
            zero_division=0,
        )
        matriz = confusion_matrix(y_true, y_pred, labels=ORDEN_CLASES_EVALUACION)

        return {
            "exactitud": exactitud,
            "f1_macro": f1_macro_val,
            "f1_weighted": f1_weighted_val,
            "f1_ponderado": f1_weighted_val,
            "cobertura": cobertura,
            "muestras_cubiertas": muestras_activas,
            "total_muestras": len(y_true),
            "reglas_activas": reglas_activas_count,
            "total_reglas": len(rgl),
            "matriz_confusion": matriz,
            "reporte": reporte_dict,
            "reporte_clasificacion": reporte_dict,
            "predicciones": y_pred,
            "centroides": centroides,
        }


ClasificadorFuzzyMamdani = SistemaDifusoMamdani


def ejecutar_smoke_test_fase_9() -> bool:
    print("\n" + "=" * 85)
    print(" [FASE 9] MOTOR DE INFERENCIA DIFUSA MAMDANI Y EVALUACION PRE-AG (SMOKE TEST)")
    print("=" * 85)

    df_vinos = cargar_dataset()
    datos_particionados = preparar_conjuntos_entrenamiento_prueba(df_vinos)

    X_train_num = datos_particionados["X_train_num"]
    y_train = datos_particionados["y_train"]
    X_test_num = datos_particionados["X_test_num"]
    y_test = datos_particionados["y_test"]

    config_mfs = ConfiguracionMFs.desde_dataframe(X_train_num)
    reglas = cargar_reglas_integradas_json(RUTA_REGLAS_INTEGRADAS_JSON)

    sistema_mamdani = SistemaDifusoMamdani(config_mfs=config_mfs, reglas=reglas)

    metricas_train = sistema_mamdani.evaluar(X_train_num, y_train)
    metricas_test = sistema_mamdani.evaluar(X_test_num, y_test)

    print(f" * Exactitud en Train : {metricas_train['exactitud'] * 100:.2f}%")
    print(f" * Exactitud en Test  : {metricas_test['exactitud'] * 100:.2f}%")
    print(f" * F1-Macro en Train   : {metricas_train['f1_macro']:.4f}")
    print(f" * F1-Macro en Test    : {metricas_test['f1_macro']:.4f}")
    print(f" * Cobertura en Test   : {metricas_test['cobertura'] * 100:.2f}%")

    guardar_configuracion_mfs_json(config_mfs)

    es_coherente = (
        metricas_test["cobertura"] >= 0.90
        and metricas_test["f1_macro"] >= 0.40
        and metricas_test["exactitud"] >= 0.50
    )
    return es_coherente


if __name__ == "__main__":
    ejecutar_smoke_test_fase_9()
