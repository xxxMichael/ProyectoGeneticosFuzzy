"""
Módulo del Algoritmo Apriori desde Cero para Minería de Reglas de Asociación
Proyecto: Clasificación de Calidad de Vinos (PRISM, Apriori, Lógica Difusa y Algoritmo Genético)
"""

import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd

from src.datos.carga.cargador import cargar_dataset
from src.datos.preparacion.discretizacion import (
    NOMBRE_CALIDAD_CATEGORICA,
    preparar_conjuntos_entrenamiento_prueba,
)
from src.apriori.itemsets.generador import (
    construir_transacciones,
    encontrar_itemsets_frecuentes,
)
from src.apriori.reglas.asociacion import (
    ReglaAsociacion,
    extraer_reglas_asociacion_filtradas,
)

# =============================================================================
# CONFIGURACIÓN Y PARÁMETROS DEL ALGORITMO APRIORI
# =============================================================================
MIN_SOPORTE: float = 0.02           # Frecuencia mínima relativa en el dataset (2%)
MIN_CONFIANZA: float = 0.60         # Probabilidad condicional mínima P(Calidad | Antecedente) (60%)
MIN_LIFT: float = 1.20              # Umbral de correlación positiva estricta (> 1.20)
MAX_LONGITUD_ITEMSET: int = 4       # Tamaño máximo de itemset (1 consecuente + hasta 3 antecedentes)
NOMBRE_VARIABLE_OBJETIVO: str = NOMBRE_CALIDAD_CATEGORICA  # 'calidad_categoria'
CLASES_CALIDAD_VALIDAS: Tuple[str, ...] = ("Baja", "Media", "Alta")  # Categorías posibles

DIRECTORIO_PROYECTO: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DIRECTORIO_RESULTADOS: str = os.path.join(DIRECTORIO_PROYECTO, "results")
RUTA_SALIDA_APRIORI: str = os.path.join(DIRECTORIO_RESULTADOS, "reglas_apriori.csv")
# =============================================================================


class Apriori:
    """
    Implementación desde cero del algoritmo Apriori para descubrimiento de
    itemsets frecuentes y generación de reglas de asociación con consecuente objetivo.
    """

    def __init__(
        self,
        min_soporte: float = MIN_SOPORTE,
        min_confianza: float = MIN_CONFIANZA,
        min_lift: float = MIN_LIFT,
        max_longitud: int = MAX_LONGITUD_ITEMSET,
        nombre_objetivo: str = NOMBRE_VARIABLE_OBJETIVO,
        clases_validas: Tuple[str, ...] = CLASES_CALIDAD_VALIDAS,
    ) -> None:
        self.min_soporte: float = min_soporte
        self.min_confianza: float = min_confianza
        self.min_lift: float = min_lift
        self.max_longitud: int = max_longitud
        self.nombre_objetivo: str = nombre_objetivo
        self.clases_validas: Tuple[str, ...] = clases_validas

        self.itemsets_frecuentes_: Dict[Tuple[str, ...], float] = {}
        self.reglas_: List[ReglaAsociacion] = []
        self.total_transacciones_: int = 0

    def construir_transacciones(
        self,
        dataframe_x_disc: pd.DataFrame,
        serie_y_cat: pd.Series,
    ) -> List[Set[str]]:
        return construir_transacciones(
            dataframe_x_disc=dataframe_x_disc,
            serie_y_cat=serie_y_cat,
            nombre_objetivo=self.nombre_objetivo,
        )

    def encontrar_itemsets_frecuentes(
        self,
        transacciones: List[Set[str]],
    ) -> Dict[Tuple[str, ...], float]:
        self.total_transacciones_ = len(transacciones)
        self.itemsets_frecuentes_ = encontrar_itemsets_frecuentes(
            transacciones=transacciones,
            min_soporte=self.min_soporte,
            max_longitud=self.max_longitud,
        )
        return self.itemsets_frecuentes_

    def generar_reglas_asociacion(
        self,
        itemsets_frecuentes: Optional[Dict[Tuple[str, ...], float]] = None,
    ) -> List[ReglaAsociacion]:
        if itemsets_frecuentes is None:
            itemsets_frecuentes = self.itemsets_frecuentes_

        self.reglas_ = extraer_reglas_asociacion_filtradas(
            itemsets_frecuentes=itemsets_frecuentes,
            nombre_objetivo=self.nombre_objetivo,
            clases_validas=self.clases_validas,
            min_confianza=self.min_confianza,
            min_lift=self.min_lift,
        )
        return self.reglas_

    def ajustar(
        self,
        dataframe_x_disc: pd.DataFrame,
        serie_y_cat: pd.Series,
    ) -> List[ReglaAsociacion]:
        transacciones = self.construir_transacciones(dataframe_x_disc, serie_y_cat)
        self.encontrar_itemsets_frecuentes(transacciones)
        return self.generar_reglas_asociacion()

    def reglas_a_dataframe(
        self,
        reglas: Optional[List[ReglaAsociacion]] = None,
    ) -> pd.DataFrame:
        lista_reglas = self.reglas_ if reglas is None else reglas
        filas = [
            regla.a_diccionario(id_regla=i + 1)
            for i, regla in enumerate(lista_reglas)
        ]
        columnas = [
            "id_regla",
            "antecedentes",
            "consecuente",
            "soporte",
            "confianza",
            "lift",
            "num_condiciones",
            "regla_texto",
        ]
        return pd.DataFrame(filas, columns=columnas)

    def guardar_reglas_csv(
        self,
        ruta_archivo: str = RUTA_SALIDA_APRIORI,
        dataframe_reglas: Optional[pd.DataFrame] = None,
    ) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(ruta_archivo)), exist_ok=True)
        df_exportar = self.reglas_a_dataframe() if dataframe_reglas is None else dataframe_reglas
        df_exportar.to_csv(ruta_archivo, index=False, encoding="utf-8")
        return ruta_archivo

    def mostrar_resumen_reglas(self, top_n: int = 15) -> None:
        print("\n" + "=" * 98)
        print(" [FASE 6] RESUMEN DE MINERIA DE REGLAS DE ASOCIACION (ALGORITMO APRIORI)")
        print("=" * 98)
        print(f" * Transacciones analizadas     : {self.total_transacciones_}")
        print(f" * Itemsets frecuentes totales  : {len(self.itemsets_frecuentes_)}")
        print(f" * Reglas de calidad generadas  : {len(self.reglas_)}")
        print(f" * Hiperparametros aplicados   : Min Soporte={self.min_soporte:.2f} | Min Confianza={self.min_confianza:.2f} | Min Lift={self.min_lift:.2f}")
        print("-" * 98)

        conteo_por_clase: Dict[str, int] = defaultdict(int)
        for regla in self.reglas_:
            conteo_por_clase[regla.consecuente] += 1

        print(" Distribucion de reglas por clase objetivo:")
        for clase in self.clases_validas:
            tag = f"{self.nombre_objetivo}={clase}"
            print(f"   - Consecuente '{tag}': {conteo_por_clase.get(tag, 0)} reglas")
        print("-" * 98)

        print(f" Top {min(top_n, len(self.reglas_))} Reglas con mayor Lift:")
        print(f"{'ID':<4} | {'Regla (SI Antecedentes ENTONCES Consecuente)':<62} | {'Sop':<6} | {'Conf':<6} | {'Lift':<6}")
        print("-" * 98)

        for i, regla in enumerate(self.reglas_[:top_n], start=1):
            texto_regla = regla.a_texto()
            if len(texto_regla) > 60:
                texto_regla = texto_regla[:57] + "..."
            print(
                f"{i:<4} | {texto_regla:<62} | {regla.soporte:<6.4f} | {regla.confianza:<6.4f} | {regla.lift:<6.4f}"
            )
        print("=" * 98 + "\n")


def ejecutar_apriori_completo(
    min_soporte: float = MIN_SOPORTE,
    min_confianza: float = MIN_CONFIANZA,
    min_lift: float = MIN_LIFT,
    ruta_salida: str = RUTA_SALIDA_APRIORI,
) -> Tuple[Apriori, pd.DataFrame]:
    print("=" * 98)
    print(" INICIO DE MINERIA DE REGLAS DE ASOCIACION — APRIORI (FASE 6)")
    print("=" * 98)

    print(" [1/4] Preparando transacciones desde los datos discretizados...")
    df_raw = cargar_dataset()
    particiones = preparar_conjuntos_entrenamiento_prueba(df_raw)

    x_train_disc = particiones["X_train_disc"]
    y_train_cat = particiones["y_train"]

    print(f"   - Instancias de entrenamiento : {len(x_train_disc)}")
    print(f"   - Atributos por transaccion   : {x_train_disc.shape[1]} variables + 1 target")

    print("\n [2/4] Ejecutando algoritmo Apriori desde cero...")
    motor_apriori = Apriori(
        min_soporte=min_soporte,
        min_confianza=min_confianza,
        min_lift=min_lift,
        max_longitud=MAX_LONGITUD_ITEMSET,
    )
    reglas = motor_apriori.ajustar(x_train_disc, y_train_cat)

    print("\n [3/4] Resumen de asociaciones extraidas:")
    motor_apriori.mostrar_resumen_reglas(top_n=12)

    print(f" [4/4] Guardando reglas filtradas en: {ruta_salida}")
    df_reglas = motor_apriori.guardar_reglas_csv(ruta_salida)
    print(f" [+] Archivo guardado con exito ({len(reglas)} reglas exportadas).\n")

    return motor_apriori, motor_apriori.reglas_a_dataframe()


if __name__ == "__main__":
    ejecutar_apriori_completo()
