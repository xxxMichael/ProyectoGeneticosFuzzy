"""
Módulo de Descubrimiento de Reglas Modulares de Clasificación — Algoritmo PRISM
Proyecto: Clasificación de Calidad de Vinos (PRISM, Apriori, Lógica Difusa y Algoritmo Genético)
"""

import os
import sys
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from src.datos.carga.cargador import cargar_dataset
from src.datos.preparacion.discretizacion import preparar_conjuntos_entrenamiento_prueba
from src.prism.induccion.selector import buscar_mejor_termino_voraz
from src.prism.reglas.regla import ReglaPRISM

# =============================================================================
# CONFIGURACIÓN Y PARÁMETROS DEL ALGORITMO PRISM
# =============================================================================
MIN_PRECISION_REGLA: float = 0.60       # Umbral mínimo de pureza/precisión para inducir una regla (60%)
MIN_INSTANCIAS_CUBIERTAS: int = 5       # Cantidad mínima de ejemplos positivos cubiertos por regla
MAX_CONDICIONES_POR_REGLA: int = 4      # Longitud máxima de términos en el antecedente (interpretabilidad)
CLASES_OBJETIVO: List[str] = ["Alta", "Baja", "Media"]  # Orden de clases para extracción iterativa
CLASE_POR_DEFECTO: str = "Media"        # Clase asignada en inferencia si ninguna regla aplica

DIRECTORIO_BASE: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUTA_SALIDA_REGLAS: str = os.path.join(DIRECTORIO_BASE, "results", "reglas_prism.csv")
# =============================================================================


class AlgoritmoPRISM:
    """
    Implementación desde cero del algoritmo modular PRISM (Cendrowska, 1987)
    para el descubrimiento de reglas de clasificación sobre datos discretizados.
    """

    def __init__(
        self,
        min_precision: float = MIN_PRECISION_REGLA,
        min_instancias: int = MIN_INSTANCIAS_CUBIERTAS,
        max_condiciones: int = MAX_CONDICIONES_POR_REGLA,
        clases_objetivo: Optional[List[str]] = None,
        clase_por_defecto: str = CLASE_POR_DEFECTO,
    ) -> None:
        self.min_precision: float = min_precision
        self.min_instancias: int = min_instancias
        self.max_condiciones: int = max_condiciones
        self.clases_objetivo: List[str] = clases_objetivo or list(CLASES_OBJETIVO)
        self.clase_por_defecto: str = clase_por_defecto
        self.reglas_: List[ReglaPRISM] = []
        self.variables_entrenamiento_: List[str] = []

    def fit(self, X_discreto: pd.DataFrame, y: pd.Series) -> "AlgoritmoPRISM":
        """
        Aprende reglas modulares desde cero aplicando el proceso iterativo de PRISM.
        """
        self.variables_entrenamiento_ = list(X_discreto.columns)
        self.reglas_ = []

        total_instancias_dataset = len(X_discreto)
        contador_reglas = 0

        df_entrenamiento = X_discreto.copy()
        df_entrenamiento["__target__"] = y.values

        for clase_actual in self.clases_objetivo:
            conjunto_e = df_entrenamiento.copy()

            while True:
                instancias_positivas_e = conjunto_e[conjunto_e["__target__"] == clase_actual]
                if len(instancias_positivas_e) < self.min_instancias:
                    break

                conjunto_s = conjunto_e.copy()
                antecedentes_regla: Dict[str, str] = {}
                variables_disponibles = list(self.variables_entrenamiento_)

                while len(variables_disponibles) > 0 and len(antecedentes_regla) < self.max_condiciones:
                    mejor_cond, mejor_prec, mejor_sop, _ = buscar_mejor_termino_voraz(
                        conjunto_s=conjunto_s,
                        variables_disponibles=variables_disponibles,
                        clase_actual=clase_actual,
                    )

                    if mejor_cond is None or mejor_sop == 0:
                        break

                    var_elegida, val_elegido = mejor_cond
                    antecedentes_regla[var_elegida] = val_elegido
                    variables_disponibles.remove(var_elegida)

                    conjunto_s = conjunto_s[conjunto_s[var_elegida] == val_elegido]

                    if mejor_prec >= self.min_precision:
                        break

                if not antecedentes_regla:
                    break

                mascara_global = pd.Series(True, index=df_entrenamiento.index)
                for var, val in antecedentes_regla.items():
                    mascara_global &= df_entrenamiento[var].astype(str) == str(val)

                instancias_cubiertas = df_entrenamiento[mascara_global]
                total_cubiertas = len(instancias_cubiertas)
                total_correctas = len(
                    instancias_cubiertas[instancias_cubiertas["__target__"] == clase_actual]
                )

                if total_correctas < self.min_instancias or total_cubiertas == 0:
                    break

                precision_global = total_correctas / total_cubiertas
                cobertura_relativa = total_cubiertas / total_instancias_dataset

                partes_texto = [f"{k} = {v}" for k, v in antecedentes_regla.items()]
                texto_antecedentes = " Y ".join(partes_texto)
                regla_texto = f"SI {texto_antecedentes} ENTONCES calidad = {clase_actual}"

                contador_reglas += 1
                nueva_regla = ReglaPRISM(
                    id_regla=contador_reglas,
                    antecedentes=antecedentes_regla,
                    consecuente=clase_actual,
                    precision=round(precision_global, 4),
                    cobertura_relativa=round(cobertura_relativa, 4),
                    cobertura_absoluta=total_cubiertas,
                    instancias_correctas=total_correctas,
                    num_condiciones=len(antecedentes_regla),
                    regla_texto=regla_texto,
                )
                self.reglas_.append(nueva_regla)

                mascara_e = pd.Series(True, index=conjunto_e.index)
                for var, val in antecedentes_regla.items():
                    mascara_e &= conjunto_e[var].astype(str) == str(val)

                indices_remover = conjunto_e[
                    mascara_e & (conjunto_e["__target__"] == clase_actual)
                ].index

                if len(indices_remover) == 0:
                    break

                conjunto_e = conjunto_e.drop(index=indices_remover)

        return self

    def predict_individual(self, fila: pd.Series) -> str:
        """
        Clasifica una sola instancia evaluando las reglas en orden de mayor precisión.
        """
        reglas_priorizadas = sorted(
            self.reglas_,
            key=lambda r: (r.precision, r.cobertura_absoluta),
            reverse=True,
        )
        for regla in reglas_priorizadas:
            if regla.cumple_antecedentes(fila):
                return regla.consecuente

        return self.clase_por_defecto

    def predict(self, X_discreto: pd.DataFrame) -> pd.Series:
        """
        Genera predicciones para todas las filas de un DataFrame discretizado.
        """
        predicciones = [self.predict_individual(fila) for _, fila in X_discreto.iterrows()]
        return pd.Series(predicciones, index=X_discreto.index, name="prediccion_prism")

    def obtener_dataframe_reglas(self) -> pd.DataFrame:
        """
        Retorna las reglas inducidas en formato tabular para persistencia y análisis.
        """
        filas = []
        for r in self.reglas_:
            filas.append(
                {
                    "id_regla": r.id_regla,
                    "antecedentes": " AND ".join(
                        [f"{k} = {v}" for k, v in r.antecedentes.items()]
                    ),
                    "consecuente": r.consecuente,
                    "precision": r.precision,
                    "cobertura_relativa": r.cobertura_relativa,
                    "cobertura_absoluta": r.cobertura_absoluta,
                    "instancias_correctas": r.instancias_correctas,
                    "num_condiciones": r.num_condiciones,
                    "regla_texto": r.regla_texto,
                }
            )
        return pd.DataFrame(filas)

    def exportar_reglas_csv(self, ruta_archivo: str = RUTA_SALIDA_REGLAS) -> str:
        """
        Exporta el conjunto de reglas a un archivo CSV.
        """
        directorio = os.path.dirname(ruta_archivo)
        if directorio and not os.path.exists(directorio):
            os.makedirs(directorio, exist_ok=True)

        df_reglas = self.obtener_dataframe_reglas()
        df_reglas.to_csv(ruta_archivo, index=False, encoding="utf-8")
        return ruta_archivo

    def mostrar_resumen_reglas(self) -> None:
        """
        Imprime un reporte detallado en consola de las reglas inducidas agrupadas por clase.
        """
        df_reglas = self.obtener_dataframe_reglas()
        print("\n" + "=" * 95)
        print(" [FASE 5] REGLAS MODULARES INDUCIDAS POR EL ALGORITMO PRISM")
        print("=" * 95)

        if df_reglas.empty:
            print(" [!] No se indujeron reglas con la configuracion actual.")
            return

        for clase in self.clases_objetivo:
            reglas_clase = df_reglas[df_reglas["consecuente"] == clase]
            print(f"\n >>> REGLAS ASOCIADAS A CALIDAD '{clase.upper()}' (Total: {len(reglas_clase)})")
            print("-" * 95)
            for _, fila in reglas_clase.iterrows():
                print(
                    f" Regla #{fila['id_regla']:02d} | Precision: {fila['precision'] * 100:>5.1f}% | "
                    f"Cobertura: {fila['cobertura_absoluta']:>3d} reg ({fila['cobertura_relativa'] * 100:>4.1f}%) | "
                    f"Condiciones: {fila['num_condiciones']}"
                )
                print(f"   -> {fila['regla_texto']}")
            print("-" * 95)

        print("\n" + "=" * 95)
        print(f" RESUMEN GLOBAL: {len(df_reglas)} reglas inducidas exitosamente.")
        print(
            f" Precision promedio global : {df_reglas['precision'].mean() * 100:.2f}% | "
            f" Longitud promedio antecedente : {df_reglas['num_condiciones'].mean():.2f} terminos"
        )
        print("=" * 95 + "\n")


def ejecutar_induccion_prism() -> Tuple[AlgoritmoPRISM, Dict[str, Any]]:
    """
    Función orquestadora principal para la ejecución, validación y exportación de PRISM.
    """
    print("=" * 95)
    print(" INICIO DEL PIPELINE DE INDUCCION MODULAR PRISM (FASE 5)")
    print("=" * 95)

    print(" [1/4] Cargando datos y generando particiones estratificadas...")
    df_vinos = cargar_dataset()
    datos = preparar_conjuntos_entrenamiento_prueba(df_vinos)

    X_train_disc = datos["X_train_disc"]
    y_train = datos["y_train"]
    X_test_disc = datos["X_test_disc"]
    y_test = datos["y_test"]

    print(f"   - Muestras Train : {len(X_train_disc)} registros")
    print(f"   - Muestras Test  : {len(X_test_disc)} registros")

    print("\n [2/4] Ejecutando algoritmo PRISM desde cero...")
    modelo_prism = AlgoritmoPRISM(
        min_precision=MIN_PRECISION_REGLA,
        min_instancias=MIN_INSTANCIAS_CUBIERTAS,
        max_condiciones=MAX_CONDICIONES_POR_REGLA,
        clases_objetivo=CLASES_OBJETIVO,
        clase_por_defecto=CLASE_POR_DEFECTO,
    )
    modelo_prism.fit(X_train_disc, y_train)
    modelo_prism.mostrar_resumen_reglas()

    print(" [3/4] Evaluando clasificador en conjuntos de Entrenamiento y Prueba...")
    y_pred_train = modelo_prism.predict(X_train_disc)
    y_pred_test = modelo_prism.predict(X_test_disc)

    acc_train = float((y_pred_train == y_train).mean())
    acc_test = float((y_pred_test == y_test).mean())

    print(f"   - Exactitud Global (Train) : {acc_train * 100:.2f}%")
    print(f"   - Exactitud Global (Test)  : {acc_test * 100:.2f}%\n")

    print(" Matriz de Confusion en Test (Filas: Real, Columnas: Prediccion):")
    matriz_conf = confusion_matrix(y_test, y_pred_test, labels=["Baja", "Media", "Alta"])
    print(f"   {'':<10} | {'Pred Baja':<10} | {'Pred Media':<10} | {'Pred Alta':<10}")
    print("   " + "-" * 42)
    for i, etiqueta_real in enumerate(["Real Baja", "Real Media", "Real Alta"]):
        print(
            f"   {etiqueta_real:<10} | {matriz_conf[i, 0]:<10} | {matriz_conf[i, 1]:<10} | {matriz_conf[i, 2]:<10}"
        )

    print(f"\n [4/4] Guardando reglas inducidas en CSV: {RUTA_SALIDA_REGLAS}")
    ruta_generada = modelo_prism.exportar_reglas_csv(RUTA_SALIDA_REGLAS)
    print(f" [+] Archivo guardado exitosamente: {ruta_generada}\n")

    metricas = {
        "acc_train": acc_train,
        "acc_test": acc_test,
        "total_reglas": len(modelo_prism.reglas_),
        "ruta_csv": ruta_generada,
    }
    return modelo_prism, metricas


if __name__ == "__main__":
    ejecutar_induccion_prism()
