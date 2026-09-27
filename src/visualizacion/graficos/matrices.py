"""
Submódulo de Gráfico de Matrices de Confusión Comparativas
Proyecto: Clasificación de Calidad de Vinos (PRISM, Apriori, Lógica Difusa y Algoritmo Genético)
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np

from src.datos.carga.cargador import cargar_dataset
from src.datos.preparacion.discretizacion import preparar_conjuntos_entrenamiento_prueba
from src.fuzzy.fuzzy import (
    SistemaDifusoMamdani,
    cargar_configuracion_mfs_json,
)
from src.reglas.regla_unificada import ReglaUnificada
from src.reglas.clasificador_base.clasificador import ClasificadorReglasDiscretas

DIRECTORIO_BASE: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RUTA_PLOTS: str = os.path.join(DIRECTORIO_BASE, "results", "plots")
RUTA_REGLAS_BASE_JSON: str = os.path.join(DIRECTORIO_BASE, "results", "reglas_base_integrada.json")
RUTA_PARAMS_MFS_INICIALES_JSON: str = os.path.join(DIRECTORIO_BASE, "results", "params_mfs_iniciales.json")
RUTA_PARAMS_MFS_OPTIMIZADAS_JSON: str = os.path.join(DIRECTORIO_BASE, "results", "mejores_params.json")
RUTA_REGLAS_OPTIMIZADAS_JSON: str = os.path.join(DIRECTORIO_BASE, "results", "reglas_optimizadas_ga.json")


def generar_grafico_matrices_confusion_comparativas(
    ruta_salida: str = os.path.join(RUTA_PLOTS, "matrices_confusion_comparativas.png"),
) -> str:
    """
    Genera una visualización lado a lado de las Matrices de Confusión sobre el conjunto
    de Test para las 3 fases del proyecto:
    1. Fase 1: Baseline Discreto (PRISM + Apriori)
    2. Fase 2: Pre-GA Difuso Mamdani
    3. Fase 3: Post-GA Difuso Optimizado Mamdani
    """
    os.makedirs(os.path.dirname(os.path.abspath(ruta_salida)), exist_ok=True)

    df_vinos = cargar_dataset()
    datos = preparar_conjuntos_entrenamiento_prueba(df_vinos)
    X_test_num = datos["X_test_num"]
    X_test_disc = datos["X_test_disc"]
    y_test = datos["y_test"]

    # 1. Baseline Discreto
    reglas_base_raw = json.load(open(RUTA_REGLAS_BASE_JSON, "r", encoding="utf-8"))
    reglas_base = [ReglaUnificada(**r) for r in reglas_base_raw]
    clf_disc = ClasificadorReglasDiscretas(reglas_base)
    met_disc = clf_disc.evaluar(X_test_disc, y_test)

    # 2. Pre-GA Difuso
    cfg_pre = cargar_configuracion_mfs_json(RUTA_PARAMS_MFS_INICIALES_JSON)
    sis_pre = SistemaDifusoMamdani(config_mfs=cfg_pre, reglas=reglas_base)
    met_pre = sis_pre.evaluar(X_test_num, y_test)

    # 3. Post-GA Difuso
    ruta_opt = RUTA_PARAMS_MFS_OPTIMIZADAS_JSON
    if not os.path.exists(ruta_opt):
        ruta_alt = os.path.join(DIRECTORIO_BASE, "results", "params_mfs_optimizadas_ga.json")
        if os.path.exists(ruta_alt):
            ruta_opt = ruta_alt

    cfg_post = cargar_configuracion_mfs_json(ruta_opt)
    reglas_post_raw = json.load(open(RUTA_REGLAS_OPTIMIZADAS_JSON, "r", encoding="utf-8"))
    reglas_post = [ReglaUnificada(**r) for r in reglas_post_raw]
    sis_post = SistemaDifusoMamdani(config_mfs=cfg_post, reglas=reglas_post)
    met_post = sis_post.evaluar(X_test_num, y_test)

    clases = ["Alta", "Baja", "Media"]
    matrices = [
        ("Fase 1: Baseline Discreto (PRISM + Apriori)", met_disc, plt.cm.Blues),
        ("Fase 2: Sistema Difuso Pre-AG (Mamdani)", met_pre, plt.cm.Oranges),
        ("Fase 3: Sistema Difuso Post-AG (Optimizado)", met_post, plt.cm.Greens),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), dpi=300)
    fig.suptitle(
        "Evaluación Comparativa en Conjunto de Prueba (Test, N=320)\nMatrices de Confusión de las 3 Fases Progresivas",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    for idx, (titulo, met, cmap) in enumerate(matrices):
        ax = axes[idx]
        mat = np.array(met["matriz_confusion"])
        totales_reales = mat.sum(axis=1, keepdims=True)

        ax.imshow(mat, interpolation="nearest", cmap=cmap)

        thresh = mat.max() / 2.0
        for i in range(len(clases)):
            for j in range(len(clases)):
                val = mat[i, j]
                pct = (val / totales_reales[i, 0] * 100.0) if totales_reales[i, 0] > 0 else 0.0
                color_texto = "white" if val > thresh else "black"
                ax.text(
                    j,
                    i,
                    f"{val}\n({pct:.1f}%)",
                    ha="center",
                    va="center",
                    color=color_texto,
                    fontweight="bold",
                    fontsize=10,
                )

        ax.set_xticks(np.arange(len(clases)))
        ax.set_yticks(np.arange(len(clases)))
        ax.set_xticklabels(clases, fontsize=10, fontweight="bold")
        ax.set_yticklabels(clases, fontsize=10, fontweight="bold")

        ax.set_xlabel("Predicción del Modelo", fontsize=10, fontweight="bold")
        if idx == 0:
            ax.set_ylabel("Clase Real (Ground Truth)", fontsize=10, fontweight="bold")

        acc = met["exactitud"] * 100
        f1_m = met["f1_macro"]
        cob = met["cobertura"] * 100
        ax.set_title(
            f"{titulo}\nAcc: {acc:.2f}% | F1-Macro: {f1_m:.4f} | Cob: {cob:.1f}%",
            fontsize=10,
            fontweight="bold",
            pad=10,
        )

    plt.tight_layout(rect=[0, 0.03, 1, 0.94])
    plt.savefig(ruta_salida, dpi=300, bbox_inches="tight")
    plt.close()
    print(f" [+] Gráfico de matrices de confusión guardado: {ruta_salida}")
    return ruta_salida
