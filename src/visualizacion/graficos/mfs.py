"""
Submódulo de Gráfico Comparativo de Funciones de Pertenencia
Proyecto: Clasificación de Calidad de Vinos (PRISM, Apriori, Lógica Difusa y Algoritmo Genético)
"""

import os
import matplotlib.pyplot as plt
import numpy as np

from src.fuzzy.fuzzy import (
    RANGOS_VARIABLES_DEFAULT,
    calcular_pertenencia_trapezoidal,
    calcular_pertenencia_triangular,
    cargar_configuracion_mfs_json,
)

DIRECTORIO_BASE: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RUTA_PLOTS: str = os.path.join(DIRECTORIO_BASE, "results", "plots")
RUTA_PARAMS_MFS_INICIALES_JSON: str = os.path.join(DIRECTORIO_BASE, "results", "params_mfs_iniciales.json")
RUTA_PARAMS_MFS_OPTIMIZADAS_JSON: str = os.path.join(DIRECTORIO_BASE, "results", "mejores_params.json")


def generar_grafico_comparativa_mfs(
    ruta_params_iniciales: str = RUTA_PARAMS_MFS_INICIALES_JSON,
    ruta_params_optimizadas: str = RUTA_PARAMS_MFS_OPTIMIZADAS_JSON,
    ruta_salida: str = os.path.join(RUTA_PLOTS, "comparativa_mfs_pre_post_ga.png"),
) -> str:
    """
    Genera un gráfico comparativo de las Funciones de Pertenencia (Bajo, Medio, Alto)
    antes y después del Algoritmo Genético para las variables fisicoquímicas más críticas.
    """
    os.makedirs(os.path.dirname(os.path.abspath(ruta_salida)), exist_ok=True)
    cfg_ini = cargar_configuracion_mfs_json(ruta_params_iniciales)

    if not os.path.exists(ruta_params_optimizadas):
        ruta_alt = os.path.join(DIRECTORIO_BASE, "results", "params_mfs_optimizadas_ga.json")
        if os.path.exists(ruta_alt):
            ruta_params_optimizadas = ruta_alt

    cfg_opt = cargar_configuracion_mfs_json(ruta_params_optimizadas)

    variables_clave = [
        "alcohol",
        "volatile acidity",
        "sulphates",
        "citric acid",
        "total sulfur dioxide",
        "pH",
    ]

    fig, axes = plt.subplots(3, 2, figsize=(14, 12), dpi=300)
    fig.suptitle(
        "Comparativa de Funciones de Pertenencia Difusas: Pre-AG (Líneas Discontinuas) vs Post-AG Optimizado (Líneas Sólidas)",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    colores = {"Bajo": "#1f77b4", "Medio": "#ff7f0e", "Alto": "#2ca02c"}

    for idx, var in enumerate(variables_clave):
        ax = axes[idx // 2, idx % 2]
        r_min, r_max = RANGOS_VARIABLES_DEFAULT[var]
        x_vals = np.linspace(r_min, r_max, 500)

        p_ini = cfg_ini.parametros[var]
        p_opt = cfg_opt.parametros[var]

        mu_bajo_ini = calcular_pertenencia_trapezoidal(x_vals, r_min, r_min, p_ini[0], p_ini[1])
        mu_med_ini = calcular_pertenencia_triangular(x_vals, p_ini[0], p_ini[1], p_ini[2])
        mu_alt_ini = calcular_pertenencia_trapezoidal(x_vals, p_ini[1], p_ini[2], r_max, r_max)

        mu_bajo_opt = calcular_pertenencia_trapezoidal(x_vals, r_min, r_min, p_opt[0], p_opt[1])
        mu_med_opt = calcular_pertenencia_triangular(x_vals, p_opt[0], p_opt[1], p_opt[2])
        mu_alt_opt = calcular_pertenencia_trapezoidal(x_vals, p_opt[1], p_opt[2], r_max, r_max)

        ax.plot(x_vals, mu_bajo_ini, color=colores["Bajo"], linestyle=":", linewidth=1.5, alpha=0.55, label="Bajo (Pre-AG)")
        ax.plot(x_vals, mu_med_ini, color=colores["Medio"], linestyle=":", linewidth=1.5, alpha=0.55, label="Medio (Pre-AG)")
        ax.plot(x_vals, mu_alt_ini, color=colores["Alto"], linestyle=":", linewidth=1.5, alpha=0.55, label="Alto (Pre-AG)")

        ax.plot(x_vals, mu_bajo_opt, color=colores["Bajo"], linestyle="-", linewidth=2.4, label="Bajo (Post-AG)")
        ax.plot(x_vals, mu_med_opt, color=colores["Medio"], linestyle="-", linewidth=2.4, label="Medio (Post-AG)")
        ax.plot(x_vals, mu_alt_opt, color=colores["Alto"], linestyle="-", linewidth=2.4, label="Alto (Post-AG)")

        ax.fill_between(x_vals, 0, mu_bajo_opt, color=colores["Bajo"], alpha=0.08)
        ax.fill_between(x_vals, 0, mu_med_opt, color=colores["Medio"], alpha=0.08)
        ax.fill_between(x_vals, 0, mu_alt_opt, color=colores["Alto"], alpha=0.08)

        ax.axvline(p_opt[0], color=colores["Bajo"], linestyle="--", linewidth=0.8, alpha=0.7)
        ax.axvline(p_opt[1], color=colores["Medio"], linestyle="--", linewidth=0.8, alpha=0.7)
        ax.axvline(p_opt[2], color=colores["Alto"], linestyle="--", linewidth=0.8, alpha=0.7)

        ax.set_title(
            f"Variable: {var.upper()}\n[Pre: ({p_ini[0]:.2f}, {p_ini[1]:.2f}, {p_ini[2]:.2f}) $\\rightarrow$ Post: ({p_opt[0]:.2f}, {p_opt[1]:.2f}, {p_opt[2]:.2f})]",
            fontsize=10,
            fontweight="bold",
        )
        ax.set_ylabel("Pertenencia $\\mu(x)$", fontsize=9)
        ax.set_xlabel(f"Valor fisicoquímico ({var})", fontsize=9)
        ax.set_ylim(-0.05, 1.08)
        ax.set_xlim(r_min, r_max)
        ax.grid(True, linestyle=":", alpha=0.5)

        if idx == 0:
            ax.legend(loc="upper right", ncol=2, fontsize=7.5, framealpha=0.9)

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    plt.savefig(ruta_salida, dpi=300, bbox_inches="tight")
    plt.close()
    print(f" [+] Gráfico comparativo de MFs guardado: {ruta_salida}")
    return ruta_salida
