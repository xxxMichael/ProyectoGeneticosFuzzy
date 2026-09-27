"""
Submódulo de Gráfico de Convergencia del Algoritmo Genético
Proyecto: Clasificación de Calidad de Vinos (PRISM, Apriori, Lógica Difusa y Algoritmo Genético)
"""

import os
import matplotlib.pyplot as plt
import pandas as pd

DIRECTORIO_BASE: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RUTA_PLOTS: str = os.path.join(DIRECTORIO_BASE, "results", "plots")
RUTA_HISTORIAL_CONVERGENCIA_CSV: str = os.path.join(DIRECTORIO_BASE, "results", "historial_convergencia_ga.csv")


def generar_grafico_convergencia_ga(
    ruta_csv: str = RUTA_HISTORIAL_CONVERGENCIA_CSV,
    ruta_salida: str = os.path.join(RUTA_PLOTS, "convergencia_fitness_ga.png"),
) -> str:
    """
    Genera la curva de convergencia del Algoritmo Genético a lo largo de las 30 generaciones:
    - Subplot superior: Mejor Fitness vs Fitness Promedio de la población.
    - Subplot inferior: F1-Macro, Cobertura y Reglas Activas.
    """
    os.makedirs(os.path.dirname(os.path.abspath(ruta_salida)), exist_ok=True)
    if not os.path.exists(ruta_csv):
        raise FileNotFoundError(f"No se encontró el archivo de convergencia: {ruta_csv}")

    df_hist = pd.read_csv(ruta_csv)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9), dpi=300, sharex=True)
    fig.suptitle(
        "Convergencia Evolutiva del Algoritmo Genético (DEAP / NumPy)\nOptimización Multi-criterio de Funciones Difusas y Base de Reglas",
        fontsize=14,
        fontweight="bold",
        y=0.97,
    )

    generaciones = df_hist["generacion"]

    # Subplot 1: Fitness
    ax1.plot(
        generaciones,
        df_hist["mejor_fitness"],
        color="#1f77b4",
        linewidth=2.5,
        marker="o",
        markersize=5,
        label="Mejor Fitness ($f_{max}$)",
    )
    ax1.plot(
        generaciones,
        df_hist["fitness_promedio"],
        color="#ff7f0e",
        linewidth=2.0,
        linestyle="--",
        marker="s",
        markersize=4,
        label="Fitness Promedio de Población ($\\bar{f}$)",
    )

    fit_ini = df_hist["mejor_fitness"].iloc[0]
    fit_fin = df_hist["mejor_fitness"].iloc[-1]
    gen_fin = generaciones.iloc[-1]
    ax1.annotate(
        f"Gen 0: {fit_ini:.4f}",
        xy=(0, fit_ini),
        xytext=(1.5, fit_ini - 0.025),
        arrowprops=dict(facecolor="#1f77b4", shrink=0.08, width=1.5, headwidth=6),
        fontweight="bold",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="#e8f4f8", ec="#1f77b4", lw=1),
    )
    ax1.annotate(
        f"Gen {gen_fin}: {fit_fin:.4f}\n(+{(fit_fin - fit_ini):.4f})",
        xy=(gen_fin, fit_fin),
        xytext=(gen_fin - 6.5, fit_fin - 0.035),
        arrowprops=dict(facecolor="#1f77b4", shrink=0.08, width=1.5, headwidth=6),
        fontweight="bold",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="#e8f4f8", ec="#1f77b4", lw=1),
    )

    ax1.set_ylabel("Fitness Multi-criterio", fontsize=11, fontweight="bold")
    ax1.set_title("Evolución de Aptitud (Fitness = 0.70·F1 + 0.20·Cob - 0.10·Reglas/26)", fontsize=11)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="lower right", framealpha=0.95, fontsize=10)
    ax1.set_ylim(min(df_hist["fitness_promedio"].min(), fit_ini) - 0.03, fit_fin + 0.03)

    # Subplot 2: F1-Macro, Cobertura y Reglas Activas
    color_f1 = "#2ca02c"
    color_cob = "#9467bd"
    color_reglas = "#d62728"

    ax2.plot(
        generaciones,
        df_hist["f1_macro"],
        color=color_f1,
        linewidth=2.2,
        marker="^",
        markersize=5,
        label="F1-Macro (Train)",
    )
    ax2.plot(
        generaciones,
        df_hist["cobertura"],
        color=color_cob,
        linewidth=2.0,
        linestyle="-.",
        marker="d",
        markersize=4,
        label="Cobertura de Reglas (Train)",
    )

    ax2.set_xlabel("Generación Evolutiva", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Métricas de Rendimiento [0.0 - 1.0]", fontsize=11, fontweight="bold")
    ax2.set_title("Evolución de Componentes: F1-Macro, Cobertura y Parquedad de Reglas", fontsize=11)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.set_ylim(0.50, 1.03)

    ax2_twin = ax2.twinx()
    ax2_twin.plot(
        generaciones,
        df_hist["reglas_activas"],
        color=color_reglas,
        linewidth=2.0,
        linestyle=":",
        marker="x",
        markersize=6,
        label="Reglas Activas ($N_{act}$)",
    )
    ax2_twin.set_ylabel("Número de Reglas Activas (de 26)", color=color_reglas, fontsize=11, fontweight="bold")
    ax2_twin.tick_params(axis="y", labelcolor=color_reglas)
    ax2_twin.set_ylim(10, 28)

    lines_1, labels_1 = ax2.get_legend_handles_labels()
    lines_2, labels_2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc="center right", framealpha=0.95, fontsize=10)

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    plt.savefig(ruta_salida, dpi=300, bbox_inches="tight")
    plt.close()
    print(f" [+] Gráfico de convergencia guardado: {ruta_salida}")
    return ruta_salida
