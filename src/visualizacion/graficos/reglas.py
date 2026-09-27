"""
Submódulo de Gráfico de Evolución de Reglas
Proyecto: Clasificación de Calidad de Vinos (PRISM, Apriori, Lógica Difusa y Algoritmo Genético)
"""

import json
import os
import matplotlib.pyplot as plt

DIRECTORIO_BASE: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RUTA_PLOTS: str = os.path.join(DIRECTORIO_BASE, "results", "plots")
RUTA_REGLAS_BASE_JSON: str = os.path.join(DIRECTORIO_BASE, "results", "reglas_base_integrada.json")
RUTA_REGLAS_OPTIMIZADAS_JSON: str = os.path.join(DIRECTORIO_BASE, "results", "reglas_optimizadas_ga.json")


def generar_grafico_evolucion_reglas(
    ruta_reglas_base: str = RUTA_REGLAS_BASE_JSON,
    ruta_reglas_optimizadas: str = RUTA_REGLAS_OPTIMIZADAS_JSON,
    ruta_salida: str = os.path.join(RUTA_PLOTS, "evolucion_reglas.png"),
) -> str:
    """
    Genera un gráfico conceptual e infografía de la evolución de la base de reglas:
    - Gráfico de barras con los pesos optimizados por el AG para las 26 reglas.
    - Indicación de reglas activas vs podadas.
    - Desglose de origen y balance de clases.
    """
    os.makedirs(os.path.dirname(os.path.abspath(ruta_salida)), exist_ok=True)
    reglas_base = json.load(open(ruta_reglas_base, "r", encoding="utf-8"))
    reglas_opt = json.load(open(ruta_reglas_optimizadas, "r", encoding="utf-8"))

    ids = [r["id_regla"] for r in reglas_opt]
    pesos = [r["peso"] for r in reglas_opt]
    origenes = [r["origen"] for r in reglas_opt]
    consecuentes = [r["consecuente"] for r in reglas_opt]
    activas = [r["activa"] for r in reglas_opt]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=300, gridspec_kw={"width_ratios": [1.4, 1.0]})
    fig.suptitle(
        "Evolución y Optimización de la Base de Reglas por el Algoritmo Genético\nParquedad, Modulación de Pesos y Poda Automática de Reglas Redundantes",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    # Subplot 1: Pesos de Reglas
    colores_barras = []
    for activa, origen in zip(activas, origenes):
        if not activa:
            colores_barras.append("#d9534f")
        elif origen == "PRISM":
            colores_barras.append("#1f77b4")
        else:
            colores_barras.append("#2ca02c")

    barras = ax1.bar(range(len(ids)), pesos, color=colores_barras, edgecolor="#333333", alpha=0.85, width=0.7)

    ax1.axhline(0.05, color="#d9534f", linestyle="--", linewidth=1.2, label="Umbral de Activación ($w_k \\geq 0.05$)")
    ax1.axhline(1.00, color="gray", linestyle=":", linewidth=1.0, label="Peso Inicial Canónico ($w_{ini} = 1.0$)")

    ax1.set_xticks(range(len(ids)))
    ax1.set_xticklabels([f"R{i}" for i in ids], rotation=45, fontsize=8, fontweight="bold")
    ax1.set_xlabel("Identificador de Regla (R1 a R26)", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Peso de Activación Ponderado ($w_k \\in [0.0, 1.0]$)", fontsize=10, fontweight="bold")
    ax1.set_title("Pesos Optimizados por Regla\n(Azul: PRISM Activa, Verde: Apriori Activa, Rojo: Regla Podada)", fontsize=10)
    ax1.set_ylim(-0.05, 1.15)
    ax1.grid(True, linestyle=":", alpha=0.6)

    for i, bar in enumerate(barras):
        h = bar.get_height()
        if h > 0.15:
            ax1.text(bar.get_x() + bar.get_width() / 2.0, h + 0.02, f"{h:.2f}", ha="center", va="bottom", fontsize=7)
        elif not activas[i]:
            ax1.text(bar.get_x() + bar.get_width() / 2.0, 0.02, "OFF", ha="center", va="bottom", color="#d9534f", fontsize=7, fontweight="bold")

    ax1.legend(loc="upper right", framealpha=0.9, fontsize=9)

    # Subplot 2: Síntesis Conceptual y Balance
    ax2.axis("off")

    num_total = len(ids)
    num_activas = sum(activas)
    num_podadas = num_total - num_activas

    conteo_activas_clase = {"Alta": 0, "Baja": 0, "Media": 0}
    conteo_activas_origen = {"PRISM": 0, "Apriori": 0}
    for a, c, o in zip(activas, consecuentes, origenes):
        if a:
            conteo_activas_clase[c] += 1
            conteo_activas_origen[o] += 1

    texto_resumen = (
        "RESUMEN DE OPTIMIZACION DE REGLAS\n"
        "--------------------------------------------------\n"
        f" • Reglas Totales Integradas : {num_total}\n"
        f" • Reglas Activas Post-AG    : {num_activas} ({(num_activas/num_total*100):.1f}%)\n"
        f" • Reglas Podadas (Inactivas): {num_podadas} ({(num_podadas/num_total*100):.1f}%)\n\n"
        "DISTRIBUCION DE REGLAS ACTIVAS\n"
        "--------------------------------------------------\n"
        f" • Por Origen:\n"
        f"     - PRISM   : {conteo_activas_origen['PRISM']} activas\n"
        f"     - Apriori : {conteo_activas_origen['Apriori']} activas\n"
        f" • Por Calidad Objetivo:\n"
        f"     - Calidad Alta  : {conteo_activas_clase['Alta']} reglas\n"
        f"     - Calidad Media : {conteo_activas_clase['Media']} reglas\n"
        f"     - Calidad Baja  : {conteo_activas_clase['Baja']} reglas\n\n"
        "TRANSFORMACION CONCEPTUAL CLAVE\n"
        "--------------------------------------------------\n"
        "Regla 1 (PRISM -> Alta):\n"
        "  SI alcohol ES alto Y vol_acidity ES bajo\n"
        "     Y tot_sulfur ES bajo Y sulphates ES alto\n"
        "  ENTONCES calidad ES alta\n"
        f"  [Peso inicial: 1.00 -> Peso optimizado: {pesos[0]:.4f}]\n\n"
        "Regla 22 (Apriori -> Alta):\n"
        "  SI alcohol ES alto Y density ES bajo\n"
        "     Y fixed_acidity ES alto\n"
        "  ENTONCES calidad ES alta\n"
        f"  [Peso inicial: 1.00 -> Peso optimizado: {pesos[21]:.4f}]"
    )

    ax2.text(
        0.05,
        0.95,
        texto_resumen,
        fontsize=9.5,
        fontfamily="monospace",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.8", fc="#f8f9fa", ec="#cccccc", lw=1.5),
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.94])
    plt.savefig(ruta_salida, dpi=300, bbox_inches="tight")
    plt.close()
    print(f" [+] Gráfico de evolución de reglas guardado: {ruta_salida}")
    return ruta_salida
