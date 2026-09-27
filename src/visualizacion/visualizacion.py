"""
Módulo de Visualización y Generación de Gráficos de Rendimiento
Proyecto: Clasificación de Calidad de Vinos (PRISM, Apriori, Lógica Difusa y Algoritmo Genético)
"""

import os
import sys
from typing import Dict
import matplotlib.pyplot as plt

from src.visualizacion.graficos.convergencia import (
    RUTA_HISTORIAL_CONVERGENCIA_CSV,
    RUTA_PLOTS,
    generar_grafico_convergencia_ga,
)
from src.visualizacion.graficos.mfs import (
    RUTA_PARAMS_MFS_INICIALES_JSON,
    RUTA_PARAMS_MFS_OPTIMIZADAS_JSON,
    generar_grafico_comparativa_mfs,
)
from src.visualizacion.graficos.matrices import generar_grafico_matrices_confusion_comparativas
from src.visualizacion.graficos.reglas import (
    RUTA_REGLAS_BASE_JSON,
    RUTA_REGLAS_OPTIMIZADAS_JSON,
    generar_grafico_evolucion_reglas,
)

# Configuración global de estilo de Matplotlib
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.linewidth"] = 1.0


def generar_todos_los_graficos() -> Dict[str, str]:
    """
    Ejecuta la generación automatizada de todos los gráficos del proyecto
    y los almacena en results/plots/.
    """
    print("\n" + "=" * 80)
    print(" [FASE 13] GENERACION AUTOMATIZADA DE GRAFICOS Y VISUALIZACIONES")
    print("=" * 80)

    os.makedirs(RUTA_PLOTS, exist_ok=True)
    rutas = {}

    rutas["convergencia_fitness_ga"] = generar_grafico_convergencia_ga()
    rutas["comparativa_mfs_pre_post_ga"] = generar_grafico_comparativa_mfs()
    rutas["matrices_confusion_comparativas"] = generar_grafico_matrices_confusion_comparativas()
    rutas["evolucion_reglas"] = generar_grafico_evolucion_reglas()

    print("-" * 80)
    print(" [OK] Todos los gráficos han sido generados exitosamente en 'results/plots/':")
    for nombre, ruta in rutas.items():
        print(f"   - {nombre:<35}: {os.path.basename(ruta)}")
    print("=" * 80 + "\n")

    return rutas


if __name__ == "__main__":
    generar_todos_los_graficos()
