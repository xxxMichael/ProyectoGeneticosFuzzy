"""
Paquete de Visualización y Gráficos
"""

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
from src.visualizacion.visualizacion import generar_todos_los_graficos

__all__ = [
    "RUTA_PLOTS",
    "RUTA_HISTORIAL_CONVERGENCIA_CSV",
    "RUTA_PARAMS_MFS_INICIALES_JSON",
    "RUTA_PARAMS_MFS_OPTIMIZADAS_JSON",
    "RUTA_REGLAS_BASE_JSON",
    "RUTA_REGLAS_OPTIMIZADAS_JSON",
    "generar_grafico_convergencia_ga",
    "generar_grafico_comparativa_mfs",
    "generar_grafico_matrices_confusion_comparativas",
    "generar_grafico_evolucion_reglas",
    "generar_todos_los_graficos",
]
