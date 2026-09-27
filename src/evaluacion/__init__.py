"""
Paquete de Evaluación Comparativa Progresiva
"""

from src.evaluacion.metricas.calculo import (
    ORDEN_CLASES,
    calcular_metricas_clasificacion,
)
from src.evaluacion.comparacion.comparador import (
    UMBRAL_PESO_REGLA_ACTIVA,
    construir_dataframe_comparativo,
    evaluar_fase_1_discreto,
    evaluar_fase_2_pre_ga,
    evaluar_fase_3_post_ga,
)
from src.evaluacion.evaluacion import (
    RUTA_PARAMS_MFS_INICIALES_JSON,
    RUTA_PARAMS_MFS_OPTIMIZADAS_JSON,
    RUTA_REGLAS_BASE_JSON,
    RUTA_REGLAS_OPTIMIZADAS_JSON,
    RUTA_SALIDA_EVALUACION_CSV,
    RUTA_SALIDA_METRICAS_JSON,
    cargar_configuracion_mfs,
    cargar_reglas_desde_json,
    ejecutar_evaluacion_comparativa_completa,
    guardar_resultados_evaluacion,
    mostrar_tabla_comparativa,
)

__all__ = [
    "ORDEN_CLASES",
    "UMBRAL_PESO_REGLA_ACTIVA",
    "RUTA_REGLAS_BASE_JSON",
    "RUTA_PARAMS_MFS_INICIALES_JSON",
    "RUTA_PARAMS_MFS_OPTIMIZADAS_JSON",
    "RUTA_REGLAS_OPTIMIZADAS_JSON",
    "RUTA_SALIDA_EVALUACION_CSV",
    "RUTA_SALIDA_METRICAS_JSON",
    "calcular_metricas_clasificacion",
    "evaluar_fase_1_discreto",
    "evaluar_fase_2_pre_ga",
    "evaluar_fase_3_post_ga",
    "construir_dataframe_comparativo",
    "cargar_reglas_desde_json",
    "cargar_configuracion_mfs",
    "guardar_resultados_evaluacion",
    "mostrar_tabla_comparativa",
    "ejecutar_evaluacion_comparativa_completa",
]
