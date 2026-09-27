"""
Paquete de Gestión, Carga y Preparación de Datos
"""

from src.datos.carga.cargador import (
    DIRECTORIO_BASE,
    ENCODING_CSV,
    RUTA_DATASET_DEFAULT,
    SEPARADOR_CSV,
    cargar_dataset,
    mostrar_resumen_carga,
    verificar_integridad,
)
from src.datos.preparacion.discretizacion import (
    ETIQUETAS_INTERVALOS,
    MAPEO_CALIDAD,
    NOMBRE_CALIDAD_CATEGORICA,
    NOMBRE_VARIABLE_OBJETIVO,
    NUMERO_INTERVALOS,
    PORCENTAJE_TEST,
    SEED_ALEATORIA,
    VARIABLES_FISICOQUIMICAS,
    calcular_estadisticas_descriptivas,
    categorizar_calidad,
    discretizar_por_cuantiles,
    mostrar_resumen_preparacion,
    preparar_conjuntos_entrenamiento_prueba,
    separar_caracteristicas_objetivo,
)
from src.datos.datos import ejecutar_pipeline_datos

__all__ = [
    "DIRECTORIO_BASE",
    "RUTA_DATASET_DEFAULT",
    "SEPARADOR_CSV",
    "ENCODING_CSV",
    "cargar_dataset",
    "verificar_integridad",
    "mostrar_resumen_carga",
    "NOMBRE_VARIABLE_OBJETIVO",
    "NOMBRE_CALIDAD_CATEGORICA",
    "VARIABLES_FISICOQUIMICAS",
    "MAPEO_CALIDAD",
    "PORCENTAJE_TEST",
    "SEED_ALEATORIA",
    "NUMERO_INTERVALOS",
    "ETIQUETAS_INTERVALOS",
    "separar_caracteristicas_objetivo",
    "calcular_estadisticas_descriptivas",
    "categorizar_calidad",
    "discretizar_por_cuantiles",
    "mostrar_resumen_preparacion",
    "preparar_conjuntos_entrenamiento_prueba",
    "ejecutar_pipeline_datos",
]
