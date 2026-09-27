"""
Paquete del Sistema Difuso Mamdani
"""

from src.fuzzy.pertenencia.funciones import (
    RANGOS_VARIABLES_DEFAULT,
    ConfiguracionMFs,
    calcular_pertenencia_trapezoidal,
    calcular_pertenencia_triangular,
    validar_cobertura_difusa,
)
from src.fuzzy.inferencia.motor import (
    UMBRAL_ACTIVACION_MINIMA,
    agregar_salida_difusa,
    evaluar_activaciones_reglas,
)
from src.fuzzy.defuzzificacion.centroide import (
    CENTROIDE_NEUTRO_FALLBACK,
    CLASE_NEUTRA_FALLBACK,
    MAX_UNIVERSO_CALIDAD,
    MIN_UNIVERSO_CALIDAD,
    PARAMS_SALIDA_CALIDAD,
    PASO_UNIVERSO_CALIDAD,
    UMBRAL_CORTE_BAJA_MEDIA,
    UMBRAL_CORTE_MEDIA_ALTA,
    UNIVERSO_CALIDAD,
    clasificar_centroide,
    defuzzificar_centroide,
)
from src.fuzzy.fuzzy import (
    ORDEN_CLASES_EVALUACION,
    RUTA_PARAMETROS_DEFAULT_JSON,
    RUTA_REGLAS_INTEGRADAS_JSON,
    ClasificadorFuzzyMamdani,
    SistemaDifusoMamdani,
    cargar_configuracion_mfs_json,
    cargar_reglas_integradas_json,
    ejecutar_smoke_test_fase_9,
    guardar_configuracion_mfs_json,
)

# Re-export de variables fisicoquímicas para compatibilidad
from src.datos.preparacion.discretizacion import VARIABLES_FISICOQUIMICAS

__all__ = [
    "SistemaDifusoMamdani",
    "ClasificadorFuzzyMamdani",
    "ConfiguracionMFs",
    "VARIABLES_FISICOQUIMICAS",
    "RANGOS_VARIABLES_DEFAULT",
    "UNIVERSO_CALIDAD",
    "PARAMS_SALIDA_CALIDAD",
    "UMBRAL_CORTE_BAJA_MEDIA",
    "UMBRAL_CORTE_MEDIA_ALTA",
    "CENTROIDE_NEUTRO_FALLBACK",
    "CLASE_NEUTRA_FALLBACK",
    "UMBRAL_ACTIVACION_MINIMA",
    "ORDEN_CLASES_EVALUACION",
    "RUTA_PARAMETROS_DEFAULT_JSON",
    "RUTA_REGLAS_INTEGRADAS_JSON",
    "calcular_pertenencia_trapezoidal",
    "calcular_pertenencia_triangular",
    "validar_cobertura_difusa",
    "evaluar_activaciones_reglas",
    "agregar_salida_difusa",
    "defuzzificar_centroide",
    "clasificar_centroide",
    "cargar_reglas_integradas_json",
    "cargar_configuracion_mfs_json",
    "guardar_configuracion_mfs_json",
    "ejecutar_smoke_test_fase_9",
]
