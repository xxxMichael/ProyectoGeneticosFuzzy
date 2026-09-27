"""
Paquete del Algoritmo Genético para Optimización Difusa
"""

from src.genetico.cromosoma.estructura import (
    LONGITUD_CROMOSOMA_MFS,
    LONGITUD_CROMOSOMA_REGLAS,
    LONGITUD_TOTAL_CROMOSOMA,
    UMBRAL_ACTIVACION_REGLA,
    decodificar_cromosoma,
)
from src.genetico.cromosoma.reparacion import reparar_individuo
from src.genetico.poblacion.inicializacion import (
    crear_individuo_perturbado,
    crear_individuo_semilla,
)
from src.genetico.cruce.cruce import cruzar_individuos_mixtos
from src.genetico.mutacion.mutacion import mutar_individuo_mixto
from src.genetico.fitness.evaluacion import (
    PESO_COBERTURA,
    PESO_F1_MACRO,
    PESO_PENALIZACION_REGLAS,
    evaluar_individuo_fitness,
)
from src.genetico.genetico import (
    NUM_GENERACIONES,
    PROBABILIDAD_CRUCE,
    PROBABILIDAD_MUTACION,
    RUTA_PARAMS_MFS_INICIALES_JSON,
    RUTA_REGLAS_INTEGRADAS_JSON,
    RUTA_SALIDA_HISTORIAL_CONVERGENCIA_CSV,
    RUTA_SALIDA_MFS_OPTIMIZADAS_JSON,
    RUTA_SALIDA_REGLAS_OPTIMIZADAS_JSON,
    SEED_ALEATORIA,
    TAMANO_POBLACION,
    TAMANO_TORNEO,
    TASA_ELITISMO,
    OptimizadorGeneticoFuzzy,
    ejecutar_smoke_test_fase_10,
)

__all__ = [
    "OptimizadorGeneticoFuzzy",
    "LONGITUD_CROMOSOMA_MFS",
    "LONGITUD_CROMOSOMA_REGLAS",
    "LONGITUD_TOTAL_CROMOSOMA",
    "UMBRAL_ACTIVACION_REGLA",
    "TAMANO_POBLACION",
    "NUM_GENERACIONES",
    "PROBABILIDAD_CRUCE",
    "PROBABILIDAD_MUTACION",
    "TASA_ELITISMO",
    "TAMANO_TORNEO",
    "SEED_ALEATORIA",
    "PESO_F1_MACRO",
    "PESO_COBERTURA",
    "PESO_PENALIZACION_REGLAS",
    "RUTA_REGLAS_INTEGRADAS_JSON",
    "RUTA_PARAMS_MFS_INICIALES_JSON",
    "RUTA_SALIDA_MFS_OPTIMIZADAS_JSON",
    "RUTA_SALIDA_REGLAS_OPTIMIZADAS_JSON",
    "RUTA_SALIDA_HISTORIAL_CONVERGENCIA_CSV",
    "crear_individuo_semilla",
    "reparar_individuo",
    "crear_individuo_perturbado",
    "cruzar_individuos_mixtos",
    "mutar_individuo_mixto",
    "decodificar_cromosoma",
    "evaluar_individuo_fitness",
    "ejecutar_smoke_test_fase_10",
]
