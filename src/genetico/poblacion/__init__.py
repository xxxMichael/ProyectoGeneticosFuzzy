"""
Submódulo de Población Inicial
"""

from src.genetico.poblacion.inicializacion import (
    crear_individuo_perturbado,
    crear_individuo_semilla,
)

__all__ = ["crear_individuo_semilla", "crear_individuo_perturbado"]
