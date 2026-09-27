"""
Submódulo de Definición de Regla Unificada
Proyecto: Clasificación de Calidad de Vinos (PRISM, Apriori, Lógica Difusa y Algoritmo Genético)
"""

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class ReglaUnificada:
    """
    Estructura unificada que representa una regla de decisión tanto en formato
    discreto (PRISM/Apriori) como preparada para la transformación difusa.
    """

    id_regla: int
    origen: str  # 'PRISM' o 'Apriori'
    antecedentes: Dict[str, str]  # Ej: {'alcohol': 'Alto', 'volatile acidity': 'Bajo'}
    consecuente: str  # 'Alta', 'Media', 'Baja'
    precision: float  # Pureza o precisión de la regla
    confianza: float  # Confianza condicional
    soporte_cobertura: float  # Soporte o cobertura relativa
    lift: float  # Fuerza de asociación (>1.0 es positiva)
    num_condiciones: int
    regla_texto: str
    peso: float = 1.0  # Peso inicial de activación (optimizable por AG)
    activa: bool = True  # Estado de selección (optimizable por AG)

    def coincide_discreto(self, instancia_discreta: Dict[str, Any]) -> bool:
        """Comprueba si una muestra discreta satisface todos los antecedentes de la regla."""
        for variable, valor_esperado in self.antecedentes.items():
            if str(instancia_discreta.get(variable)) != str(valor_esperado):
                return False
        return True

    def a_texto_difuso(self) -> str:
        """Genera la representación lingüística para el motor de lógica difusa."""
        terminos = [f"{var} ES {val.lower()}" for var, val in self.antecedentes.items()]
        return f"SI {' Y '.join(terminos)} ENTONCES calidad ES {self.consecuente.lower()}"
