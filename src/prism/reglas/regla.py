"""
Submódulo de Estructura de Reglas PRISM
Proyecto: Clasificación de Calidad de Vinos (PRISM, Apriori, Lógica Difusa y Algoritmo Genético)
"""

from dataclasses import dataclass
from typing import Dict
import pandas as pd


@dataclass
class ReglaPRISM:
    """
    Representación estructurada de una regla modular descubierta por el algoritmo PRISM.
    """

    id_regla: int
    antecedentes: Dict[str, str]        # Pares {variable: valor_discreto}
    consecuente: str                    # Clase de calidad predicha ('Baja', 'Media', 'Alta')
    precision: float                    # Precisión de la regla sobre el dataset de entrenamiento
    cobertura_relativa: float           # Proporción de registros del dataset cubiertos por el antecedente
    cobertura_absoluta: int             # Cantidad total de registros que satisfacen el antecedente
    instancias_correctas: int           # Cantidad de registros que satisfacen antecedente y consecuente
    num_condiciones: int                # Número de términos combinados en el antecedente
    regla_texto: str                    # Cadena interpretable en formato: SI cond1 Y cond2 ENTONCES clase

    def cumple_antecedentes(self, fila: pd.Series) -> bool:
        """
        Evalúa si una instancia individual satisface todas las condiciones del antecedente.

        Args:
            fila (pd.Series): Muestra con variables discretizadas.

        Returns:
            bool: True si la instancia cumple todos los antecedentes, False en caso contrario.
        """
        for variable, valor in self.antecedentes.items():
            if variable not in fila or str(fila[variable]) != str(valor):
                return False
        return True

    def evaluar_en_dataframe(self, dataframe_x: pd.DataFrame) -> pd.Series:
        """
        Genera una máscara booleana indicando qué filas de un DataFrame cumplen la regla.

        Args:
            dataframe_x (pd.DataFrame): Dataset con variables discretizadas.

        Returns:
            pd.Series: Serie booleana con True en las instancias cubiertas.
        """
        mascara = pd.Series(True, index=dataframe_x.index)
        for variable, valor in self.antecedentes.items():
            if variable in dataframe_x.columns:
                mascara &= dataframe_x[variable].astype(str) == str(valor)
            else:
                mascara &= False
        return mascara
