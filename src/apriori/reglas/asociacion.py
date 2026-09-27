"""
Submódulo de Reglas de Asociación — Algoritmo Apriori
Proyecto: Clasificación de Calidad de Vinos (PRISM, Apriori, Lógica Difusa y Algoritmo Genético)
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass(frozen=True)
class ReglaAsociacion:
    """
    Representa una regla de asociación descubierta por Apriori orientada a clasificación.
    Estructura: SI (condicion_1 Y condicion_2 ...) ENTONCES calidad_categoria = Clase
    """
    antecedentes: Tuple[str, ...]
    consecuente: str
    soporte: float
    confianza: float
    lift: float
    num_condiciones: int

    def a_texto(self) -> str:
        """Devuelve una representación en lenguaje natural estructurado."""
        condiciones = " Y ".join(self.antecedentes)
        return f"SI {condiciones} ENTONCES {self.consecuente}"

    def a_diccionario(self, id_regla: Optional[int] = None) -> Dict[str, Any]:
        """Serializa la regla en un diccionario para conversión a DataFrame o JSON."""
        datos: Dict[str, Any] = {}
        if id_regla is not None:
            datos["id_regla"] = id_regla
        datos.update({
            "antecedentes": ", ".join(self.antecedentes),
            "consecuente": self.consecuente,
            "soporte": round(float(self.soporte), 4),
            "confianza": round(float(self.confianza), 4),
            "lift": round(float(self.lift), 4),
            "num_condiciones": self.num_condiciones,
            "regla_texto": self.a_texto(),
        })
        return datos


def extraer_reglas_asociacion_filtradas(
    itemsets_frecuentes: Dict[Tuple[str, ...], float],
    nombre_objetivo: str,
    clases_validas: Tuple[str, ...],
    min_confianza: float,
    min_lift: float,
) -> List[ReglaAsociacion]:
    """
    Genera reglas de asociación (A -> B) aplicando el FILTRO ESTRICTO:
    - El consecuente B debe ser una etiqueta válida de calidad_categoria (Baja, Media o Alta).
    - El antecedente A contiene solo características predictoras (no target).
    - Confianza(A -> B) >= min_confianza.
    - Lift(A -> B) > min_lift.
    """
    reglas_descubiertas: List[ReglaAsociacion] = []
    prefijo_objetivo = f"{nombre_objetivo}="
    items_objetivo_validos = {f"{prefijo_objetivo}{clase}" for clase in clases_validas}

    for itemset, soporte_regla in itemsets_frecuentes.items():
        if len(itemset) < 2:
            continue

        items_en_set = set(itemset)
        items_target_presentes = items_en_set.intersection(items_objetivo_validos)

        if len(items_target_presentes) != 1:
            continue

        consecuente = next(iter(items_target_presentes))
        antecedente_items = sorted([item for item in itemset if item != consecuente])
        antecedente_tuple = tuple(antecedente_items)

        soporte_antecedente = itemsets_frecuentes.get(antecedente_tuple)
        soporte_consecuente = itemsets_frecuentes.get((consecuente,))

        if soporte_antecedente is None or soporte_consecuente is None or soporte_antecedente == 0:
            continue

        confianza = soporte_regla / soporte_antecedente
        lift = confianza / soporte_consecuente

        if confianza >= min_confianza and lift > min_lift:
            regla = ReglaAsociacion(
                antecedentes=antecedente_tuple,
                consecuente=consecuente,
                soporte=soporte_regla,
                confianza=confianza,
                lift=lift,
                num_condiciones=len(antecedente_tuple),
            )
            reglas_descubiertas.append(regla)

    reglas_descubiertas.sort(key=lambda r: (r.lift, r.confianza, r.soporte), reverse=True)
    return reglas_descubiertas
