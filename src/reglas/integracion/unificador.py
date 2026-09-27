"""
Submódulo de Integración de Reglas (PRISM y Apriori)
Proyecto: Clasificación de Calidad de Vinos (PRISM, Apriori, Lógica Difusa y Algoritmo Genético)
"""

import json
import os
from dataclasses import asdict
from typing import Dict, List, Set
import pandas as pd

from src.reglas.regla_unificada import ReglaUnificada

# Parámetros por defecto de integración
DIRECTORIO_BASE: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RUTA_SALIDA_INTEGRADA_CSV: str = os.path.join(DIRECTORIO_BASE, "results", "reglas_base_integrada.csv")
RUTA_SALIDA_INTEGRADA_JSON: str = os.path.join(DIRECTORIO_BASE, "results", "reglas_base_integrada.json")

MAX_REGLAS_APRIORI_POR_CLASE: int = 5    # Top de mejores reglas asociativas a integrar por clase
MIN_CONFIANZA_APRIORI_FILTRO: float = 0.65  # Filtro estricto de confianza para Apriori
MIN_LIFT_APRIORI_FILTRO: float = 1.30       # Filtro estricto de correlación para Apriori
ORDEN_CLASES: List[str] = ["Alta", "Baja", "Media"]


def parsear_antecedentes_prism(texto_antecedentes: str) -> Dict[str, str]:
    """Parsea el formato 'var1 = Val1 AND var2 = Val2' de PRISM a diccionario."""
    antecedentes: Dict[str, str] = {}
    partes = texto_antecedentes.split(" AND ")
    for parte in partes:
        if "=" in parte:
            var, val = parte.split("=")
            antecedentes[var.strip()] = val.strip()
    return antecedentes


def parsear_antecedentes_apriori(texto_antecedentes: str) -> Dict[str, str]:
    """Parsea el formato 'var1=Val1, var2=Val2' de Apriori a diccionario."""
    antecedentes: Dict[str, str] = {}
    partes = texto_antecedentes.split(",")
    for parte in partes:
        if "=" in parte:
            var, val = parte.split("=")
            antecedentes[var.strip()] = val.strip()
    return antecedentes


def cargar_reglas_prism_csv(ruta_csv: str) -> List[ReglaUnificada]:
    """Carga y normaliza las reglas generadas por el algoritmo PRISM."""
    if not os.path.exists(ruta_csv):
        return []

    df = pd.read_csv(ruta_csv)
    reglas: List[ReglaUnificada] = []

    for _, fila in df.iterrows():
        antecedentes = parsear_antecedentes_prism(str(fila["antecedentes"]))
        regla = ReglaUnificada(
            id_regla=int(fila["id_regla"]),
            origen="PRISM",
            antecedentes=antecedentes,
            consecuente=str(fila["consecuente"]).strip(),
            precision=float(fila["precision"]),
            confianza=float(fila["precision"]),
            soporte_cobertura=float(fila["cobertura_relativa"]),
            lift=1.5,
            num_condiciones=int(fila["num_condiciones"]),
            regla_texto=str(fila["regla_texto"]),
        )
        reglas.append(regla)

    return reglas


def cargar_reglas_apriori_csv(ruta_csv: str) -> List[ReglaUnificada]:
    """Carga y normaliza las reglas generadas por el algoritmo Apriori."""
    if not os.path.exists(ruta_csv):
        return []

    df = pd.read_csv(ruta_csv)
    reglas: List[ReglaUnificada] = []

    for _, fila in df.iterrows():
        antecedentes = parsear_antecedentes_apriori(str(fila["antecedentes"]))
        consecuente = str(fila["consecuente"]).replace("calidad_categoria=", "").strip()
        regla = ReglaUnificada(
            id_regla=int(fila["id_regla"]),
            origen="Apriori",
            antecedentes=antecedentes,
            consecuente=consecuente,
            precision=float(fila["confianza"]),
            confianza=float(fila["confianza"]),
            soporte_cobertura=float(fila["soporte"]),
            lift=float(fila["lift"]),
            num_condiciones=int(fila["num_condiciones"]),
            regla_texto=str(fila["regla_texto"]),
        )
        reglas.append(regla)

    return reglas


def integrar_base_reglas(
    reglas_prism: List[ReglaUnificada],
    reglas_apriori: List[ReglaUnificada],
    max_apriori_por_clase: int = MAX_REGLAS_APRIORI_POR_CLASE,
    min_confianza: float = MIN_CONFIANZA_APRIORI_FILTRO,
    min_lift: float = MIN_LIFT_APRIORI_FILTRO,
) -> List[ReglaUnificada]:
    """
    Integra las reglas de PRISM como base principal y selecciona las mejores reglas
    complementarias de Apriori, eliminando redundancias exactas.
    """
    base_integrada: List[ReglaUnificada] = []
    antecedentes_registrados: Set[str] = set()

    for regla in reglas_prism:
        firma = f"{sorted(regla.antecedentes.items())} -> {regla.consecuente}"
        antecedentes_registrados.add(firma)
        regla.id_regla = len(base_integrada) + 1
        base_integrada.append(regla)

    apriori_filtradas = [
        r
        for r in reglas_apriori
        if r.confianza >= min_confianza and r.lift >= min_lift
    ]
    apriori_filtradas.sort(key=lambda r: (r.lift, r.confianza), reverse=True)

    conteo_apriori_por_clase: Dict[str, int] = {clase: 0 for clase in ORDEN_CLASES}

    for regla in apriori_filtradas:
        clase = regla.consecuente
        if conteo_apriori_por_clase.get(clase, 0) >= max_apriori_por_clase:
            continue

        firma = f"{sorted(regla.antecedentes.items())} -> {regla.consecuente}"
        if firma not in antecedentes_registrados:
            antecedentes_registrados.add(firma)
            conteo_apriori_por_clase[clase] = conteo_apriori_por_clase.get(clase, 0) + 1
            regla.id_regla = len(base_integrada) + 1
            base_integrada.append(regla)

    return base_integrada


def guardar_base_integrada(
    reglas: List[ReglaUnificada],
    ruta_csv: str = RUTA_SALIDA_INTEGRADA_CSV,
    ruta_json: str = RUTA_SALIDA_INTEGRADA_JSON,
) -> None:
    """Exporta la base integrada a formatos CSV y JSON para su uso en el motor difuso."""
    os.makedirs(os.path.dirname(os.path.abspath(ruta_csv)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(ruta_json)), exist_ok=True)

    filas_csv = []
    for r in reglas:
        filas_csv.append(
            {
                "id_regla": r.id_regla,
                "origen": r.origen,
                "antecedentes": str(r.antecedentes),
                "consecuente": r.consecuente,
                "precision": r.precision,
                "confianza": r.confianza,
                "soporte_cobertura": r.soporte_cobertura,
                "lift": r.lift,
                "num_condiciones": r.num_condiciones,
                "regla_texto": r.regla_texto,
                "regla_difusa_texto": r.a_texto_difuso(),
                "peso": r.peso,
                "activa": r.activa,
            }
        )
    df_reglas = pd.DataFrame(filas_csv)
    df_reglas.to_csv(ruta_csv, index=False, encoding="utf-8")

    lista_json = [asdict(r) for r in reglas]
    with open(ruta_json, "w", encoding="utf-8") as f:
        json.dump(lista_json, f, indent=2, ensure_ascii=False)
