"""
Módulo de Carga y Exploración Inicial del Dataset (Shim de Compatibilidad hacia src.datos)
Proyecto: Clasificación de Calidad de Vinos (PRISM, Apriori, Lógica Difusa y Algoritmo Genético)
"""

from src.datos import (
    DIRECTORIO_BASE,
    ENCODING_CSV,
    RUTA_DATASET_DEFAULT,
    SEPARADOR_CSV,
    cargar_dataset,
    mostrar_resumen_carga,
    verificar_integridad,
)

__all__ = [
    "DIRECTORIO_BASE",
    "RUTA_DATASET_DEFAULT",
    "SEPARADOR_CSV",
    "ENCODING_CSV",
    "cargar_dataset",
    "verificar_integridad",
    "mostrar_resumen_carga",
]

if __name__ == "__main__":
    df = cargar_dataset()
    mostrar_resumen_carga(df)
