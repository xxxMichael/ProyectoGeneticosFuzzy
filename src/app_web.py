"""
Servidor Web Interactivo para el Sistema de Clasificación de Calidad de Vinos
Proyecto: PRISM + Apriori + Lógica Difusa Mamdani + Algoritmo Genético
"""

import json
import os
import sys
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request, send_from_directory

# Añadir directorio raíz al path
DIRECTORIO_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, DIRECTORIO_BASE)

from src.carga_datos import cargar_dataset
from src.fuzzy import (
    ConfiguracionMFs,
    SistemaDifusoMamdani,
    VARIABLES_FISICOQUIMICAS,
    RANGOS_VARIABLES_DEFAULT,
    PARAMS_SALIDA_CALIDAD,
    UNIVERSO_CALIDAD,
)
from src.reglas import ReglaUnificada

# =============================================================================
# CONFIGURACIÓN DEL SERVIDOR WEB
# =============================================================================
PUERTO_WEB: int = 5000
HOST_WEB: str = "127.0.0.1"
DEBUG_MODE: bool = False

RUTA_TEMPLATES: str = os.path.join(DIRECTORIO_BASE, "src", "web", "templates")
RUTA_STATIC: str = os.path.join(DIRECTORIO_BASE, "results")
RUTA_PLOTS: str = os.path.join(DIRECTORIO_BASE, "results", "plots")

RUTA_PARAMS_MFS_OPT: str = os.path.join(DIRECTORIO_BASE, "results", "mejores_params.json")
RUTA_PARAMS_MFS_INI: str = os.path.join(DIRECTORIO_BASE, "results", "params_mfs_iniciales.json")
RUTA_REGLAS_OPT: str = os.path.join(DIRECTORIO_BASE, "results", "reglas_optimizadas_ga.json")
RUTA_REGLAS_BASE: str = os.path.join(DIRECTORIO_BASE, "results", "reglas_base_integrada.json")
RUTA_EVALUACION_CSV: str = os.path.join(DIRECTORIO_BASE, "results", "resultados_evaluacion.csv")
RUTA_CONVERGENCIA_CSV: str = os.path.join(DIRECTORIO_BASE, "results", "historial_convergencia_ga.csv")

# Perfiles de Vino Predefinidos para Demostración Rápida
PERFILES_PREDEFINIDOS: Dict[str, Dict[str, float]] = {
    "Vino Gran Reserva (Alta Calidad)": {
        "fixed acidity": 8.5,
        "volatile acidity": 0.28,
        "citric acid": 0.45,
        "residual sugar": 2.1,
        "chlorides": 0.055,
        "free sulfur dioxide": 12.0,
        "total sulfur dioxide": 28.0,
        "density": 0.9942,
        "pH": 3.28,
        "sulphates": 0.82,
        "alcohol": 12.8,
    },
    "Vino de Mesa Estándar (Calidad Media)": {
        "fixed acidity": 7.8,
        "volatile acidity": 0.52,
        "citric acid": 0.22,
        "residual sugar": 2.2,
        "chlorides": 0.082,
        "free sulfur dioxide": 15.0,
        "total sulfur dioxide": 45.0,
        "density": 0.9968,
        "pH": 3.32,
        "sulphates": 0.60,
        "alcohol": 10.1,
    },
    "Vino Picado / Ácido (Calidad Baja)": {
        "fixed acidity": 7.2,
        "volatile acidity": 0.85,
        "citric acid": 0.05,
        "residual sugar": 2.0,
        "chlorides": 0.098,
        "free sulfur dioxide": 6.0,
        "total sulfur dioxide": 75.0,
        "density": 0.9975,
        "pH": 3.45,
        "sulphates": 0.45,
        "alcohol": 9.2,
    },
}
# =============================================================================

app = Flask(__name__, template_folder=RUTA_TEMPLATES)


def inicializar_sistema_difuso() -> Tuple[SistemaDifusoMamdani, List[ReglaUnificada], ConfiguracionMFs]:
    """Carga los parámetros y reglas calibradas por el AG para el clasificador en vivo."""
    # Cargar MFs optimizadas
    if os.path.exists(RUTA_PARAMS_MFS_OPT):
        with open(RUTA_PARAMS_MFS_OPT, "r", encoding="utf-8") as f:
            params_dict = json.load(f)
        config_mfs = ConfiguracionMFs(parametros=params_dict)
    else:
        config_mfs = ConfiguracionMFs()

    # Cargar Reglas optimizadas
    if os.path.exists(RUTA_REGLAS_OPT):
        with open(RUTA_REGLAS_OPT, "r", encoding="utf-8") as f:
            reglas_data = json.load(f)
        reglas = [ReglaUnificada(**r) for r in reglas_data]
    elif os.path.exists(RUTA_REGLAS_BASE):
        with open(RUTA_REGLAS_BASE, "r", encoding="utf-8") as f:
            reglas_data = json.load(f)
        reglas = [ReglaUnificada(**r) for r in reglas_data]
    else:
        reglas = []

    sistema = SistemaDifusoMamdani(config_mfs=config_mfs, reglas=reglas)
    return sistema, reglas, config_mfs


# Inicializar instancia global
SISTEMA_DIFUSO_GLOBAL, REGLAS_GLOBALES, CONFIG_MFS_GLOBAL = inicializar_sistema_difuso()


@app.route("/")
def index():
    """Página principal con la interfaz interactiva."""
    return render_template("index.html")


@app.route("/plots/<path:filename>")
def serve_plot(filename: str):
    """Servir gráficos estáticos generados en results/plots/."""
    return send_from_directory(RUTA_PLOTS, filename)


@app.route("/api/info-inicial", methods=["GET"])
def api_info_inicial():
    """Devuelve metadatos, rangos de variables y perfiles de prueba para la interfaz."""
    return jsonify(
        {
            "variables": VARIABLES_FISICOQUIMICAS,
            "rangos": RANGOS_VARIABLES_DEFAULT,
            "perfiles": PERFILES_PREDEFINIDOS,
            "total_reglas": len(REGLAS_GLOBALES),
            "reglas_activas": sum(1 for r in REGLAS_GLOBALES if r.activa),
        }
    )


@app.route("/api/metricas", methods=["GET"])
def api_metricas():
    """Devuelve las métricas de la evaluación comparativa en las 3 fases."""
    if os.path.exists(RUTA_EVALUACION_CSV):
        df_eval = pd.read_csv(RUTA_EVALUACION_CSV)
        return jsonify(df_eval.to_dict(orient="records"))
    return jsonify([])


@app.route("/api/convergencia", methods=["GET"])
def api_convergencia():
    """Devuelve el historial generacional del Algoritmo Genético."""
    if os.path.exists(RUTA_CONVERGENCIA_CSV):
        df_conv = pd.read_csv(RUTA_CONVERGENCIA_CSV)
        return jsonify(df_conv.to_dict(orient="records"))
    return jsonify([])


@app.route("/api/reglas", methods=["GET"])
def api_reglas():
    """Devuelve la lista detallada de reglas difusas con su estado de activación y pesos."""
    resultado = []
    for r in REGLAS_GLOBALES:
        resultado.append(
            {
                "id_regla": r.id_regla,
                "origen": r.origen,
                "antecedentes": r.antecedentes,
                "consecuente": r.consecuente,
                "precision": round(r.precision, 4),
                "confianza": round(r.confianza, 4),
                "lift": round(r.lift, 4),
                "num_condiciones": r.num_condiciones,
                "regla_texto": r.regla_texto,
                "regla_difusa_texto": r.a_texto_difuso(),
                "peso": round(r.peso, 4),
                "activa": r.activa,
            }
        )
    return jsonify(resultado)


@app.route("/api/mfs", methods=["GET"])
def api_mfs():
    """Devuelve los parámetros de las funciones de pertenencia antes y después del AG."""
    mfs_iniciales = {}
    if os.path.exists(RUTA_PARAMS_MFS_INI):
        with open(RUTA_PARAMS_MFS_INI, "r", encoding="utf-8") as f:
            mfs_iniciales = json.load(f)

    mfs_optimizadas = CONFIG_MFS_GLOBAL.parametros
    return jsonify({"mfs_iniciales": mfs_iniciales, "mfs_optimizadas": mfs_optimizadas})


@app.route("/api/clasificar", methods=["POST"])
def api_clasificar():
    """
    Endpoint principal de inferencia difusa:
    Recibe las 11 variables fisicoquímicas continuas, calcula los grados de activación,
    ejecuta la defuzzificación Mamdani por Centroide y devuelve la explicación de la calidad.
    """
    datos = request.get_json(force=True)
    if not datos:
        return jsonify({"error": "No se recibieron datos de entrada"}), 400

    # Construir muestra asegurando float en las 11 variables
    muestra: Dict[str, float] = {}
    for var in VARIABLES_FISICOQUIMICAS:
        if var not in datos:
            min_v, max_v = RANGOS_VARIABLES_DEFAULT.get(var, (0.0, 10.0))
            muestra[var] = (min_v + max_v) / 2.0
        else:
            muestra[var] = float(datos[var])

    # 1. Fuzzificación individual
    pertenencias_vars = SISTEMA_DIFUSO_GLOBAL.fuzzificar_muestra(muestra)

    # 2. Inferencia difusa y centroide
    clase_predicha, centroide, activaciones = SISTEMA_DIFUSO_GLOBAL.predecir_muestra(muestra)

    # 3. Detalle de reglas activadas con grado alpha_k > 0.001
    detalles_reglas = []
    for r in REGLAS_GLOBALES:
        alpha = activaciones.get(r.id_regla, 0.0)
        if alpha > 0.005:
            detalles_reglas.append(
                {
                    "id_regla": r.id_regla,
                    "origen": r.origen,
                    "consecuente": r.consecuente,
                    "alpha": round(alpha, 4),
                    "peso": round(r.peso, 4),
                    "regla_difusa": r.a_texto_difuso(),
                    "antecedentes": r.antecedentes,
                }
            )

    detalles_reglas.sort(key=lambda x: x["alpha"], reverse=True)

    # 4. Generación de Explicación en Lenguaje Natural (XAI)
    explicacion = _generar_explicacion_enologica(muestra, clase_predicha, centroide, detalles_reglas)

    return jsonify(
        {
            "clase_predicha": clase_predicha,
            "centroide": round(centroide, 4),
            "pertenencias_variables": pertenencias_vars,
            "reglas_activadas": detalles_reglas,
            "total_activadas": len(detalles_reglas),
            "explicacion": explicacion,
        }
    )


def _generar_explicacion_enologica(
    muestra: Dict[str, float],
    clase: str,
    centroide: float,
    reglas_activas: List[Dict[str, Any]],
) -> str:
    """Genera una justificación textual explicativa e interpretable sobre la predicción."""
    alc = muestra.get("alcohol", 10.0)
    va = muestra.get("volatile acidity", 0.5)
    sul = muestra.get("sulphates", 0.6)

    justificaciones = []
    if alc >= 11.5:
        justificaciones.append(f"Graduación alcohólica alta ({alc:.1f}%), aportando estructura y calidez.")
    elif alc <= 9.8:
        justificaciones.append(f"Bajo nivel de alcohol ({alc:.1f}%), asociado a vinos con menor cuerpo.")

    if va <= 0.40:
        justificaciones.append(f"Acidez volátil muy baja ({va:.2f} g/dm³), reflejando pureza aromática sin defectos de vinagre.")
    elif va >= 0.65:
        justificaciones.append(f"Acidez volátil elevada ({va:.2f} g/dm³), sugiriendo indicios de picado o acescencia.")

    if sul >= 0.70:
        justificaciones.append(f"Concentración óptima de sulfatos ({sul:.2f} g/dm³), favoreciendo la conservación antioxidante.")
    elif sul <= 0.50:
        justificaciones.append(f"Nivel bajo de sulfatos ({sul:.2f} g/dm³).")

    texto_reglas = ""
    if reglas_activas:
        top_regla = reglas_activas[0]
        texto_reglas = (
            f"La inferencia estuvo liderada por la Regla #{top_regla['id_regla']} ({top_regla['regla_difusa']}) "
            f"con un grado de activacion de alpha = {top_regla['alpha'] * 100:.1f}%."
        )

    texto_just = " ".join(justificaciones)
    return (
        f"El sistema clasificó este vino como de CALIDAD {clase.upper()} con un centroide Mamdani de {centroide:.2f}/10.0. "
        f"{texto_just} {texto_reglas}"
    )


def iniciar_servidor():
    """Función para arrancar la aplicación web."""
    print("\n" + "=" * 80)
    print(" [WEB SERVER] SERVIDOR WEB DE CLASIFICACION DE CALIDAD DE VINOS (XAI)")
    print("=" * 80)
    print(f" * Interfaz disponible en: http://{HOST_WEB}:{PUERTO_WEB}")
    print(f" * Modo: Produccion / Presentacion Interactiva")
    print("=" * 80 + "\n")
    app.run(host=HOST_WEB, port=PUERTO_WEB, debug=DEBUG_MODE)


if __name__ == "__main__":
    iniciar_servidor()
