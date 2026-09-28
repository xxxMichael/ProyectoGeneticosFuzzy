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
from src.datos.preparacion.discretizacion import preparar_conjuntos_entrenamiento_prueba
from src.genetico.genetico import OptimizadorGeneticoFuzzy
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

# Perfiles de Vino Predefinidos de Prueba
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
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.jinja_env.auto_reload = True


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

PARAMS_MFS_INICIALES: Dict[str, List[float]] = {}
if os.path.exists(RUTA_PARAMS_MFS_INI):
    try:
        with open(RUTA_PARAMS_MFS_INI, "r", encoding="utf-8") as f:
            PARAMS_MFS_INICIALES = json.load(f)
    except Exception as e:
        print(f"Advertencia al cargar MFs iniciales: {e}")


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


DATOS_TRAIN_CACHE = None


@app.route("/api/ejecutar-ga-micro", methods=["POST"])
def api_ejecutar_ga_micro():
    """
    Ejecuta una micro-optimización genética interactiva en vivo usando DEAP.
    Permite al usuario experimentar con diferentes hiperparámetros y ver la convergencia en tiempo real.
    """
    global DATOS_TRAIN_CACHE
    datos = request.get_json(silent=True) or {}

    num_gen = min(15, max(2, int(datos.get("generaciones", 4))))
    tam_pob = min(25, max(6, int(datos.get("poblacion", 10))))
    prob_cruce = float(datos.get("prob_cruce", 0.85))
    prob_mutacion = float(datos.get("prob_mutacion", 0.25))
    peso_f1 = float(datos.get("peso_f1", 0.70))
    peso_cob = float(datos.get("peso_cobertura", 0.20))
    peso_poda = float(datos.get("peso_penalizacion", 0.10))

    try:
        if DATOS_TRAIN_CACHE is None:
            df = cargar_dataset()
            part = preparar_conjuntos_entrenamiento_prueba(df)
            DATOS_TRAIN_CACHE = (part["X_train_num"], part["y_train_cat"])

        X_tr, y_tr = DATOS_TRAIN_CACHE

        opt = OptimizadorGeneticoFuzzy(
            X_train=X_tr,
            y_train=y_tr,
            reglas_base=REGLAS_GLOBALES,
            peso_f1=peso_f1,
            peso_cobertura=peso_cob,
            peso_penalizacion=peso_poda,
        )

        res = opt.optimizar(
            num_generaciones=num_gen,
            tamano_poblacion=tam_pob,
            prob_cruce=prob_cruce,
            prob_mutacion=prob_mutacion,
            verbose=False,
        )

        historial = []
        for p in res["historial_progreso"]:
            historial.append({
                "generacion": int(p["generacion"]),
                "mejor_fitness": round(float(p["mejor_fitness"]), 4),
                "fitness_promedio": round(float(p["fitness_promedio"]), 4),
                "f1_macro": round(float(p["f1_macro"]), 4),
                "cobertura": round(float(p["cobertura"]) * 100, 2),
                "reglas_activas": int(p["reglas_activas"]),
                "tiempo_seg": round(float(p["tiempo_seg"]), 2),
            })

        return jsonify({
            "status": "success",
            "parametros": {
                "generaciones": num_gen,
                "poblacion": tam_pob,
                "prob_cruce": prob_cruce,
                "prob_mutacion": prob_mutacion,
                "pesos": {"f1": peso_f1, "cobertura": peso_cob, "poda": peso_poda},
            },
            "mejor_fitness": round(float(res["mejor_fitness"]), 4),
            "f1_macro": round(float(res["metricas_train"]["f1_macro"]), 4),
            "cobertura": round(float(res["metricas_train"]["cobertura"]) * 100, 2),
            "reglas_activas": int(res["metricas_train"]["reglas_activas"]),
            "tiempo_total_seg": round(float(res["tiempo_total_seg"]), 2),
            "historial": historial,
        })
    except Exception as e:
        return jsonify({"status": "error", "mensaje": str(e)}), 500


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
    ejecuta la defuzzificación Mamdani por Centroide y devuelve la interpretación enológica de la calidad.
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

    # 4. Generación de Diagnóstico Enológico en Lenguaje Natural
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


@app.route("/fuzzy-explorer")
def fuzzy_explorer():
    """Vista dedicada: Visualizador Pedagógico de Lógica Difusa Mamdani en Acción (XAI)."""
    return render_template("fuzzy_explorer.html")


@app.route("/api/fuzzy-explorer-data", methods=["GET", "POST"])
def api_fuzzy_explorer_data():
    """
    Devuelve la descomposición matemática completa de la inferencia difusa:
    - Puntos de corte (a, b, c) y pertenencias de cada una de las 11 variables.
    - Cuello de botella (mínimo) y fuerza de disparo para cada regla activa.
    - Polígono de defuzzificación continua recortado (Mamdani MAX) y centroide CoG.
    """
    if request.method == "POST":
        datos = request.get_json(silent=True) or {}
    else:
        datos = {}

    muestra: Dict[str, float] = {}
    for var in VARIABLES_FISICOQUIMICAS:
        var_underscore = var.replace(" ", "_")
        if var in datos:
            muestra[var] = float(datos[var])
        elif var_underscore in datos:
            muestra[var] = float(datos[var_underscore])
        else:
            min_v, max_v = RANGOS_VARIABLES_DEFAULT.get(var, (0.0, 10.0))
            muestra[var] = round((min_v + max_v) / 2.0, 3)

    pertenencias = SISTEMA_DIFUSO_GLOBAL.fuzzificar_muestra(muestra)
    activaciones = SISTEMA_DIFUSO_GLOBAL.evaluar_activaciones_reglas(pertenencias)
    mu_agregado, max_por_clase = SISTEMA_DIFUSO_GLOBAL.agregar_salida_difusa(activaciones)
    centroide = SISTEMA_DIFUSO_GLOBAL.defuzzificar_centroide(mu_agregado)
    clase_predicha = SISTEMA_DIFUSO_GLOBAL.clasificar_centroide(centroide)

    # Mapeo de unidades para UI
    unidades_map = {
        "fixed acidity": "g/dm³",
        "volatile acidity": "g/dm³",
        "citric acid": "g/dm³",
        "residual sugar": "g/dm³",
        "chlorides": "g/dm³",
        "free sulfur dioxide": "mg/dm³",
        "total sulfur dioxide": "mg/dm³",
        "density": "g/cm³",
        "pH": "pH",
        "sulphates": "g/dm³",
        "alcohol": "% vol",
    }

    variables_info = {}
    for var in VARIABLES_FISICOQUIMICAS:
        min_v, max_v = RANGOS_VARIABLES_DEFAULT[var]
        a, b, c = CONFIG_MFS_GLOBAL.parametros[var]
        cortes_ini = PARAMS_MFS_INICIALES.get(var, [a, b, c])
        val = muestra[var]
        mu_b = pertenencias[var].get("Bajo", 0.0)
        mu_m = pertenencias[var].get("Medio", 0.0)
        mu_a = pertenencias[var].get("Alto", 0.0)
        variables_info[var] = {
            "nombre": var,
            "unidad": unidades_map.get(var, ""),
            "min": min_v,
            "max": max_v,
            "cortes": [round(a, 4), round(b, 4), round(c, 4)],
            "puntos_corte": {"a": round(a, 4), "b": round(b, 4), "c": round(c, 4)},
            "cortes_iniciales": [round(cortes_ini[0], 4), round(cortes_ini[1], 4), round(cortes_ini[2], 4)],
            "puntos_corte_iniciales": {
                "a": round(cortes_ini[0], 4),
                "b": round(cortes_ini[1], 4),
                "c": round(cortes_ini[2], 4),
            },
            "valor_actual": round(val, 4),
            "pertenencias": {
                "Bajo": round(mu_b, 4),
                "Medio": round(mu_m, 4),
                "Alto": round(mu_a, 4),
            },
        }

    # Desglose de Reglas (Matriz completa de 26 reglas y lista de activas con Cuello de Botella)
    matriz_26_reglas = []
    for r in REGLAS_GLOBALES:
        alpha = activaciones.get(r.id_regla, 0.0)
        es_podada = (not r.activa) or (r.peso <= 0.0)
        antecedentes_eval = []
        min_mu = 1.0
        cuello_botella_idx = -1

        for idx, (v, etiqueta) in enumerate(r.antecedentes.items()):
            mu_val = pertenencias[v].get(etiqueta, 0.0)
            if mu_val < min_mu:
                min_mu = mu_val
                cuello_botella_idx = idx
            antecedentes_eval.append({
                "variable": v,
                "etiqueta": etiqueta,
                "mu": round(mu_val, 4),
                "es_cuello_botella": False,
            })

        if 0 <= cuello_botella_idx < len(antecedentes_eval):
            antecedentes_eval[cuello_botella_idx]["es_cuello_botella"] = True

        matriz_26_reglas.append({
            "id_regla": r.id_regla,
            "origen": r.origen,
            "consecuente": r.consecuente,
            "peso": round(r.peso, 4),
            "alpha": round(alpha, 4),
            "activa": r.activa,
            "podada": es_podada,
            "cuello_botella_mu": round(min_mu, 4),
            "antecedentes": antecedentes_eval,
            "regla_texto": r.regla_texto,
            "regla_difusa": r.a_texto_difuso(),
        })

    # Reglas activas para la lista jerárquica
    reglas_detalle = [r for r in matriz_26_reglas if not r["podada"]]
    reglas_detalle.sort(key=lambda x: x["alpha"], reverse=True)

    # Ordenar matriz de 26 por ID
    matriz_26_reglas_ordenada = sorted(matriz_26_reglas, key=lambda x: x["id_regla"])

    # Defuzzificación y Polígono de Salida
    universo = SISTEMA_DIFUSO_GLOBAL.universo_calidad.tolist()
    corte_baja = np.minimum(max_por_clase.get("Baja", 0.0), SISTEMA_DIFUSO_GLOBAL.mf_salida_baja).tolist()
    corte_media = np.minimum(max_por_clase.get("Media", 0.0), SISTEMA_DIFUSO_GLOBAL.mf_salida_media).tolist()
    corte_alta = np.minimum(max_por_clase.get("Alta", 0.0), SISTEMA_DIFUSO_GLOBAL.mf_salida_alta).tolist()

    return jsonify({
        "variables": variables_info,
        "muestra_actual": muestra,
        "reglas": reglas_detalle,
        "matriz_26_reglas": matriz_26_reglas_ordenada,
        "total_reglas_activas": len([r for r in reglas_detalle if r["alpha"] > 0.005]),
        "defuzzificacion": {
            "universo": universo,
            "mu_agregado": [round(float(v), 4) for v in mu_agregado],
            "corte_baja": [round(float(v), 4) for v in corte_baja],
            "corte_media": [round(float(v), 4) for v in corte_media],
            "corte_alta": [round(float(v), 4) for v in corte_alta],
            "alphas_clase": {k: round(float(v), 4) for k, v in max_por_clase.items()},
            "centroide": round(float(centroide), 4),
            "clase_predicha": clase_predicha,
            "umbrales": {
                "baja_media": SISTEMA_DIFUSO_GLOBAL.umbral_baja_media,
                "media_alta": SISTEMA_DIFUSO_GLOBAL.umbral_media_alta,
            },
        },
        "perfiles_predefinidos": {
            "gran_reserva": PERFILES_PREDEFINIDOS["Vino Gran Reserva (Alta Calidad)"],
            "vino_comercial": PERFILES_PREDEFINIDOS["Vino de Mesa Estándar (Calidad Media)"],
            "vino_picado": PERFILES_PREDEFINIDOS["Vino Picado / Ácido (Calidad Baja)"],
            "descriptivos": PERFILES_PREDEFINIDOS,
        },
    })


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
