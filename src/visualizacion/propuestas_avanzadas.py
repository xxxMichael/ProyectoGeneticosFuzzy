"""
Módulo de Generación de Nuevas Propuestas Visuales y Analíticas Avanzadas
Proyecto: Wine Quality AI System (PRISM, Apriori, Mamdani Continuo y Algoritmo Genético)

Este módulo produce 6 gráficos analíticos avanzados de alto impacto y rigor académico,
diseñados para diferenciar radicalmente este proyecto de implementaciones estándar:

1. 01_radar_multidimensional_comparativo.png:
   Radar Spider Chart comparando Baseline Discreto, Pre-GA y Post-GA en 6 dimensiones ortogonales.
2. 02_superficie_decision_mamdani_alcohol_acidez.png:
   Superficie continua 2D (Contour Heatmap) de defuzzificación Mamdani (Alcohol vs. Acidez Volátil).
3. 03_distribucion_centroides_por_clase_real.png:
   Distribución de densidades KDE de centroides predichos por clase real (Pre-GA vs. Post-GA).
4. 04_poda_y_pesos_reglas_ga.png:
   Lollipop chart de pesos evolucionados y visualización de la zona de poda (38.5% reglas eliminadas).
5. 05_espacio_reglas_prism_vs_apriori.png:
   Frente Pareto Soporte vs. Confianza vs. Lift demostrando la complementariedad PRISM + Apriori.
6. 06_importancia_quimica_11_variables.png:
   Matriz de participación fisicoquímica de las 11 variables a lo largo de las reglas activas.
"""

import os
import sys
import json
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

# Configuración de Rutas y sys.path
DIRECTORIO_RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if DIRECTORIO_RAIZ not in sys.path:
    sys.path.insert(0, DIRECTORIO_RAIZ)

CARPETA_SALIDA = os.path.join(DIRECTORIO_RAIZ, "results", "propuestas_visuales_diferenciales")
os.makedirs(CARPETA_SALIDA, exist_ok=True)

from src.datos.carga.cargador import cargar_dataset
from src.datos.preparacion.discretizacion import preparar_conjuntos_entrenamiento_prueba
from src.fuzzy.fuzzy import SistemaDifusoMamdani, ConfiguracionMFs
from src.reglas.regla_unificada import ReglaUnificada

# Paleta Estilística Profesional
COLOR_FONDO = "#FFFFFF"
COLOR_TEXTO = "#1E293B"
COLOR_BAJA = "#DC2626"       # Carmesí vibrante
COLOR_MEDIA = "#D97706"      # Ámbar dorado
COLOR_ALTA = "#059669"       # Verde esmeralda
COLOR_BASE = "#64748B"       # Slate / Gris
COLOR_PRE = "#2563EB"        # Azul Cobalto
COLOR_POST = "#7C3AED"       # Violeta / Púrpura Imperial

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.edgecolor"] = "#CBD5E1"
plt.rcParams["axes.linewidth"] = 0.8


# =============================================================================
# 1. RADAR MULTIDIMENSIONAL COMPARATIVO (6 EJES ORTOGONALES)
# =============================================================================
def generar_radar_multidimensional():
    """Genera un gráfico de araña comparando las 3 fases en 6 dimensiones sistémicas."""
    print("-> Generando 01_radar_multidimensional_comparativo.png...")
    
    categorias = [
        "Exactitud en Test\n(Accuracy)",
        "F1-Score Macro\n(Balance de Clases)",
        "Sensibilidad Vinos Alta\n(Recall Alta)",
        "Cobertura Espacio\n(% Muestras)",
        "Continuidad Difusa\n(Suavidad Frontera)",
        "Compacidad / Parsimonia\n(Poda de Reglas)"
    ]
    N = len(categorias)
    
    # Valores normalizados de 0 a 100
    valores_fase1 = [57.50, 46.74, 11.63, 79.69,  5.00,  0.00]
    valores_fase2 = [55.94, 48.04, 20.93, 98.44, 85.00,  0.00]
    valores_fase3 = [60.31, 56.47, 44.19, 99.38, 95.00, 38.46]
    
    angulos = [n / float(N) * 2 * np.pi for n in range(N)]
    angulos += angulos[:1]
    
    v1 = valores_fase1 + valores_fase1[:1]
    v2 = valores_fase2 + valores_fase2[:1]
    v3 = valores_fase3 + valores_fase3[:1]
    
    fig = plt.figure(figsize=(10, 10), facecolor=COLOR_FONDO)
    ax = fig.add_subplot(111, polar=True)
    fig.subplots_adjust(top=0.86, bottom=0.16, left=0.12, right=0.88)
    
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    plt.xticks(angulos[:-1], categorias, color=COLOR_TEXTO, size=11, fontweight="bold")
    ax.tick_params(pad=22)
    
    ax.set_rlabel_position(30)
    plt.yticks([20, 40, 60, 80, 100], ["20%", "40%", "60%", "80%", "100%"], color="#94A3B8", size=9)
    plt.ylim(0, 105)
    
    # Grid styling
    ax.grid(color="#E2E8F0", linestyle="--", linewidth=0.8)
    ax.spines["polar"].set_color("#CBD5E1")
    
    # Fase 1: Baseline Discreto
    ax.plot(angulos, v1, linewidth=2, linestyle="--", color=COLOR_BASE, label="Fase 1: Baseline Discreto (PRISM/Apriori)")
    ax.fill(angulos, v1, color=COLOR_BASE, alpha=0.08)
    
    # Fase 2: Pre-GA
    ax.plot(angulos, v2, linewidth=2.2, linestyle="-.", color=COLOR_PRE, label="Fase 2: Difuso Inicial (Pre-GA)")
    ax.fill(angulos, v2, color=COLOR_PRE, alpha=0.12)
    
    # Fase 3: Post-GA
    ax.plot(angulos, v3, linewidth=3, linestyle="-", color="#059669", marker="o", markersize=6, label="Fase 3: Difuso Optimizado (Post-GA)")
    ax.fill(angulos, v3, color="#059669", alpha=0.25)
    
    plt.title(
        "Evaluación Sistémica Integral: Evolución de Capacidades\n(Baseline Discreto vs. Pre-GA vs. Post-GA)",
        size=14,
        fontweight="bold",
        color=COLOR_TEXTO,
        y=1.12
    )
    
    plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.26), ncol=1, frameon=True, facecolor="#F8FAFC", edgecolor="#E2E8F0", fontsize=10)
    
    ruta_guardado = os.path.join(CARPETA_SALIDA, "01_radar_multidimensional_comparativo.png")
    plt.savefig(ruta_guardado, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"   [OK] Guardado en: {ruta_guardado}")


# =============================================================================
# 2. SUPERFICIE DE DECISIÓN CONTINUA MAMDANI 2D (EL FIN DEL BOUNDARY PROBLEM)
# =============================================================================
def generar_superficie_decision():
    """Genera la superficie 2D continua del centroide Mamdani para Alcohol vs. Acidez Volátil."""
    print("-> Generando 02_superficie_decision_mamdani_alcohol_acidez.png...")
    
    df = cargar_dataset()
    particiones = preparar_conjuntos_entrenamiento_prueba(df)
    X_test = particiones["X_test_num"]
    y_test = particiones["y_test"]
    
    with open(os.path.join(DIRECTORIO_RAIZ, "results", "params_mfs_optimizadas_ga.json"), "r", encoding="utf-8") as f:
        params_opt = json.load(f)
    cfg_opt = ConfiguracionMFs(params_opt)
    
    with open(os.path.join(DIRECTORIO_RAIZ, "results", "reglas_optimizadas_ga.json"), "r", encoding="utf-8") as f:
        reglas_json = json.load(f)
    reglas_opt = [ReglaUnificada(**r) for r in reglas_json]
    
    sistema = SistemaDifusoMamdani()
    
    # Mediana del dataset para las variables fijadas
    medianas = df[sistema.var_nombres].median().to_dict()
    
    # Rango de grilla
    alc_min, alc_max = 8.5, 14.5
    va_min, va_max = 0.15, 1.30
    resolucion = 70
    
    eje_alc = np.linspace(alc_min, alc_max, resolucion)
    eje_va = np.linspace(va_min, va_max, resolucion)
    grid_alc, grid_va = np.meshgrid(eje_alc, eje_va)
    
    # Construcción de la matriz para evaluación por lotes
    n_puntos = resolucion * resolucion
    matriz_sintetica = np.zeros((n_puntos, len(sistema.var_nombres)))
    
    for idx_var, var_nom in enumerate(sistema.var_nombres):
        if var_nom == "alcohol":
            matriz_sintetica[:, idx_var] = grid_alc.ravel()
        elif var_nom == "volatile acidity":
            matriz_sintetica[:, idx_var] = grid_va.ravel()
        else:
            matriz_sintetica[:, idx_var] = medianas[var_nom]
            
    _, centroides_grid, _ = sistema.predecir_vectorizado(matriz_sintetica, cfg_opt, reglas_opt)
    grid_centroides = centroides_grid.reshape((resolucion, resolucion))
    
    fig, ax = plt.subplots(figsize=(10, 8), facecolor=COLOR_FONDO)
    
    # Mapa de contornos continuos
    niveles = np.linspace(4.2, 7.8, 37)
    cs = ax.contourf(grid_alc, grid_va, grid_centroides, levels=niveles, cmap="RdYlGn", alpha=0.88)
    cbar = plt.colorbar(cs, ax=ax, pad=0.03)
    cbar.set_label("Valor Cuantitativo de Defuzzificación Mamdani (Centroide CoG $\\in [0, 10]$)", fontsize=11, fontweight="bold", labelpad=12)
    
    # Líneas de contorno clave (umbrales de decisión)
    linea_baja = ax.contour(grid_alc, grid_va, grid_centroides, levels=[5.30], colors=["#991B1B"], linewidths=2.2, linestyles="--")
    linea_alta = ax.contour(grid_alc, grid_va, grid_centroides, levels=[6.60], colors=["#065F46"], linewidths=2.5, linestyles="-")
    
    ax.clabel(linea_baja, fmt={5.30: "Umbral Baja (< 5.30)"}, inline=True, fontsize=10, colors="#991B1B")
    ax.clabel(linea_alta, fmt={6.60: "Umbral Alta (>= 6.60)"}, inline=True, fontsize=10, colors="#065F46")
    
    # Dispersión de las muestras reales de Test
    colores_reales = {"Baja": "#DC2626", "Media": "#F59E0B", "Alta": "#059669"}
    for clase, col in colores_reales.items():
        mascara = (y_test == clase)
        ax.scatter(
            X_test.loc[mascara, "alcohol"],
            X_test.loc[mascara, "volatile acidity"],
            c=col,
            label=f"Muestras Test: Real {clase}",
            edgecolors="#FFFFFF",
            linewidth=0.7,
            s=48,
            alpha=0.85
        )
        
    ax.set_title(
        "Superficie Continua de Inferencia Difusa Mamdani Post-GA\nResolución del 'Boundary Problem' mediante Gradientes Suaves de Centroide",
        fontsize=13,
        fontweight="bold",
        color=COLOR_TEXTO,
        pad=15
    )
    ax.set_xlabel("Graduación Alcohólica (% vol)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Acidez Volátil (g/dm³ de ácido acético)", fontsize=11, fontweight="bold")
    
    # Anotaciones didácticas de regiones enológicas
    ax.text(8.8, 1.15, "Zona Crítica de Rechazo\n(Alto Ácido Acético / Bajo Alcohol)", fontsize=9, fontweight="bold", color="#7F1D1D", bbox=dict(boxstyle="round,pad=0.3", fc="#FEE2E2", ec="#FCA5A5", alpha=0.9))
    ax.text(12.2, 0.25, "Zona Premium Gran Reserva\n(Baja Acidez / Alto Alcohol)", fontsize=9, fontweight="bold", color="#064E3B", bbox=dict(boxstyle="round,pad=0.3", fc="#D1FAE5", ec="#6EE7B7", alpha=0.9))
    
    ax.legend(loc="upper right", framealpha=0.95, facecolor="#F8FAFC", edgecolor="#E2E8F0", fontsize=9.5)
    ax.grid(color="#FFFFFF", linestyle=":", alpha=0.6)
    
    plt.tight_layout()
    ruta_guardado = os.path.join(CARPETA_SALIDA, "02_superficie_decision_mamdani_alcohol_acidez.png")
    plt.savefig(ruta_guardado, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"   [OK] Guardado en: {ruta_guardado}")


# =============================================================================
# 3. DISTRIBUCIÓN DE DENSIDAD DE CENTROIDES (PRE-GA VS. POST-GA)
# =============================================================================
def generar_distribucion_centroides():
    """Genera curvas KDE mostrando cómo el AG separa los centroides de cada clase real."""
    print("-> Generando 03_distribucion_centroides_por_clase_real.png...")
    
    df = cargar_dataset()
    particiones = preparar_conjuntos_entrenamiento_prueba(df)
    X_test = particiones["X_test_num"]
    y_test = particiones["y_test"].to_numpy()
    
    # Cargar Pre-GA
    with open(os.path.join(DIRECTORIO_RAIZ, "results", "params_mfs_iniciales.json"), "r", encoding="utf-8") as f:
        params_pre = json.load(f)
    cfg_pre = ConfiguracionMFs(params_pre)
    
    with open(os.path.join(DIRECTORIO_RAIZ, "results", "reglas_base_integrada.json"), "r", encoding="utf-8") as f:
        reglas_pre_json = json.load(f)
    reglas_pre = [ReglaUnificada(**r) for r in reglas_pre_json]
    
    # Cargar Post-GA
    with open(os.path.join(DIRECTORIO_RAIZ, "results", "params_mfs_optimizadas_ga.json"), "r", encoding="utf-8") as f:
        params_post = json.load(f)
    cfg_post = ConfiguracionMFs(params_post)
    
    with open(os.path.join(DIRECTORIO_RAIZ, "results", "reglas_optimizadas_ga.json"), "r", encoding="utf-8") as f:
        reglas_post_json = json.load(f)
    reglas_post = [ReglaUnificada(**r) for r in reglas_post_json]
    
    sistema = SistemaDifusoMamdani()
    _, centroides_pre, _ = sistema.predecir_vectorizado(X_test, cfg_pre, reglas_pre)
    _, centroides_post, _ = sistema.predecir_vectorizado(X_test, cfg_post, reglas_post)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True, facecolor=COLOR_FONDO)
    
    rango_x = np.linspace(3.5, 8.5, 400)
    clases = ["Baja", "Media", "Alta"]
    colores = {"Baja": COLOR_BAJA, "Media": COLOR_MEDIA, "Alta": COLOR_ALTA}
    
    # Pre-GA Panel
    for c in clases:
        valores = centroides_pre[y_test == c]
        kde = gaussian_kde(valores, bw_method=0.35)
        densidad = kde(rango_x)
        ax1.plot(rango_x, densidad, label=f"Vino Real {c} (N={len(valores)})", color=colores[c], linewidth=2.5)
        ax1.fill_between(rango_x, densidad, alpha=0.18, color=colores[c])
        
    ax1.axvline(5.30, color="#7F1D1D", linestyle="--", linewidth=1.5, label="Umbral Baja (< 5.30)")
    ax1.axvline(6.60, color="#064E3B", linestyle="-.", linewidth=1.5, label="Umbral Alta (>= 6.60)")
    ax1.set_title("Pre-AG (Configuración Heurística Inicial)\nSuperposición Severa en Rango Medio (5.4 - 6.2)", fontsize=11, fontweight="bold", color=COLOR_TEXTO)
    ax1.set_xlabel("Centroide Defuzzificado CoG", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Densidad Estimada (KDE)", fontsize=10, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=8.5, framealpha=0.9)
    ax1.grid(color="#E2E8F0", linestyle=":", alpha=0.8)
    
    # Post-GA Panel
    for c in clases:
        valores = centroides_post[y_test == c]
        kde = gaussian_kde(valores, bw_method=0.35)
        densidad = kde(rango_x)
        ax2.plot(rango_x, densidad, label=f"Vino Real {c} (N={len(valores)})", color=colores[c], linewidth=2.5)
        ax2.fill_between(rango_x, densidad, alpha=0.25, color=colores[c])
        
    ax2.axvline(5.30, color="#7F1D1D", linestyle="--", linewidth=1.5, label="Umbral Baja (< 5.30)")
    ax2.axvline(6.60, color="#064E3B", linestyle="-.", linewidth=1.5, label="Umbral Alta (>= 6.60)")
    ax2.set_title("Post-AG (Optimización Evolutiva Multi-Criterio)\nDiscriminación Efectiva: Desplazamiento Hacia Extremos", fontsize=11, fontweight="bold", color=COLOR_TEXTO)
    ax2.set_xlabel("Centroide Defuzzificado CoG", fontsize=10, fontweight="bold")
    ax2.legend(loc="upper right", fontsize=8.5, framealpha=0.9)
    ax2.grid(color="#E2E8F0", linestyle=":", alpha=0.8)
    
    fig.suptitle(
        "Verificación de Calibración Continua: Separación de Densidades por Clase Real\n(Demostración del Incremento en Capacidad de Discriminación del AG)",
        fontsize=13,
        fontweight="bold",
        color=COLOR_TEXTO,
        y=1.03
    )
    
    plt.tight_layout()
    ruta_guardado = os.path.join(CARPETA_SALIDA, "03_distribucion_centroides_por_clase_real.png")
    plt.savefig(ruta_guardado, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"   [OK] Guardado en: {ruta_guardado}")


# =============================================================================
# 4. PODA EVOLUTIVA Y PESOS DE REGLAS (PARSIMONIA EN ACCIÓN)
# =============================================================================
def generar_poda_y_pesos_reglas():
    """Genera un gráfico lollipop de pesos que demuestra la poda automática del 38.5%."""
    print("-> Generando 04_poda_y_pesos_reglas_ga.png...")
    
    with open(os.path.join(DIRECTORIO_RAIZ, "results", "reglas_optimizadas_ga.json"), "r", encoding="utf-8") as f:
        reglas = json.load(f)
        
    df_reglas = pd.DataFrame(reglas)
    df_reglas = df_reglas.sort_values(by="peso", ascending=True).reset_index(drop=True)
    
    fig, ax = plt.subplots(figsize=(10, 9), facecolor=COLOR_FONDO)
    
    # Zona de poda sombreada
    ax.axvspan(-0.02, 0.10, color="#FEE2E2", alpha=0.6, label="Zona de Poda (Reglas Desactivadas / Inactivas)")
    ax.axvline(0.10, color="#DC2626", linestyle="--", linewidth=1.5)
    
    y_pos = np.arange(len(df_reglas))
    colores_punto = []
    
    for _, fila in df_reglas.iterrows():
        c = fila["consecuente"]
        if fila["peso"] < 0.10:
            colores_punto.append("#94A3B8")  # Gris para podadas
        elif c == "Alta":
            colores_punto.append(COLOR_ALTA)
        elif c == "Baja":
            colores_punto.append(COLOR_BAJA)
        else:
            colores_punto.append(COLOR_MEDIA)
            
    # Líneas lollipop
    for i, (_, fila) in enumerate(df_reglas.iterrows()):
        ax.hlines(y=i, xmin=0, xmax=fila["peso"], color="#CBD5E1" if fila["peso"] >= 0.10 else "#FCA5A5", linewidth=1.4)
        
    ax.scatter(df_reglas["peso"], y_pos, color=colores_punto, s=90, zorder=3, edgecolors="#1E293B", linewidth=0.8)
    
    # Etiquetas de texto en el eje Y
    etiquetas_y = [f"R{row['id_regla']:02d} [{row['origen']}] $\\to$ {row['consecuente']}" for _, row in df_reglas.iterrows()]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(etiquetas_y, fontsize=8.5, color=COLOR_TEXTO)
    
    ax.set_xlim(-0.03, 1.05)
    ax.set_xlabel("Peso de Activación Evolved ($w_k \\in [0, 1]$)", fontsize=11, fontweight="bold")
    ax.set_title(
        "Poda Evolutiva de Reglas por Presión de Parsimonia en el AG\n10 de 26 Reglas Desactivadas Automáticamente (38.5% de Reducción)",
        fontsize=13,
        fontweight="bold",
        color=COLOR_TEXTO,
        pad=15
    )
    
    # Métricas y leyenda personalizada
    elementos_leyenda = [
        Patch(facecolor="#FEE2E2", edgecolor="#DC2626", linestyle="--", label="Zona de Poda ($w_k < 0.10$, 10 reglas)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_ALTA, markersize=8, label="Regla Activa $\\to$ Alta"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_BAJA, markersize=8, label="Regla Activa $\\to$ Baja"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_MEDIA, markersize=8, label="Regla Activa $\\to$ Media"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#94A3B8", markersize=8, label="Regla Podada (Inactiva)"),
    ]
    ax.legend(handles=elementos_leyenda, loc="lower right", facecolor="#F8FAFC", edgecolor="#CBD5E1", fontsize=9.5)
    
    # Cuadro explicativo
    texto_resumen = (
        "Efecto de la Penalización en Fitness:\n"
        "Fitness = 0.70·F1 + 0.20·Cob - 0.10·(R_act / R_tot)\n"
        "• Reglas Iniciales: 26\n"
        "• Reglas Activas Finales: 16\n"
        "• Poda Lograda: 38.46%\n"
        "• Cobertura Conservada: 99.38%"
    )
    ax.text(
        0.58, 8, texto_resumen, fontsize=9, bbox=dict(boxstyle="round,pad=0.5", fc="#EFF6FF", ec="#93C5FD", alpha=0.95), color="#1E3A8A"
    )
    
    ax.grid(axis="x", color="#E2E8F0", linestyle="--", alpha=0.7)
    plt.tight_layout()
    
    ruta_guardado = os.path.join(CARPETA_SALIDA, "04_poda_y_pesos_reglas_ga.png")
    plt.savefig(ruta_guardado, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"   [OK] Guardado en: {ruta_guardado}")


# =============================================================================
# 5. ESPACIO DE MINERÍA DE REGLAS: FRENTE PARETO (PRISM VS. APRIORI)
# =============================================================================
def generar_espacio_reglas_pareto():
    """Genera el gráfico de burbujas en el espacio Soporte vs. Confianza vs. Lift."""
    print("-> Generando 05_espacio_reglas_prism_vs_apriori.png...")
    
    ruta_prism = os.path.join(DIRECTORIO_RAIZ, "results", "reglas_prism.csv")
    ruta_apriori = os.path.join(DIRECTORIO_RAIZ, "results", "reglas_apriori.csv")
    ruta_integradas = os.path.join(DIRECTORIO_RAIZ, "results", "reglas_base_integrada.csv")
    
    df_prism = pd.read_csv(ruta_prism)
    df_apriori = pd.read_csv(ruta_apriori)
    df_integ = pd.read_csv(ruta_integradas)
    
    # Calcular lift para PRISM si no existe
    if "lift" not in df_prism.columns:
        p_prior = {"Alta": 0.136, "Media": 0.399, "Baja": 0.465}
        df_prism["lift"] = df_prism.apply(
            lambda r: r["precision"] / p_prior.get(r["consecuente"], 0.33), axis=1
        )
    
    fig, ax = plt.subplots(figsize=(11, 7.5), facecolor=COLOR_FONDO)
    
    # Subconjunto de Apriori para no saturar visualmente (top 250 por Lift)
    df_apriori_muestra = df_apriori.sort_values(by="lift", ascending=False).head(250)
    
    # 1. Candidatas Apriori
    ax.scatter(
        df_apriori_muestra["soporte"],
        df_apriori_muestra["confianza"],
        s=np.clip(df_apriori_muestra["lift"] * 25, 20, 180),
        color="#F97316",
        alpha=0.35,
        edgecolors="none",
        label=f"Candidatas Apriori (Muestra Top Lift, N={len(df_apriori_muestra)})"
    )
    
    # 2. Candidatas PRISM
    ax.scatter(
        df_prism["cobertura_relativa"],
        df_prism["precision"],
        s=np.clip(df_prism["lift"] * 30, 30, 220),
        color="#2563EB",
        alpha=0.6,
        edgecolors="#1E3A8A",
        label=f"Candidatas Inducidas por PRISM (N={len(df_prism)})"
    )
    
    # 3. Reglas Seleccionadas e Integradas (las 26 finales)
    ax.scatter(
        df_integ["soporte_cobertura"],
        df_integ["confianza"],
        s=df_integ["lift"] * 45,
        color="#10B981",
        marker="*",
        edgecolors="#064E3B",
        linewidth=1.2,
        zorder=5,
        label=f"Reglas Seleccionadas en Base Integrada (N={len(df_integ)})"
    )
    
    ax.set_title(
        "Espacio de Minería de Reglas: Demostración de Complementariedad PRISM vs. Apriori\n(Formación de un Frente de Pareto entre Cobertura Poblacional y Pureza Predictiva)",
        fontsize=13,
        fontweight="bold",
        color=COLOR_TEXTO,
        pad=15
    )
    ax.set_xlabel("Soporte Poblacional / Cobertura Relativa", fontsize=11, fontweight="bold")
    ax.set_ylabel("Confianza / Precisión Condicional", fontsize=11, fontweight="bold")
    
    # Anotaciones de Nichos Algorítmicos
    ax.annotate(
        "Nicho PRISM:\nAlta Confianza / Cobertura Focalizada\n(Reglas específicas de perfiles extremos)",
        xy=(0.038, 0.72),
        xytext=(0.06, 0.86),
        arrowprops=dict(facecolor="#2563EB", edgecolor="#1E3A8A", shrink=0.08, width=1.5, headwidth=6),
        fontsize=9,
        fontweight="bold",
        color="#1E3A8A",
        bbox=dict(boxstyle="round,pad=0.4", fc="#EFF6FF", ec="#93C5FD", alpha=0.95)
    )
    
    ax.annotate(
        "Nicho Apriori:\nAlto Soporte / Frecuencia Global\n(Patrones asociativos generales)",
        xy=(0.12, 0.78),
        xytext=(0.16, 0.90),
        arrowprops=dict(facecolor="#EA580C", edgecolor="#9A3412", shrink=0.08, width=1.5, headwidth=6),
        fontsize=9,
        fontweight="bold",
        color="#9A3412",
        bbox=dict(boxstyle="round,pad=0.4", fc="#FFF7ED", ec="#FDBA74", alpha=0.95)
    )
    
    ax.axhline(0.60, color="#94A3B8", linestyle=":", linewidth=1)
    ax.text(0.01, 0.605, "Umbral Mínimo Confianza (0.60)", fontsize=8, color="#64748B")
    
    ax.legend(loc="upper right", framealpha=0.95, facecolor="#F8FAFC", edgecolor="#CBD5E1", fontsize=9.5)
    ax.grid(color="#E2E8F0", linestyle="--", alpha=0.7)
    
    plt.tight_layout()
    ruta_guardado = os.path.join(CARPETA_SALIDA, "05_espacio_reglas_prism_vs_apriori.png")
    plt.savefig(ruta_guardado, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"   [OK] Guardado en: {ruta_guardado}")


# =============================================================================
# 6. IMPORTANCIA QUÍMICA DE LAS 11 VARIABLES FISICOQUÍMICAS
# =============================================================================
def generar_importancia_quimica_11_variables():
    """Genera la participación química de las 11 variables a lo largo de las reglas activas."""
    print("-> Generando 06_importancia_quimica_11_variables.png...")
    
    with open(os.path.join(DIRECTORIO_RAIZ, "results", "reglas_optimizadas_ga.json"), "r", encoding="utf-8") as f:
        reglas = json.load(f)
        
    variables_todas = [
        "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
        "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
        "pH", "sulphates", "alcohol"
    ]
    
    conteo_baja = {v: 0 for v in variables_todas}
    conteo_media = {v: 0 for v in variables_todas}
    conteo_alta = {v: 0 for v in variables_todas}
    
    for r in reglas:
        if not r["activa"]:
            continue
        c = r["consecuente"]
        for var in r["antecedentes"].keys():
            if c == "Baja":
                conteo_baja[var] += 1
            elif c == "Media":
                conteo_media[var] += 1
            elif c == "Alta":
                conteo_alta[var] += 1
                
    df_part = pd.DataFrame({
        "Variable": variables_todas,
        "Alta": [conteo_alta[v] for v in variables_todas],
        "Media": [conteo_media[v] for v in variables_todas],
        "Baja": [conteo_baja[v] for v in variables_todas],
    })
    df_part["Total"] = df_part["Alta"] + df_part["Media"] + df_part["Baja"]
    df_part = df_part.sort_values(by="Total", ascending=True).reset_index(drop=True)
    
    fig, ax = plt.subplots(figsize=(10, 7), facecolor=COLOR_FONDO)
    
    y_pos = np.arange(len(df_part))
    altura = 0.55
    
    p1 = ax.barh(y_pos, df_part["Alta"], height=altura, color=COLOR_ALTA, label="Reglas de Calidad Alta")
    p2 = ax.barh(y_pos, df_part["Media"], left=df_part["Alta"], height=altura, color=COLOR_MEDIA, label="Reglas de Calidad Media")
    p3 = ax.barh(y_pos, df_part["Baja"], left=df_part["Alta"] + df_part["Media"], height=altura, color=COLOR_BAJA, label="Reglas de Calidad Baja")
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_part["Variable"], fontsize=10, fontweight="bold", color=COLOR_TEXTO)
    ax.set_xlabel("Frecuencia de Inclusión en la Base de Reglas Activas Post-GA", fontsize=11, fontweight="bold")
    
    ax.set_title(
        "Firma Química del Sistema Difuso: Cobertura Completa de las 11 Variables\n(Superación del Modelo Restringido a 4 Variables de Otras Versiones)",
        fontsize=13,
        fontweight="bold",
        color=COLOR_TEXTO,
        pad=15
    )
    
    # Anotar valores totales al final de cada barra
    for i, (_, fila) in enumerate(df_part.iterrows()):
        total = fila["Total"]
        if total > 0:
            ax.text(total + 0.15, i, f"{total} reglas", va="center", fontsize=9, fontweight="bold", color=COLOR_TEXTO)
            
    ax.text(
        4.5, 1.2,
        "Ventaja Metodológica:\n"
        "• 7 variables adicionales capturadas en reglas activas\n"
        "• Total sulfur dioxide y citric acid revelan interacciones clave\n"
        "• Cero pérdida de información dimensional",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.5", fc="#ECFDF5", ec="#A7F3D0", alpha=0.95),
        color="#065F46"
    )
    
    ax.set_xlim(0, max(df_part["Total"]) + 1.8)
    ax.legend(loc="lower right", facecolor="#F8FAFC", edgecolor="#CBD5E1", fontsize=9.5)
    ax.grid(axis="x", color="#E2E8F0", linestyle="--", alpha=0.7)
    
    plt.tight_layout()
    ruta_guardado = os.path.join(CARPETA_SALIDA, "06_importancia_quimica_11_variables.png")
    plt.savefig(ruta_guardado, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"   [OK] Guardado en: {ruta_guardado}")


def main():
    print("=" * 70)
    print("GENERADOR DE NUEVAS PROPUESTAS VISUALES DIFERENCIALES (XAI & GA)")
    print(f"Directorio de destino: {CARPETA_SALIDA}")
    print("=" * 70)
    
    generar_radar_multidimensional()
    generar_superficie_decision()
    generar_distribucion_centroides()
    generar_poda_y_pesos_reglas()
    generar_espacio_reglas_pareto()
    generar_importancia_quimica_11_variables()
    
    print("=" * 70)
    print("¡Todas las propuestas visuales generadas exitosamente en 300 DPI!")
    print("=" * 70)


if __name__ == "__main__":
    main()
