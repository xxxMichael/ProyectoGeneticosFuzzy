# Registro de Progreso y Decisiones Técnicas — Clasificación de Calidad de Vinos

Este documento registra el avance del proyecto, el estado de cada una de las fases y las decisiones técnicas acordadas durante la sesión de `/grill-me`.

---

## 1. Decisiones Técnicas Consensuadas (/grill-me)

* **Pila de Desarrollo y Algoritmos:**
  * **PRISM:** Implementación completa desde cero en Python puro / NumPy.
  * **Apriori:** Implementación completa desde cero en Python puro / NumPy.
  * **Lógica Difusa:** Implementado con motor Mamdani continuo vectorizado y funciones de pertenencia analíticas (`scikit-fuzzy` / NumPy).
  * **Algoritmo Genético:** Implementado mediante librerías (`deap` / `numpy`).
* **Estrategia de Discretización:**
  * Discretización por cuantiles / percentiles (**Equal-Frequency: 33% Bajo, 33% Medio, 33% Alto**) para las 11 variables fisicoquímicas continuas.
* **Integración y Selección de Reglas:**
  * **PRISM** como motor principal de reglas de clasificación.
  * Incorporación de reglas de **Apriori** únicamente cuando el consecuente apunte a la calidad (`quality = Baja/Media/Alta`) con $\text{Confianza} \ge 0.6$ y $\text{Lift} > 1.2$, descartando duplicados.
* **Alcance de Variables Difusas:**
  * Fuzzificación obligatoria de las **11 variables fisicoquímicas** del dataset.
* **Funciones de Pertenencia (MFs):**
  * **Configuración Híbrida Trapezoidal-Triangular:**
    * `Bajo`: Trapezoidal con hombro izquierdo abierto ($\mu=1.0$ en valores mínimos).
    * `Medio`: Triangular centrada.
    * `Alto`: Trapezoidal con hombro derecho abierto ($\mu=1.0$ en valores altos).
* **Cromosoma del Algoritmo Genético:**
  * **Representación Mixta (59 genes):** 33 genes reales para puntos de corte de las MFs de las 11 variables + 26 genes reales para activación y modulación de peso de reglas en $[0.0, 1.0]$.
* **Función de Aptitud (Fitness):**
  * Multi-criterio ponderado:
    $$\text{Fitness} = 0.70 \cdot \text{F1\_Macro} + 0.20 \cdot \text{Cobertura} - 0.10 \cdot \left(\frac{\text{Reglas\_Activas}}{\text{Reglas\_Totales}}\right)$$
* **Defuzzificación y Clasificación:**
  * Inferencia tipo **Mamdani continuo con Centroide**: Asignación cuantitativa ($Baja=4.0, Media=6.0, Alta=7.8$), cálculo de centroide y umbralización para la predicción final.
* **Modelo Base y Comparación Progresiva:**
  * **Sin algoritmos de ML externos** (no árboles de decisión externos ni random forest).
  * Comparación estrictamente interna y progresiva en 3 fases:
    1. **Fase 1 (Modelo Base):** Clasificador de reglas discretas iniciales (PRISM + Apriori).
    2. **Fase 2:** Reglas con Lógica Difusa inicial (Pre-AG).
    3. **Fase 3:** Reglas con Lógica Difusa optimizadas con Algoritmo Genético (Post-AG).
* **Artefactos y Visualizaciones:**
  * Curva de convergencia del AG, gráficos de MFs (antes/después), matrices de confusión por fase y tabla comparativa de evolución de reglas.

---

## 2. Estado de Fases de Implementación (100% Completado)

| Fase | Descripción | Estado | Archivos Principales |
|---|---|---|---|
| **Fase 1** | Carga y exploración del dataset | ✅ Completada | [`src/datos/carga/cargador.py`](file:///c:/Dev/7-semestre/AI2/ProyectoGeneticosFuzzy/src/datos/carga/cargador.py) |
| **Fase 2** | Limpieza, separación y estadísticas descriptivas | ✅ Completada | [`src/datos/preparacion/discretizacion.py`](file:///c:/Dev/7-semestre/AI2/ProyectoGeneticosFuzzy/src/datos/preparacion/discretizacion.py) |
| **Fase 3** | Creación de categorías de calidad (Baja, Media, Alta) | ✅ Completada | [`src/datos/preparacion/discretizacion.py`](file:///c:/Dev/7-semestre/AI2/ProyectoGeneticosFuzzy/src/datos/preparacion/discretizacion.py) |
| **Fase 4** | Discretización de variables (Equal-Frequency 33%) | ✅ Completada | [`src/datos/preparacion/discretizacion.py`](file:///c:/Dev/7-semestre/AI2/ProyectoGeneticosFuzzy/src/datos/preparacion/discretizacion.py) |
| **Fase 5** | Implementación de PRISM desde cero | ✅ Completada | [`src/prism/prism.py`](file:///c:/Dev/7-semestre/AI2/ProyectoGeneticosFuzzy/src/prism/prism.py), [`results/reglas_prism.csv`](file:///c:/Dev/7-semestre/AI2/ProyectoGeneticosFuzzy/results/reglas_prism.csv) |
| **Fase 6** | Implementación de Apriori desde cero | ✅ Completada | [`src/apriori/apriori.py`](file:///c:/Dev/7-semestre/AI2/ProyectoGeneticosFuzzy/src/apriori/apriori.py), [`results/reglas_apriori.csv`](file:///c:/Dev/7-semestre/AI2/ProyectoGeneticosFuzzy/results/reglas_apriori.csv) |
| **Fase 7** | Selección e integración de reglas (Modelo Base) | ✅ Completada | [`src/reglas/reglas.py`](file:///c:/Dev/7-semestre/AI2/ProyectoGeneticosFuzzy/src/reglas/reglas.py), [`results/reglas_base_integrada.csv`](file:///c:/Dev/7-semestre/AI2/ProyectoGeneticosFuzzy/results/reglas_base_integrada.csv) |
| **Fase 8** | Diseño de variables difusas (11 variables, Trap-Tri) | ✅ Completada | [`src/fuzzy/pertenencia/funciones.py`](file:///c:/Dev/7-semestre/AI2/ProyectoGeneticosFuzzy/src/fuzzy/pertenencia/funciones.py), [`results/params_mfs_iniciales.json`](file:///c:/Dev/7-semestre/AI2/ProyectoGeneticosFuzzy/results/params_mfs_iniciales.json) |
| **Fase 9** | Motor de inferencia difusa y Mamdani centroide | ✅ Completada | [`src/fuzzy/fuzzy.py`](file:///c:/Dev/7-semestre/AI2/ProyectoGeneticosFuzzy/src/fuzzy/fuzzy.py) |
| **Fase 10** | Implementación de Algoritmo Genético (DEAP/NumPy) | ✅ Completada | [`src/genetico/genetico.py`](file:///c:/Dev/7-semestre/AI2/ProyectoGeneticosFuzzy/src/genetico/genetico.py) |
| **Fase 11** | Optimización de funciones de pertenencia y reglas | ✅ Completada | [`src/genetico/genetico.py`](file:///c:/Dev/7-semestre/AI2/ProyectoGeneticosFuzzy/src/genetico/genetico.py), [`results/params_mfs_optimizadas_ga.json`](file:///c:/Dev/7-semestre/AI2/ProyectoGeneticosFuzzy/results/params_mfs_optimizadas_ga.json) |
| **Fase 12** | Evaluación comparativa en 3 fases | ✅ Completada | [`src/evaluacion/evaluacion.py`](file:///c:/Dev/7-semestre/AI2/ProyectoGeneticosFuzzy/src/evaluacion/evaluacion.py) |
| **Fase 13** | Visualización, gráficos y orquestación final | ✅ Completada | [`src/main.py`](file:///c:/Dev/7-semestre/AI2/ProyectoGeneticosFuzzy/src/main.py), [`src/visualizacion/visualizacion.py`](file:///c:/Dev/7-semestre/AI2/ProyectoGeneticosFuzzy/src/visualizacion/visualizacion.py) |

---

## 3. Resumen de Descubrimientos de PRISM (Fase 5)

* **Reglas Inducidas:** 16 reglas modulares en total.
  * **Calidad Alta (2 reglas):** Patrones con `alcohol = Alto`, `volatile acidity = Bajo`, `sulphates = Alto` (Precisión 65.2% - 71.7%).
  * **Calidad Baja (3 reglas):** Patrones con `alcohol = Bajo` (Precisión 71.2%, cobertura de 438 muestras), `volatile acidity = Alto` (Precisión 65.1%) y `sulphates = Bajo` (Precisión 82.6%).
  * **Calidad Media (11 reglas):** Patrones intermedios combinando `pH`, `sulphates`, `free sulfur dioxide` y `density` (Precisión 61.8% - 88.9%).
* **Métricas Globales de Reglas:**
  * **Precisión promedio de reglas:** 71.54%
  * **Longitud promedio de antecedentes:** 3.00 términos
* **Artefacto generado:** [`results/reglas_prism.csv`](file:///c:/Dev/7-semestre/AI2/ProyectoGeneticosFuzzy/results/reglas_prism.csv)

---

## 4. Resumen de Minería de Reglas con Apriori (Fase 6)

* **Muestras / Transacciones Analizadas:** 1279 transacciones de entrenamiento.
* **Itemsets Frecuentes Descubiertos:** 12,678 itemsets ($k=1 \dots 4$).
* **Filtros Estrictos Aplicados:**
  * Soporte $\ge 0.02$ (2%).
  * Confianza $\ge 0.60$ (60%).
  * Lift $> 1.20$.
  * Consecuente exclusivo: `calidad_categoria=Baja`, `calidad_categoria=Media` o `calidad_categoria=Alta`.
* **Reglas de Calidad Generadas:** 1,031 reglas en total.
  * **Calidad Baja:** 955 reglas (Confianza hasta 94.74%, Lift hasta 2.04).
  * **Calidad Media:** 75 reglas (Confianza hasta 77.50%, Lift hasta 1.94).
  * **Calidad Alta:** 1 regla destacada: `SI alcohol=Alto Y density=Bajo Y fixed acidity=Alto ENTONCES calidad_categoria=Alta` (Confianza 62.79%, Lift 4.62).
* **Artefacto generado:** [`results/reglas_apriori.csv`](file:///c:/Dev/7-semestre/AI2/ProyectoGeneticosFuzzy/results/reglas_apriori.csv)

---

## 5. Resumen de Integración y Modelo Base Discreto (Fase 7)

* **Reglas Seleccionadas para la Base Integrada:** **26 reglas** en total.
  * **16 reglas de PRISM** (núcleo predictivo directo).
  * **10 reglas complementarias de Apriori** (filtradas por mayor Lift y Confianza, no redundantes).
  * **Distribución por clase:** Media (16), Baja (8), Alta (2).
* **Métricas del Modelo Base Discreto (Baseline Inicial):**
  * **Exactitud en Entrenamiento (Train):** 63.41%
  * **Exactitud en Prueba (Test):** 57.50%
  * **F1-Macro en Train:** 0.5704 | **F1-Macro en Test:** 0.4674
  * **Cobertura de Reglas en Test:** 79.69%
* **Artefactos generados:**
  * [`results/reglas_base_integrada.csv`](file:///c:/Dev/7-semestre/AI2/ProyectoGeneticosFuzzy/results/reglas_base_integrada.csv)
  * [`results/reglas_base_integrada.json`](file:///c:/Dev/7-semestre/AI2/ProyectoGeneticosFuzzy/results/reglas_base_integrada.json)

---

## 6. Validación de Funciones Difusas e Inferencia Mamdani (Fases 8 y 9)

* **Variables Fuzzificadas:** 11 variables fisicoquímicas continuas con funciones híbridas trapezoidales-triangulares.
* **Forma de MFs:** Hombros abiertos en extremos para asegurar $\mu=1.0$ en mínimos/máximos y triángulo central para 'Medio'.
* **Smoke Test & Cobertura:** 1,599 registros evaluados. Cobertura difusa continua $\ge 98\%$.
* **Motor de Inferencia Difusa Mamdani (`SistemaDifusoMamdani`):**
  * Fuzzificación analítica vectorizada: $\alpha_k = w_k \cdot \min_{(v, l) \in Ant} \mu_{v, l}(x)$.
  * Defuzzificación continua por Centroide cuantitativo (COG) con fallback neutro a 5.80 ('Media').
  * Umbralización lingüística: Centroide $< 5.30 \rightarrow \text{'Baja'}$, $\ge 6.60 \rightarrow \text{'Alta'}$, resto $\rightarrow \text{'Media'}$.
* **Métricas del Sistema Difuso Inicial (Pre-AG Mamdani):**
  * **Exactitud en Train:** 58.64% | **Exactitud en Test:** 55.94%
  * **F1-Macro en Train:** 0.5283 | **F1-Macro en Test:** 0.4804
  * **Cobertura de Reglas en Test:** 98.44% (315 de 320 muestras cubiertas por disparo difuso)

---

## 7. Optimización Evolutiva con Algoritmo Genético (Fases 10 y 11)

* **Framework:** DEAP + NumPy vectorizado.
* **Estructura del Cromosoma Mixto (59 genes):**
  * **Parte 1 (33 genes reales):** 11 variables $\times$ 3 puntos de corte ($a_i, b_i, c_i$) con restricción $min_i \le a_i < b_i < c_i \le max_i$.
  * **Parte 2 (26 genes reales en $[0.0, 1.0]$):** Pesos y modulación de activación de las 26 reglas unificadas.
* **Evolución del Fitness (30 Generaciones, Población 50):**
  * **Generación 0 (Semilla canónica):** Fitness = `0.4892` (F1-Macro: 0.5560, Cobertura: 100%, Reglas Activas: 26).
  * **Generación 10:** Fitness = `0.5289` (F1-Macro: 0.5876, Cobertura: 99.14%, Reglas Activas: 21).
  * **Generación 20:** Fitness = `0.5538` (F1-Macro: 0.6061, Cobertura: 99.37%, Reglas Activas: 18).
  * **Generación 30 (Convergencia final):** Fitness = `0.5619` (F1-Macro: 0.6072, Cobertura: 99.22%, Reglas Activas: 16).
* **Poda Automática y Parquedad:** El AG desactivó 10 reglas redundantes o contradictorias ($w_k < 0.05$), reduciendo la complejidad del sistema en un 38.5% manteniendo una cobertura superior al 99%.
* **Artefactos generados:**
  * [`results/params_mfs_optimizadas_ga.json`](file:///c:/Dev/7-semestre/AI2/ProyectoGeneticosFuzzy/results/params_mfs_optimizadas_ga.json)
  * [`results/reglas_optimizadas_ga.json`](file:///c:/Dev/7-semestre/AI2/ProyectoGeneticosFuzzy/results/reglas_optimizadas_ga.json)
  * [`results/historial_convergencia_ga.csv`](file:///c:/Dev/7-semestre/AI2/ProyectoGeneticosFuzzy/results/historial_convergencia_ga.csv)

---

## 8. Evaluación Comparativa Rigurosa de las 3 Fases (Fase 12)

La evaluación en el conjunto independiente de prueba (**Test Set, N=320**) demuestra la superioridad progresiva del enfoque híbrido:

| Métrica de Evaluación | 1. Baseline Discreto (PRISM + Apriori) | 2. Pre-AG Difuso (Mamdani Canónico) | 3. Post-AG Difuso (Optimizado con AG) | Ganancia vs Baseline | Ganancia vs Pre-AG |
|---|:---:|:---:|:---:|:---:|:---:|
| **Exactitud (Accuracy Test)** | 57.50% | 55.94% | **60.31%** | **+2.81 pts** | **+4.37 pts** |
| **F1-Score Macro (Test)** | 0.4674 | 0.4804 | **0.5647** | **+0.0973 (+9.73 pts)** | **+0.0843 (+8.43 pts)** |
| **F1-Score Weighted (Test)** | 0.5236 | 0.5263 | **0.6059** | **+8.23 pts** | **+7.96 pts** |
| **Cobertura de Muestras (Test)** | 79.69% | 98.44% | **99.38%** | **+19.69 pts** | **+0.94 pts** |
| **Recall en Clase 'Alta'** | 11.63% (5/43) | 34.88% (15/43) | **44.19% (19/43)** | **+32.56 pts** | **+9.31 pts** |
| **Reglas Activas Utilizadas** | 26 reglas | 26 reglas | **16 reglas (38.5% podadas)** | **Mayor parsimonia** | **Mayor parsimonia** |

---

## 9. Visualizaciones y Orquestación Final (Fase 13)

* **Generación Automatizada de Figuras:** Cuatro gráficos de alta resolución (`dpi=300`) guardados en [`results/plots/`](file:///c:/Dev/7-semestre/AI2/ProyectoGeneticosFuzzy/results/plots):
  1. `convergencia_fitness_ga.png`: Evolución de Fitness, F1-Macro, Cobertura y Reglas Activas.
  2. `comparativa_mfs_pre_post_ga.png`: Desplazamiento de funciones de pertenencia antes vs después del AG.
  3. `matrices_confusion_comparativas.png`: Comparación lado a lado de las matrices de confusión para las 3 fases.
  4. `evolucion_reglas.png`: Distribución de pesos $w_k$, reglas podadas y análisis conceptual.
* **Pipeline Maestro (`src/main.py`):**
  * Permite ejecución completa (`python src/main.py --all`) o por fases (`python src/main.py --fase <nombre>`).
  * Soporta banderas de optimización como `--skip-ga`, `--generaciones` y `--poblacion`.
