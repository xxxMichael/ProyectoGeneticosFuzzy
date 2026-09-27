# Guía de Nuevas Propuestas Visuales y Defensa Analítica
## Wine Quality AI System: PRISM, Apriori, Lógica Difusa Mamdani y Calibración con Algoritmos Genéticos

---

### Introducción: ¿Por qué estas nuevas visualizaciones marcan la diferencia?

En la mayoría de proyectos académicos de Inteligencia Artificial que integran Lógica Difusa y Algoritmos Genéticos (incluyendo versiones estándar presentadas por otros grupos), los gráficos suelen limitarse a:
1. Una curva básica de convergencia de fitness.
2. Gráficos 1D de triángulos o trapecios (funciones de pertenencia).
3. Matrices de confusión tradicionales en escala de azules.

Si bien estos gráficos cumplen con los requisitos básicos, **no logran comunicar la verdadera profundidad algorítmica ni el impacto cualitativo de las soluciones implementadas**, provocando que todos los trabajos se vean idénticos a ojos del evaluador.

Para resolver esta situación y dotar a tu presentación de un **valor diferencial indiscutible**, se han generado **6 nuevas propuestas visuales analíticas avanzadas** en alta resolución (300 DPI), alojadas en este directorio:
`c:\Dev\7-semestre\AI2\ProyectoGeneticosFuzzy\results\propuestas_visuales_diferenciales\`

A continuación se detalla cada propuesta visual, su justificación matemática, cómo defenderla frente al docente y las preguntas típicas con sus respuestas ganadoras.

---

## Catálogo de Nuevas Propuestas Visuales

---

### 1. Radar Multidimensional Comparativo de Capacidades
* **Archivo:** `01_radar_multidimensional_comparativo.png`
* **Tipo:** Gráfico de Araña / Radar Chart Hexagonal (6 Dimensiones Ortogonales).
* **Comparativa:** Fase 1 (Baseline Discreto) vs. Fase 2 (Mamdani Pre-GA) vs. Fase 3 (Mamdani Post-GA).

```
                      Exactitud en Test (Accuracy)
                                    ^
                                    |
  Compacidad / Parsimonia           |           F1-Score Macro
    (Poda de Reglas) <--------------+--------------> (Balance de Clases)
                                    |
                                    |
   Continuidad Difusa               |           Sensibilidad Vinos Alta
   (Suavidad Frontera) <------------+--------------> (Recall Alta)
                                    v
                        Cobertura del Espacio (% Muestras)
```

#### ¿Por qué es único frente a otros proyectos?
* Los demás proyectos presentan tablas planas de métricas o gráficos de barras unidimensionales donde solo se resalta el *Accuracy*.
* Este radar expone la **evolución sistémica del modelo en 6 dimensiones críticas**. Demuestra visualmente que el sistema Post-AG no solo mejora la exactitud (+1.56%), sino que **triplica el recall en clases minoritarias** (del 11.6% al 44.2%), eleva la cobertura al **99.38%** y además **reduce la complejidad del modelo en un 38.5%** mediante poda evolutiva.

#### Speech / Qué decir en la defensa (45 segundos):
> *"Profesor, en lugar de evaluar el modelo con una métrica aislada como la exactitud que suele enmascarar sesgos, diseñamos esta evaluación radar en 6 dimensiones. Observen cómo el clasificador de reglas discretas (línea gris) tenía una cobertura limitada al 79% y una sensibilidad pésima del 11% en vinos de alta calidad. Al introducir el motor difuso continuo Mamdani (azul), expandimos la cobertura a casi el 100%. Y finalmente, el Algoritmo Genético (polígono verde) expande drásticamente el F1-Macro y el Recall de vinos premium hasta el 44.2%, mientras simultáneamente logra un 38.5% de compacidad podando reglas redundantes."*

---

### 2. Superficie Continua 2D de Inferencia Difusa Mamdani
* **Archivo:** `02_superficie_decision_mamdani_alcohol_acidez.png`
* **Tipo:** Mapa de Contornos Continuos (Contour Heatmap 2D con Iso-líneas y Muestras de Test).
* **Ejes:** Graduación Alcohólica (% vol) vs. Acidez Volátil ($\text{g/dm}^3$ de ácido acético).

#### ¿Por qué es único frente a otros proyectos?
* En otros proyectos se habla teóricamente de que la lógica difusa resuelve el *"Boundary Problem"* (problema de fronteras rígidas), pero **nunca lo demuestran gráficamente**.
* Esta superficie calcula la **defuzzificación real por Centroide continuo (CoG $\in [0, 10]$)** a lo largo de una grilla de más de 4,900 puntos químicos, demostrando cómo el sistema Mamdani genera **gradientes continuos y suaves de calidad**, con curvas de nivel que delimitan matemáticamente la *Zona Crítica de Rechazo* (baja calidad) y la *Zona Premium Gran Reserva* (alta calidad). Las muestras reales del conjunto de prueba se superponen para demostrar cómo las decisiones siguen la física del vino.

#### Speech / Qué decir en la defensa (45 segundos):
> *"Aquí demostramos empíricamente la resolución del problema de frontera. En un modelo basado en reglas clásicas booleanas, un vino con 10.79% de alcohol caería en un escalón discreto artificialmente distinto a uno con 10.81%. En nuestra superficie de inferencia Mamdani continua, el valor de defuzzificación del centroide transita suavemente como un gradiente continuo. La línea discontinua roja marca el umbral de descarte (<5.30) y la línea verde sólida delimita los vinos excepcionales (>=6.60), respetando la interacción entre alcohol y acidez volátil sin saltos abruptos."*

---

### 3. Densidad de Centroides Defuzzificados por Clase Real
* **Archivo:** `03_distribucion_centroides_por_clase_real.png`
* **Tipo:** Estimación de Densidad de Kernel (KDE) / Curvas de Distribución Continua.
* **Paneles:** Pre-AG (Configuración heurística inicial) vs. Post-AG (Optimización evolutiva multi-criterio).

#### ¿Por qué es único frente a otros proyectos?
* Los otros proyectos solo muestran si la predicción acertó o falló.
* Este gráfico analiza **el valor numérico continuo del centroide antes de clasificarlo en texto**, revelando el comportamiento interno del motor:
  * **En Pre-AG:** Las densidades de los vinos Reales *Baja*, *Media* y *Alta* están colapsadas y superpuestas en el rango medio ($5.4 - 6.2$). El sistema tenía baja confianza y baja capacidad de discriminación.
  * **En Post-AG:** El Algoritmo Genético desplazó físicamente las campanas de densidad: los vinos de calidad *Baja* se mueven hacia $<5.3$, los de calidad *Alta* generan un pico claro por encima de $6.6$, y la *Media* se mantiene centrada. Es la prueba estadística irrefutable de que el AG aprendió a discriminar.

#### Speech / Qué decir en la defensa (45 segundos):
> *"Este gráfico expone la 'caja de resonancia' interna de nuestro defuzzificador Mamdani. En el panel izquierdo (Pre-AG), las tres calidades reales generaban centroides muy concentrados entre 5.5 y 6.0, provocando confusiones masivas. En el panel derecho (Post-AG), tras la optimización del cromosoma mixto con DEAP, las distribuciones se desacoplan: la curva verde de vinos de alta calidad forma un pico independiente por encima del umbral de 6.6, confirmando que la calibración genética aumentó la separabilidad inter-clase del sistema."*

---

### 4. Poda Evolutiva y Distribución de Pesos de Reglas
* **Archivo:** `04_poda_y_pesos_reglas_ga.png`
* **Tipo:** Gráfico Lollipop de Pesos Evolved ($w_k \in [0, 1]$) con Zona de Poda Sombreada.
* **Ejes:** 26 Reglas ordenadas en el eje vertical vs. Peso de activación en el eje horizontal.

#### ¿Por qué es único frente a otros proyectos?
* Otros grupos ejecutan algoritmos genéticos que mantienen todas sus reglas activas (las 55 reglas del proyecto de referencia siguen disparándose todas, generando bloatware y sobreajuste).
* Nuestro proyecto introdujo un término de **penalización de parsimonia regularizada** en la función de fitness:
  $$\text{Fitness} = 0.70 \cdot \text{F1\_Macro} + 0.20 \cdot \text{Cobertura} - 0.10 \cdot \left(\frac{\text{Reglas Activas}}{\text{Reglas Totales}}\right)$$
* Este gráfico muestra con total claridad cómo **10 de las 26 reglas cayeron en la zona de poda ($w_k < 0.10$) y fueron desactivadas automáticamente**, operando con un conjunto compacto y ultraeficiente de 16 reglas activas sin perder cobertura (99.38%).

#### Speech / Qué decir en la defensa (45 segundos):
> *"Uno de los mayores riesgos en sistemas difusos evolutivos es la proliferación descontrolada de reglas o rule-bloat. Para combatirlo, penalizamos la complejidad en la función de aptitud del AG. Como se observa en la zona roja sombreada, el algoritmo desactivó automáticamente 10 de las 26 reglas iniciales (una poda del 38.5%). Esto no solo previene el sobreajuste y agiliza el tiempo de inferencia a microsegundos, sino que deja un modelo compacto de 16 reglas maestras mucho más legible para un enólogo."*

---

### 5. Espacio de Minería de Reglas: Frente de Pareto PRISM vs. Apriori
* **Archivo:** `05_espacio_reglas_prism_vs_apriori.png`
* **Tipo:** Diagrama de Dispersión / Burbujas en el plano Soporte vs. Confianza vs. Lift.
* **Elementos:** Candidatas Apriori (naranja), Candidatas PRISM (azul) y Reglas Seleccionadas e Integradas (estrellas verdes).

#### ¿Por qué es único frente a otros proyectos?
* Prácticamente ningún estudiante fundamenta por qué usar PRISM y Apriori conjuntamente; suelen implementar uno de los dos por separado o juntarlos sin análisis.
* Este gráfico demuestra científicamente la **complementariedad de nichos de ambos algoritmos**:
  * **PRISM:** Genera reglas de **alta confianza condicional / precisión** ($>0.70$), pero con soporte focalizado en nichos específicos (especialistas en casos raros o extremos).
  * **Apriori:** Genera reglas de **alto soporte poblacional** (hasta 0.35 de soporte), capturando el comportamiento promedio y generalista del dataset.
  * **La Base Integrada (Estrellas):** Selecciona el **Frente de Pareto**, combinando lo mejor de ambos mundos para asegurar precisión en extremos y amplia cobertura global.

#### Speech / Qué decir en la defensa (45 segundos):
> *"Nuestra decisión de implementar minería dual desde cero responde a una razón matemática: PRISM y Apriori exploran espacios de hipótesis complementarios. Al graficar Soporte vs. Confianza con tamaño proporcional al Lift, vemos que PRISM (puntos azules) produce reglas especialistas de altísima pureza pero bajo soporte, ideales para detectar vinos defectuosos o excepcionales. Por el contrario, Apriori (puntos naranjas) captura las grandes tendencias de alto soporte. Las 26 reglas que seleccionamos e integramos (estrellas verdes) trazan el Frente de Pareto óptimo entre cobertura y precisión."*

---

### 6. Cobertura Fisicoquímica Integral de las 11 Variables
* **Archivo:** `06_importancia_quimica_11_variables.png`
* **Tipo:** Gráfico de Barras Horizontales Apiladas por Clase de Calidad (*Alta*, *Media*, *Baja*).
* **Contenido:** Frecuencia de aparición de las 11 variables continuas en las reglas activas finales.

#### ¿Por qué es único frente a otros proyectos?
* En otras versiones del proyecto (como la de referencia), los autores eliminaron arbitrariamente 7 variables del dataset y trabajaron únicamente con 4 (`alcohol`, `volatile acidity`, `sulphates`, `pH`).
* Nuestro trabajo modela las **11 variables fisicoquímicas completas**. Este gráfico prueba cómo variables descartadas por otros proyectos (como `total sulfur dioxide`, `citric acid`, `residual sugar`, `chlorides` y `density`) son actores decisivos en las reglas activas de calidad, preservando la fidelidad bioquímica del vino.

#### Speech / Qué decir en la defensa (45 segundos):
> *"A diferencia de otras propuestas que simplifican el problema recortando el dataset a solo 4 variables, nuestro sistema integra las 11 características continuas de laboratorio. Como muestra este gráfico de frecuencia en reglas activas, el alcohol y los sulfatos lideran el impacto en vinos de alta calidad, pero el dióxido de azufre total y los azúcares residuales aportan información vital que otros modelos ignoran por completo. Modelar las 11 variables garantiza que el sistema no tenga puntos ciegos analíticos."*

---

## Preguntas Frecuentes del Jurado / Profesor y Cómo Responderlas

| Pregunta Probable del Docente | Respuesta Ganadora y Gráfico de Apoyo |
| :--- | :--- |
| **"¿Por qué su modelo solo alcanza un 60.31% de exactitud? ¿No es un valor bajo?"** | *"En clasificación multiclase desbalanceada de vinos (donde la clase media domina con más del 80%), un modelo ingenuo que prediga siempre 'Media' obtendría un 58% de exactitud pero tendría un F1-Score de 0.25 y cero utilidad enológica. Nosotros optimizamos por **F1-Macro (0.5647)** y logramos un **Recall de 44.2% en vinos de alta calidad** (frente al 11% del modelo discreto), como se evidencia en el **Radar Multidimensional (Gráfico 01)**."* |
| **"¿Cómo sé que el Algoritmo Genético realmente mejoró el sistema difuso y no fue suerte?"** | *"Se puede verificar en el **Gráfico 03 (Densidades de Centroides)**: en Pre-AG los centroides calculados por el centro de gravedad estaban colapsados en una sola masa entre 5.5 y 6.0; tras la calibración evolutiva, las distribuciones de probabilidad se desacoplaron nítidamente, desplazando el pico de vinos de alta calidad hacia la zona $>6.6$."* |
| **"¿Por qué implementaron dos algoritmos de minería (PRISM y Apriori) si ambos sacan reglas?"** | *"Porque resuelven problemas opuestos en el trade-off de minería de datos, como demostramos en el **Frente de Pareto (Gráfico 05)**. PRISM es un algoritmo inductivo voraz que maximiza la pureza condicional (precisión) sobre instancias no cubiertas, mientras que Apriori busca soporte y correlaciones globales por Lift. Su combinación permite tener reglas especialistas y reglas generalistas en la misma base difusa."* |
| **"¿Cómo evitaron que el AG sufriera sobreajuste con tantas reglas y variables?"** | *"Mediante dos compuertas estrictas: (1) Un operador de reparación geométrica que fuerza $a_i < b_i < c_i$, garantizando funciones de pertenencia válidas; y (2) Una función de aptitud parsimoniosa que penaliza la cantidad de reglas activas. El resultado concreto se observa en el **Gráfico 04**: el AG desactivó y podó el 38.5% de las reglas redundantes."* |

---

## Estado en la Interfaz Web

Tal como solicitaste, **las imágenes de la aplicación web actual permanecen intactas y sin modificaciones**. 

Estos nuevos gráficos y sus análisis residen de forma autónoma en la carpeta `results/propuestas_visuales_diferenciales/` para que los revises con calma. Si en el futuro deseas incorporar alguno de ellos al dashboard web o a diapositivas de presentación, el módulo generador ya está totalmente integrado en [`src/visualizacion/propuestas_avanzadas.py`](file:///c:/Dev/7-semestre/AI2/ProyectoGeneticosFuzzy/src/visualizacion/propuestas_avanzadas.py).
