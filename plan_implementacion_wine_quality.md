# Plan de implementación — Clasificación de calidad de vinos con PRISM, Apriori, lógica difusa y algoritmo genético

## 1. Objetivo del proyecto

Equipo, el objetivo es desarrollar un sistema capaz de **determinar la calidad de un vino a partir de sus características fisicoquímicas** y, además, descubrir qué combinaciones de características están relacionadas con vinos de calidad baja, media o alta.

El sistema seguirá este flujo:

**Dataset → Preparación → PRISM + Apriori → Base de reglas → Lógica difusa → Algoritmo genético → Clasificación final**

El resultado final debe ser interpretable. Para cada vino nuevo, el sistema debe indicar:

- Calidad **baja**
- Calidad **media**
- Calidad **alta**

Además, debe permitir identificar las reglas que llevaron a esa clasificación.

---

## 2. Dataset

Trabajaremos con el dataset **Wine Quality – Red Wine**, específicamente con el archivo:

`winequality-red.csv`

El dataset contiene aproximadamente **1.599 registros y 12 columnas**.

Las variables disponibles son:

| Variable | Descripción | Tipo |
|---|---|---|
| `fixed acidity` | Acidez fija | Numérica |
| `volatile acidity` | Acidez volátil | Numérica |
| `citric acid` | Ácido cítrico | Numérica |
| `residual sugar` | Azúcar residual | Numérica |
| `chlorides` | Cloruros | Numérica |
| `free sulfur dioxide` | Dióxido de azufre libre | Numérica |
| `total sulfur dioxide` | Dióxido de azufre total | Numérica |
| `density` | Densidad | Numérica |
| `pH` | Nivel de pH | Numérica |
| `sulphates` | Sulfatos | Numérica |
| `alcohol` | Porcentaje de alcohol | Numérica |
| `quality` | Calidad del vino | Numérica/discreta |

La variable `quality` será transformada en tres categorías:

- **3–5 → Baja**
- **6 → Media**
- **7–8 → Alta**

Antes de fijar definitivamente esta agrupación, debemos verificar la distribución real de las clases para comprobar que no exista un desequilibrio excesivo.

---

## 3. Preparación de los datos

El equipo debe realizar las siguientes tareas:

1. Cargar el archivo CSV.
2. Verificar la cantidad de registros y variables.
3. Comprobar valores faltantes.
4. Comprobar registros duplicados.
5. Revisar valores anómalos o inconsistentes.
6. Analizar la distribución de las variables numéricas.
7. Separar las características del resultado `quality`.
8. Crear la variable categórica de calidad: baja, media y alta.
9. Dividir los datos en entrenamiento y prueba.
10. Crear una versión discretizada de las variables para PRISM y Apriori.
11. Mantener también los valores numéricos originales para la lógica difusa.

### Importante

Debemos conservar **dos representaciones de los datos**:

- **Datos numéricos originales:** se utilizarán principalmente en la lógica difusa.
- **Datos discretizados:** se utilizarán para descubrir reglas con PRISM y Apriori.

Esto evita perder información numérica antes de aplicar la lógica difusa.

---

## 4. PRISM: descubrimiento de reglas de clasificación

PRISM se utilizará para descubrir **reglas que permitan clasificar la calidad del vino**.

Su objetivo será encontrar condiciones que permitan identificar vinos de calidad baja, media o alta.

Ejemplo ilustrativo:

```text
SI alcohol = alto
Y volatile acidity = baja
ENTONCES calidad = alta
```

Este ejemplo es únicamente ilustrativo. Las reglas definitivas deben ser descubiertas a partir de los datos.

### Para cada regla obtenida debemos registrar:

- Condiciones utilizadas.
- Clase de calidad que predice.
- Precisión.
- Cobertura.
- Cantidad de registros cubiertos.
- Cantidad de condiciones de la regla.

### Resultado esperado de PRISM

Obtener un conjunto de reglas de clasificación que expliquen qué características están asociadas con cada nivel de calidad.

---

## 5. Apriori: descubrimiento de asociaciones

Apriori tendrá un objetivo diferente a PRISM.

Mientras PRISM buscará principalmente **reglas para clasificar la calidad**, Apriori buscará **combinaciones frecuentes de características**.

Para utilizar Apriori será necesario discretizar las variables numéricas.

Por ejemplo:

```text
alcohol = bajo
alcohol = medio
alcohol = alto

volatile acidity = baja
volatile acidity = media
volatile acidity = alta

sulphates = bajos
sulphates = medios
sulphates = altos
```

Después se buscarán asociaciones frecuentes entre estas categorías.

Ejemplo ilustrativo:

```text
alcohol = alto
+
volatile acidity = baja
+
sulphates = altos
→
quality = alta
```

Nuevamente, esta regla es solamente un ejemplo. Las reglas reales serán generadas por Apriori.

### Métricas principales

Las reglas de Apriori deben evaluarse utilizando:

- **Support:** qué tan frecuente es la combinación.
- **Confidence:** qué tan frecuentemente se cumple la consecuencia cuando se cumplen las condiciones.
- **Lift:** qué tan fuerte es la asociación respecto a una ocurrencia aleatoria.

---

## 6. Integración de PRISM y Apriori

Después de ejecutar ambos algoritmos, tendremos dos conjuntos de conocimiento:

### PRISM

Reglas orientadas a la **clasificación**.

```text
Condiciones → Calidad
```

### Apriori

Reglas orientadas a encontrar **asociaciones frecuentes**.

```text
Condiciones frecuentes → Asociación
```

El equipo debe combinar ambos resultados en una **base de reglas del sistema**.

No debemos utilizar automáticamente todas las reglas obtenidas.

Se deben seleccionar las reglas más útiles considerando:

- Precisión.
- Cobertura.
- Confianza.
- Lift.
- Simplicidad.
- Cantidad de condiciones.
- Relevancia para la clasificación de calidad.

La finalidad es evitar una base de reglas demasiado grande, repetitiva o difícil de interpretar.

---

## 7. Lógica difusa

La lógica difusa se utilizará para evitar que el sistema dependa de límites completamente rígidos.

Por ejemplo, un sistema tradicional podría establecer:

```text
SI alcohol >= 12
ENTONCES alcohol = alto
```

El problema es que un vino con `11.99` quedaría fuera de la categoría alta y uno con `12.00` entraría inmediatamente.

La lógica difusa permite representar que un valor puede pertenecer parcialmente a varias categorías.

Por ejemplo:

```text
Alcohol = 11.8

Bajo  → 0.00
Medio → 0.35
Alto  → 0.65
```

Los valores anteriores son únicamente ilustrativos.

De esta forma, el sistema puede trabajar con conceptos como:

- Bajo.
- Medio.
- Alto.

En lugar de depender exclusivamente de límites rígidos.

---

## 8. Variables que pueden utilizar lógica difusa

No es necesario convertir las 11 variables en variables difusas.

Inicialmente se deben analizar los resultados de PRISM, Apriori y el análisis estadístico para seleccionar las variables más relevantes.

Como candidatos iniciales podemos considerar:

- `alcohol`
- `volatile acidity`
- `fixed acidity`
- `pH`
- `sulphates`
- `density`

La selección definitiva debe basarse en los resultados obtenidos durante el análisis.

---

## 9. Funciones de pertenencia

Para cada variable seleccionada se definirán conjuntos difusos.

Por ejemplo, para `alcohol`:

```text
Alcohol
├── Bajo
├── Medio
└── Alto
```

Para `volatile acidity`:

```text
Acidez volátil
├── Baja
├── Media
└── Alta
```

Las funciones de pertenencia pueden comenzar con funciones triangulares o trapezoidales.

Ejemplo conceptual:

```text
              Medio
             /-----\
            /       \
Bajo ______/         \______ Alto
```

Los parámetros exactos de estas funciones no deben fijarse arbitrariamente. Se establecerán inicialmente a partir de los datos y posteriormente serán optimizados por el algoritmo genético.

---

## 10. Conversión de las reglas a reglas difusas

Las reglas obtenidas con PRISM y Apriori deben transformarse en reglas que puedan trabajar con conceptos lingüísticos.

Ejemplo:

### Regla original

```text
SI alcohol > 11
Y volatile acidity < 0.6
ENTONCES calidad = alta
```

### Regla difusa

```text
SI alcohol ES alto
Y acidez volátil ES baja
ENTONCES calidad ES alta
```

La ventaja es que la regla deja de depender de un único límite exacto y puede trabajar con grados de pertenencia.

---

## 11. Base de reglas difusas

La base final debe contener reglas como:

```text
REGLA 1:
SI alcohol ES alto
Y acidez volátil ES baja
ENTONCES calidad ES alta
```

```text
REGLA 2:
SI alcohol ES medio
Y acidez volátil ES media
Y sulfatos ES medios
ENTONCES calidad ES media
```

```text
REGLA 3:
SI alcohol ES bajo
Y acidez volátil ES alta
ENTONCES calidad ES baja
```

Estas reglas son ejemplos de estructura y no deben considerarse resultados reales del dataset.

---

## 12. Algoritmo genético

El algoritmo genético se utilizará para **optimizar el sistema difuso**.

No se utilizará como un clasificador independiente.

Su función será buscar una configuración de parámetros que permita mejorar el funcionamiento de las reglas difusas.

Principalmente puede optimizar:

- Límites de las funciones de pertenencia.
- Posición de los conjuntos Bajo, Medio y Alto.
- Parámetros de las reglas.
- Selección o eliminación de reglas poco útiles.

---

## 13. Representación de un individuo

Cada individuo del algoritmo genético representará una posible configuración del sistema.

Por ejemplo, una parte de un cromosoma podría representar parámetros relacionados con `alcohol`:

```text
[10.5, 12.0, 14.0]
```

Estos valores son solamente ilustrativos.

La representación real dependerá de las funciones de pertenencia que implementemos.

Un individuo completo podría representar parámetros de varias variables:

```text
[parámetros alcohol,
 parámetros acidez volátil,
 parámetros pH,
 parámetros sulfatos,
 ...]
```

---

## 14. Función de evaluación

Cada configuración generada por el algoritmo genético debe ser evaluada mediante una función de aptitud.

La función debe considerar principalmente:

- Calidad de las predicciones.
- Cobertura de los datos.
- Complejidad del sistema.
- Cantidad de reglas.

Una propuesta inicial es:

```text
Fitness =
    Precisión
    + Cobertura
    - Penalización por complejidad
```

La fórmula definitiva debe ser definida y justificada durante la implementación experimental.

La penalización es importante porque no buscamos únicamente aumentar la precisión. También queremos mantener un sistema comprensible y con una cantidad razonable de reglas.

---

## 15. Funcionamiento del algoritmo genético

El proceso será:

```text
1. Crear población inicial
        ↓
2. Evaluar cada individuo
        ↓
3. Seleccionar los mejores individuos
        ↓
4. Aplicar cruce
        ↓
5. Aplicar mutación
        ↓
6. Crear nueva generación
        ↓
7. Evaluar nuevamente
        ↓
8. Repetir hasta cumplir el criterio de parada
        ↓
9. Seleccionar la mejor configuración
```

El criterio de parada puede ser:

- Número máximo de generaciones.
- Ausencia de mejora durante varias generaciones.
- Alcanzar una determinada calidad de evaluación.

---

## 16. Modelo final

Una vez terminadas todas las etapas, el sistema deberá funcionar de la siguiente manera:

```text
NUEVO VINO
    ↓
Características fisicoquímicas
    ↓
Transformación a variables difusas
    ↓
Evaluación de reglas
    ↓
Reglas optimizadas por algoritmo genético
    ↓
Clasificación
    ↓
BAJA / MEDIA / ALTA
```

Por ejemplo:

```text
Entrada:

alcohol = 12.4
volatile acidity = 0.42
sulphates = 0.72
pH = 3.2
...

↓

Sistema difuso

alcohol → alto
acidez volátil → baja
sulfatos → medio/alto

↓

Reglas activadas

Regla 1 → grado de activación: ...
Regla 2 → grado de activación: ...

↓

Resultado:

CALIDAD = ALTA
```

Los valores y reglas mostrados son únicamente ejemplos de funcionamiento.

---

## 17. Evaluación del proyecto

El sistema debe evaluarse por etapas para demostrar qué aporta cada técnica.

Se propone comparar:

| Etapa | Objetivo |
|---|---|
| Modelo base | Tener una referencia inicial |
| PRISM | Evaluar reglas de clasificación |
| PRISM + Apriori | Incorporar asociaciones |
| Reglas + lógica difusa | Evaluar flexibilidad de las reglas |
| Lógica difusa + algoritmo genético | Evaluar la optimización final |

### Métricas de clasificación

Utilizar:

- Accuracy.
- Precision.
- Recall.
- F1-score.
- Matriz de confusión.

### Métricas de reglas

Utilizar:

- Support.
- Confidence.
- Lift.
- Cobertura.
- Cantidad de reglas.
- Complejidad promedio de las reglas.

La evaluación debe realizarse sobre datos de prueba que no hayan sido utilizados para construir las reglas o ajustar los parámetros.

---

## 18. Representación visual del proyecto

La presentación debe mostrar claramente qué aporta cada algoritmo.

### Flujo general

```text
┌─────────────────────┐
│  🍷 Dataset Wine    │
│      Quality        │
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│ Preparación de datos│
│ Limpieza + análisis │
└──────────┬──────────┘
           ↓
      ┌────┴────┐
      ↓         ↓
┌──────────┐ ┌──────────┐
│  PRISM   │ │  Apriori │
│Clasifica │ │Asocia    │
└────┬─────┘ └────┬─────┘
     │             │
     └──────┬──────┘
            ↓
┌─────────────────────┐
│   Base de reglas    │
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│   Lógica difusa     │
│ Bajo / Medio / Alto │
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│ Algoritmo genético  │
│    Optimización     │
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│ Clasificación final │
│ Baja / Media / Alta │
└─────────────────────┘
```

---

## 19. Visualización de la evolución de una regla

Una parte importante de la presentación será mostrar cómo una regla evoluciona durante el proyecto.

### Paso 1 — Condición simple

```text
alcohol > 11
        ↓
calidad alta
```

### Paso 2 — Regla descubierta

```text
alcohol > 11
+
acidez volátil < 0.6
        ↓
calidad alta
```

### Paso 3 — Regla difusa

```text
alcohol ES alto
+
acidez volátil ES baja
        ↓
calidad ES alta
```

### Paso 4 — Regla optimizada

```text
alcohol ES alto
+
acidez volátil ES baja
+
parámetros optimizados
        ↓
calidad ES alta
```

La idea visual es demostrar que cada técnica agrega una función diferente:

```text
PRISM
  ↓
Descubre reglas de clasificación

Apriori
  ↓
Descubre asociaciones frecuentes

Lógica difusa
  ↓
Hace las reglas más flexibles

Algoritmo genético
  ↓
Optimiza los parámetros del sistema
```

---

## 20. Estructura recomendada del código

Para mantener el proyecto organizado, se recomienda separar cada etapa:

```text
proyecto/
│
├── data/
│   └── winequality-red.csv
│
├── src/
│   ├── carga_datos.py
│   ├── preparacion.py
│   ├── prism.py
│   ├── apriori.py
│   ├── reglas.py
│   ├── fuzzy.py
│   ├── genetico.py
│   ├── evaluacion.py
│   └── main.py
│
├── notebooks/
│   └── exploracion.ipynb
│
├── results/
│   ├── reglas_prism.csv
│   ├── reglas_apriori.csv
│   ├── reglas_fuzzy.csv
│   └── resultados_evaluacion.csv
│
└── README.md
```

La separación por módulos permitirá probar cada algoritmo individualmente y comparar sus resultados.

---

## 21. Orden de implementación

El equipo debe implementar el proyecto en este orden:

```text
FASE 1
Carga y exploración del dataset
        ↓
FASE 2
Limpieza y preparación
        ↓
FASE 3
Creación de las categorías de calidad
        ↓
FASE 4
Discretización de variables
        ↓
FASE 5
Implementación de PRISM
        ↓
FASE 6
Implementación de Apriori
        ↓
FASE 7
Selección e integración de reglas
        ↓
FASE 8
Diseño de variables difusas
        ↓
FASE 9
Implementación del sistema difuso
        ↓
FASE 10
Implementación del algoritmo genético
        ↓
FASE 11
Optimización de las funciones de pertenencia
        ↓
FASE 12
Evaluación completa
        ↓
FASE 13
Visualización y documentación
```

---

## 22. Resultado esperado

Al finalizar, tendremos un sistema que no solamente diga:

```text
Este vino es de calidad alta.
```

Sino que también permita explicar **por qué**:

```text
CALIDAD: ALTA

Reglas activadas:
- Alcohol alto
- Acidez volátil baja
- Sulfatos medio/alto

La combinación de estas características activó
principalmente las reglas asociadas con calidad alta.
```

De esta manera, el proyecto no se limitará a realizar una clasificación, sino que permitirá **descubrir, representar y optimizar reglas interpretables** utilizando PRISM, Apriori, lógica difusa y algoritmos genéticos.

---

## 23. Objetivo técnico final

La implementación completa debe conseguir que las cuatro técnicas tengan una función clara y diferenciada:

| Técnica | Función dentro del proyecto |
|---|---|
| **PRISM** | Descubrir reglas de clasificación de calidad |
| **Apriori** | Descubrir asociaciones frecuentes entre características |
| **Lógica difusa** | Representar conceptos como bajo, medio y alto de forma flexible |
| **Algoritmo genético** | Optimizar los parámetros y configuración del sistema difuso |

El resultado será un **sistema híbrido e interpretable para la clasificación de calidad del vino**, donde cada algoritmo aporta una etapa concreta del proceso.
