# Notas de Aplicación Práctica: Árboles de Decisión y Algoritmos de Ensamble

Este documento recopila la metodología técnica, fundamentos matemáticos y buenas prácticas aplicadas durante el desarrollo de modelos predictivos basados en árboles (Decision Trees, Random Forest y XGBoost). Sirve como guía de consulta técnica para proyectos de regresión tabular.

---

## 1. Preparación del Entorno y Datos

### Conceptos Clave
- **Reproducibilidad:** Definición de semillas aleatorias fijas (`SEED = 1`) a nivel global en librerías (`numpy`, `random_state` de `scikit-learn` y XGBoost).
- **Tratamiento de Valores Faltantes (Imputación):** En modelos basados en árboles de decisión estándar de `scikit-learn`, los valores nulos (`NaN`) deben ser tratados antes de ajustar el modelo. Imputar con `0` convierte los valores faltantes en un nivel o valor escalar específico sin romper la estructura matricial.

### División del Dataset (Split Directo)
Para evitar el fuga de información (*data leakage*), el dataset se divide secuencialmente en tres subconjuntos independientes:
1. **Entrenamiento (Train - 60%):** Utilizado para ajustar los parámetros de los modelos.
2. **Validación (Val - 20%):** Utilizado para el ajuste de hiperparámetros (`max_depth`, `n_estimators`, `eta`).
3. **Prueba (Test - 20%):** Reservado exclusivamente para la evaluación final del modelo seleccionado.

---

## 2. Vectorización y Codificación de Variables Categoricas

### `DictVectorizer`
A diferencia de transformadores como `OneHotEncoder` aplicados directamente sobre DataFrames, `DictVectorizer` opera sobre listas de diccionarios Python (`df.to_dict(orient='records')`).

- **Ventajas:** Maneja automáticamente variables numéricas y categóricas. Las columnas numéricas se conservan sin alteración, mientras que las categóricas se transforman mediante codificación One-Hot.
- **Formato Disperso (`sparse=True`):** Optimiza el uso de memoria RAM al almacenar matrices dispersas (Scipy CSR matrix), ideal para datasets con alta cardinalidad categórica.

---

## 3. Árboles de Decisión (Decision Tree Regressor)

### Fundamento
Un árbol de decisión divide iterativamente el espacio de características en regiones ortogonales. La métrica habitual para problemas de regresión es el **Error Cuadrático Medio (MSE)** o la **Varianza de la Variable Objetivo**.

$$\text{MSE} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y})^2$$

### Interpretación del primer Split (`max_depth=1`)
- Evaluar un árbol con profundidad 1 (Stump) permite identificar cuál es la variable con mayor poder de separación inicial (la que logra la mayor reducción de varianza en el primer nodo).
- **Punto de decisión:** Permite diagnosticar qué variable tiene una correlación no lineal primaria con el objetivo.

---

## 4. Bosques Aleatorios (Random Forest Regressor)

### Fundamento de Bagging (Bootstrap Aggregating)
Random Forest es un ensamble por promediado. Entrena múltiples árboles de decisión independientes construidos sobre muestras *bootstrap* del dataset original y subconjuntos aleatorios de variables.

$$\hat{y}_{\text{RF}} = \frac{1}{B} \sum_{b=1}^{B} T_b(x)$$

### Evaluación de Hiperparámetros
1. **`n_estimators` (Número de Árboles):**
   - Incrementar el número de árboles reduce la varianza general del ensamble sin aumentar el sobreajuste (*overfitting*).
   - Existe un punto de retornos decrecientes donde el beneficio marginal de añadir más árboles no justifica el costo computacional adicional.

2. **`max_depth` (Profundidad Máxima por Árbol):**
   - Controla la complejidad de los árboles individuales.
   - Una profundidad muy alta permite que los árboles individuales aprendan el ruido del conjunto de entrenamiento.
   - Restringir `max_depth` actua como una técnica eficaz de regularización.

---

## 5. Importancia de Variables (Feature Importance)

### Métrica Gini / Inpuridad de MSE
La importancia de una variable se calcula sumando la reducción total del criterio de división (ej. reducción de varianza/MSE) que aporta esa variable en todos los árboles del bosque, ponderada por la cantidad de muestras que pasan por cada nodo.

### Consideración Técnica con One-Hot Encoding
Cuando se aplica `DictVectorizer`, las variables categóricas se dividen en múltiples características binarias (ej. `origin=USA`, `origin=Europe`). Para analizar la importancia real de la columna original:
- Se agrupan los valores de importancia sumando los componentes binarios derivados de la misma variable raíz.

---

## 6. Boosting Estocástico mediante XGBoost

### Fundamento de Gradient Boosting
A diferencia de Random Forest (donde los árboles se entrenan en paralelo e independientemente), en XGBoost los árboles se construyen de forma **secuencial**. Cada nuevo árbol intenta corregir los residuos (errores) cometidos por los árboles anteriores.

$$y_i^{(t)} = y_i^{(t-1)} + f_t(x_i)$$

### Estructura de Datos Nativa: `xgb.DMatrix`
- Estructura de memoria optimizada interna de XGBoost que mejora la velocidad de entrenamiento y la eficiencia de distribución en CPU/GPU.

### Hiperparámetros Críticos
- **`eta` (Learning Rate / Tasa de Aprendizaje):** Escala la contribución de cada nuevo árbol.
  - Valores más altos (`0.3`): Aprendizaje rápido, mayor riesgo de sobrepasar el óptimo o sobreajustar.
  - Valores moderados/bajos (`0.1` o menores): Aprendizaje más estable y robusto, requiere más rondas de boosting (`num_boost_round`).
- **`max_depth`:** Limita la profundidad de los árboles individuales (típicamente entre 3 y 10 en Boosting, ya que los árboles débiles suelen ser preferibles).

---

## 7. Buenas Prácticas Generales para Proyectos Tabulares

1. **Monitoreo de Métricas:** Utilizar la métrica RMSE en el conjunto de validación para realizar cualquier selección final de hiperparámetros.
2. **Visualización de Resultados:** Acompañar el ajuste de modelos con gráficas de evolución de error (RMSE vs `n_estimators`) y diagramas de barras para la importancia de características.
3. **Selección del Modelo Final:** Elegir la arquitectura basada en el balance entre precisión (menor RMSE), interpretabilidad y tiempo de inferencia en producción.