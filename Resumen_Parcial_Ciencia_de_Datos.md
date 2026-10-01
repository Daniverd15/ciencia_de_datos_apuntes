# Apuntes de Ciencia de Datos para el parcial teórico

**Base del curso:** repositorio del profesor [`adiacla/Apuntes-Ciencia-de-Datos`](https://github.com/adiacla/Apuntes-Ciencia-de-Datos). Para contrastar y completar los materiales también se consultó [`AndiLinUnab/ciencia-de-datos-apuntes`](https://github.com/AndiLinUnab/ciencia-de-datos-apuntes), el repositorio de un compañero.

**Alcance de estos apuntes:** explicación personal de los conceptos de los cuadernos de clase. En la copia consultada del repositorio del profesor estaban las carpetas `Perceptron`, `Redes densas`, `imagenes` y `taller`; los cuadernos de procesamiento digital de imágenes y CNN se revisaron desde los archivos adjuntos y la copia del compañero. Los repositorios son fuentes de estudio, no se presenta este archivo como historial original de cambios en ellos.

La guía integra los temas y ejemplos de los materiales consultados. Los cuadernos de imágenes 6, 7 y 8 se trabajaron con los archivos compartidos en clase; algunas explicaciones y advertencias se añadieron para que los conceptos queden claros y para distinguir inconsistencias de comentarios del notebook.

> El README de `Redes densas/` dice que los temas del quiz son: **las 3 formas de declarar modelos en Keras (model_a, model_b, model_c)**, el flujo **`.compile()` → `.fit()` → evaluación** y el manejo de **`input_shape`** (`(1000, 20)` en el ejemplo sintético y `(12,)` en Celsius→Fahrenheit). Estúdialos primero.

---

## 0. Mapa del repositorio (qué hay y en qué orden estudiarlo)

| # | Archivo | Tema |
|---|---|---|
| 1 | `Perceptron/Readme.md` | Perceptrón, regla de actualización, SSE vs MSE, SGD/Batch/Mini-batch, descenso del gradiente y funciones de activación |
| 2 | `Perceptron/Avanzada_Cuaderno_1_ANN_El_Perceptron.ipynb` | Perceptrón manual (alumnos: Nota IA vs PGA) + `sklearn.linear_model.Perceptron` |
| 3 | `Perceptron/Cuanderno_1_1_Perceptron_con_Sklearn.ipynb` | Perceptrón con sklearn sobre `pacientes.csv` (edad, colesterol → problema cardíaco), MinMaxScaler, accuracy y **recall** |
| 4 | `Perceptron/MLP Circulo concetricos.ipynb` | `MLPClassifier` (16, 8, 4) sobre círculos concéntricos (células malignas y benignas) |
| 5 | `Perceptron/Avanzada_Cuaderno_2_...keras_tensorflow.ipynb` | Keras, PyTorch, TensorFlow y sklearn; tensores, softmax, `GradientTape`, `@tf.function`, `tf.Module`, SavedModel, entrenamiento manual y con Keras |
| 6 | `Redes densas/Readme.md` | Guía de estudio: modelos secuenciales y **temas del examen** |
| 7 | `Redes densas/Avanzada_Cuaderno_3_...regresion_lineal_Ejemplo.ipynb` | Teoría: 3 formas de declarar la red, summary/pesos, compile, fit, history, evaluate/predict **+** Celsius→Fahrenheit |
| 8 | `Redes densas/Avanzada_Cuaderno_4_...(Redes_densas).ipynb` | Activaciones, pérdidas, pérdida vs métrica, activación lineal por defecto, batch, tamaño de capas, train/val/test, overfitting, EarlyStopping **+** práctica `make_circles` |
| 9 | `Redes densas/Avanzada_Cuaderno_5_Regresion_Aplicación_Gasolina.ipynb` | Regresión con Auto-MPG (consumo de gasolina) con una red densa |
| 10 | `Redes densas/Reconocimiento de prendas.ipynb` | Fashion-MNIST: Flatten + Dense + Softmax (multiclase) |
| 11 | `Redes densas/Configuración de la red neuronal.ipynb` | Solo configuración del entorno: montar Drive, versiones, `nvidia-smi` |
| 12 | `procesamiento de imagenes/Avanzada Cuaderno 6 CNN Procesamiento digital de imágenes.ipynb` (archivo de clase / copia complementaria) | Imágenes como arreglos, OpenCV, canales, escala de grises, **convolución/kernels**, filtros, Sobel, segmentación (umbral, Otsu, K-Means), contornos |
| 13 | `taller/` (`procesamiento.py`, `app.py`) | Pipeline completo: limpieza, estandarización Z-score ×2, JSON para el *playground*, red escrita a mano y app en **Streamlit** |
| 14 | `Avanzada_Cuaderno_7_CNN_Redes_Neuronales_Convolucionales_.ipynb` (archivo de clase / copia del compañero) | CNN desde cero para clasificación multiclase con CIFAR-10: convolución, pooling, logits, pérdida, evaluación y predicción |
| 15 | `Avanzada Cuaderno 8 CNN Transferencia de aprendizaje.ipynb` (archivo de clase / copia del compañero) | Transfer learning con VGG16, extracción de características, congelar/descongelar capas, aumento de datos y clasificación de gatos y perros |

---

## 1. El Perceptrón

### 1.1 Definición y partes
- Lo propuso **Frank Rosenblatt en 1958**. Es la red neuronal más simple: **una sola neurona** que hace **clasificación binaria** (salida 0 o 1).
- **Entradas** $x_1..x_n$: las características.
- **Pesos** $w_1..w_n$: la importancia de cada entrada.
- **Sesgo (bias) $b$**: desplaza la frontera de decisión y permite que la neurona se active aunque todas las entradas valgan 0.
- **Suma ponderada:** $z = \sum_i w_i x_i + b$
- **Función de activación escalón:** $\hat y = 1$ si $z \ge 0$ y $\hat y = 0$ en otro caso. En el Cuaderno 1 el código usa `z + b > 0`, con desigualdad estricta.

### 1.2 Limitaciones (pregunta muy probable)
- Solo resuelve problemas **linealmente separables**, es decir, los que una recta o un hiperplano puede dividir. Por eso resuelve AND y OR.
- **No resuelve XOR**, que no es linealmente separable.
- La solución son varias neuronas organizadas en capas: el **MLP (Perceptrón Multicapa)**, base de las redes modernas.

### 1.3 Regla de aprendizaje del perceptrón
$$\text{Error} = y_{real} - y_{pred} \in \{-1, 0, 1\}$$
$$w_{nuevo} = w_{viejo} + \eta \cdot \text{Error} \cdot x \qquad b_{nuevo} = b_{viejo} + \eta \cdot \text{Error}$$
- **¿Por qué aparece $x$ en la actualización del peso?** Si $x_i = 0$, esa entrada no influyó en $z$, así que su peso **no se modifica**.
- Si el error es 0, no se cambia nada.

### 1.4 Ejemplo numérico del README (hazlo a mano)
- Datos: $x_1=1,\ x_2=0,\ y=1$; parámetros iniciales $w_1=0.3,\ w_2=-0.2,\ b=-0.5,\ \eta=0.1$.
- Forward: $z = 0.3 + 0 - 0.5 = -0.2 \Rightarrow \hat y = 0$.
- Error $= 1 - 0 = 1$.
- Actualización: $w_1 = 0.3 + 0.1 \cdot 1 \cdot 1 = \mathbf{0.4}$; $w_2 = -0.2$ (sin cambio, porque $x_2=0$); $b = -0.5 + 0.1 = \mathbf{-0.4}$.
- Verificación: $z = 0.4 - 0.4 = 0 \Rightarrow \hat y = 1$, ya acierta.

### 1.5 Compuerta OR con Python puro (README)
Parámetros iniciales: $w_1=0.3,\ w_2=-0.2,\ b=-0.1,\ \eta=0.1$, 4 épocas y escalón con $z \ge 0$. Traza verificada:

| Época | Correcciones | $w_1$ | $w_2$ | $b$ |
|---|---|---|---|---|
| 1 | 1 | 0.3 | −0.1 | 0.0 |
| 2 | 2 | 0.3 | 0.0 | 0.0 |
| 3 | 2 | 0.3 | 0.1 | 0.0 |
| 4 | 1 | 0.3 | 0.1 | −0.1 |

Resultado final: $w_1=0.3,\ w_2=0.1,\ b=-0.1$, que clasifica bien los 4 casos. Una 5ª época ya no haría correcciones.

### 1.6 SSE vs MSE
- **SSE** $= \sum (y - \hat y)^2$. En el perceptrón con escalón el error es −1, 0 o 1, así que el cuadrado vale 0 o 1: la SSE es el **número de clasificaciones erróneas** en la época. Eso es lo que imprime el Cuaderno 1 como `error_total`.
- **MSE** $= \frac{1}{N}\sum (y - \hat y)^2$. Elevar al cuadrado cumple tres funciones:
  1. Evita que errores positivos y negativos se cancelen.
  2. Penaliza más los errores grandes.
  3. Da una función **suave y derivable**, lo que permite usar descenso del gradiente.
- Con $L=\frac12(y-\hat y)^2$ se obtiene $\frac{\partial L}{\partial w} = -(y-\hat y)\,x$. De ahí sale la regla de actualización.

### 1.7 Instancia, época y estrategias de actualización
- **Instancia (muestra):** una fila del dataset.
- **Época:** una pasada completa por todo el conjunto de entrenamiento.

| Estrategia | Cuándo actualiza | Nº de actualizaciones (10 datos, 5 épocas) |
|---|---|---|
| **Estocástico (SGD)** | Después de cada instancia | **50** |
| **Batch GD (lote completo)** | Una vez por época, con el error promedio | **5** |
| **Mini-batch** (tamaño 2) | Una vez por bloque | **25** |

Fórmula general: actualizaciones por época $= \lceil N / \text{batch\_size} \rceil$.

### 1.8 Algoritmo de entrenamiento (pseudocódigo del Cuaderno 1)
1. Inicializar pesos y umbral al azar (`np.random.uniform(-1, 1)` con `seed(42)`).
2. `epocas_max = 100` y `tasa = 0.01`.
3. Mientras `epoca < epocas_max`: para cada instancia se predice, se calcula el error y se actualizan $w$ y $b$.
4. Terminar cuando el error sea 0 o se acaben las épocas. En el cuaderno el error llega a 0 cerca de la **época 65**: ahí el modelo **convergió**.

Los datos del Cuaderno 1 son 30 alumnos con `[Nota_IA, PGA]` normalizados entre 0 y 1; la clase 1 es "se gradúa" y la 0 "se retira". Primero se hace un scatter para comprobar que son linealmente separables. Luego se dibujan las **regiones de decisión** con una grilla de 0.05 coloreada según la predicción.

### 1.9 Épocas y tasa de aprendizaje
- **Pocas épocas** producen **underfitting** (el modelo no alcanza a aprender). **Demasiadas** pueden producir **overfitting**. Una regla práctica es empezar con 100 a 500.
- La **tasa de aprendizaje $\eta$** es el tamaño del paso. Suele estar entre **0.001 y 0.1**, y se recomienda empezar entre **0.01 y 0.1**.
  - Si es muy alta ($\ge 0.5$), el modelo oscila, se pasa del mínimo y puede **no converger**.
  - Si es muy baja ($\le 0.0001$), aprende muy lento y necesita muchas épocas.
  - Si el error oscila, **baja** la tasa. Si baja muy despacio, **súbela** un poco.
- Analogía del cuaderno: la tasa es el **largo de cada paso** y las épocas son **cuántos pasos** estás dispuesto a dar.
- Técnicas avanzadas que menciona: **learning-rate decay**, **grid search / random search**, validación.

### 1.10 Perceptrón en scikit-learn (Cuadernos 1 y 1.1)
```python
from sklearn.linear_model import Perceptron
p = Perceptron(max_iter=1000, eta0=1.0, tol=1e-3, random_state=123,
               early_stopping=False, verbose=1)
p.fit(X_train, y_train); p.predict([[0.2, 0.2], [0.8, 0.8]])
```
- **Valores por defecto:** `max_iter=1000` (máximo de épocas), `eta0=1.0` (tasa inicial) y `tol=1e-3`. Con `tol` el entrenamiento para si la mejora entre épocas es menor que ese valor. Una tolerancia más pequeña, como `1e-5`, da más precisión pero tarda más.
- `random_state` da **reproducibilidad**.
- Atributos del modelo entrenado:
  - `n_iter_`: épocas realizadas.
  - `coef_`: pesos.
  - `intercept_`: bias.
  - `n_features_in_`: número de características.
  - `classes_`: clases que predice.
- **Early stopping:** detiene el entrenamiento cuando la validación deja de mejorar durante `patience` épocas. Sirve para evitar el sobreajuste.
- **Persistencia:** `joblib.dump(modelo, "archivo.joblib")` y `joblib.load(...)`.

### 1.11 Cuaderno 1.1: flujo completo con `pacientes.csv`
1. `pd.read_csv(url)`, luego `df.shape` y `df.dropna(inplace=True)` para quitar nulos.
2. Guardar `cardiaco_limpio.csv` y revisar `df.describe().T`.
3. `X = df[['edad', 'colesterol']]` y `y = df['problema_cardiaco']`.
4. `train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)`. **`stratify`** mantiene la misma proporción de clases en train y en test.
5. `MinMaxScaler` escala a [0, 1]. Se usa **`fit_transform` solo en train** y **`transform` en test**; el scaler se guarda con joblib. Hacer `fit` sobre test sería **data leakage**.
6. Entrenar el Perceptron y medir `accuracy_score` en train y test, y `recall_score`.

**¿Por qué el recall importa en medicina?**
- $\text{Recall} = \frac{TP}{TP + FN}$, también llamado sensibilidad o tasa de verdaderos positivos.
- Minimiza los **falsos negativos**, es decir, decirle "sano" a alguien enfermo, que es el error peligroso.
- Un falso positivo solo cuesta pruebas extra o ansiedad.
- Con clases desbalanceadas (enfermedades raras), un modelo que siempre dice "sano" tiene un accuracy alto y un recall pésimo.
- Para enfermedades graves se busca un recall de **≥ 90 %**. El umbral final lo fijan expertos médicos y reguladores.

**Métricas de la descripción del Cuaderno 1** (complemento con fórmulas):
- $\text{Accuracy} = \frac{TP+TN}{\text{total}}$
- $\text{Precision} = \frac{TP}{TP+FP}$
- $F1 = 2\frac{P\cdot R}{P+R}$
- **Matriz de confusión**: tabla de aciertos y errores por clase.
- **Curva ROC y AUC**: TPR vs FPR para distintos umbrales. AUC = 1 es perfecto y AUC = 0.5 equivale al azar.

### 1.12 MLP con sklearn: círculos concéntricos (células)
- Los datos son sintéticos: clase 0 con radio de 0 a 0.4 (círculo interno, "malignas") y clase 1 con radio de 0.7 a 1.0 (anillo externo, "benignas"). Se agrega ruido gaussiano $\sigma = 0.08$ a $x_2$.
- **No son linealmente separables**, así que un perceptrón no alcanza y se necesita un MLP.
- Pasos: `train_test_split(..., stratify=y)` y luego **`StandardScaler`** (Z-score).
```python
MLPClassifier(hidden_layer_sizes=(16, 8, 4), activation='relu',
              learning_rate_init=0.03, max_iter=100, tol=0.001,
              n_iter_no_change=10, random_state=42)
```
- Atributos: `n_iter_`, `loss_` (pérdida final), `loss_curve_` (para graficar la pérdida) y `predict_proba` (probabilidad de cada clase, en el orden de `classes_`).
- Parámetros: $2\cdot16+16=48$, $16\cdot8+8=136$, $8\cdot4+4=36$ y $4\cdot1+1=5$, en total **225**.
- ⚠️ Detalles del cuaderno que pueden salir como pregunta trampa:
  - Una celda nombra las clases al revés: dice que 1 es "maligna", pero en los datos 1 es benigna.
  - La segunda celda de predicción pasa `[[0.55, 0.0]]` **sin escalar**, y siempre hay que aplicar `escalador.transform` antes de predecir.
  - Los comentarios dicen "x1=0.1", pero el código usa 0.55.

---

## 2. Descenso del gradiente

- La **superficie de error** es una "montaña 3D": el piso son los pesos y el sesgo, y la altura es el error. El objetivo es llegar al **mínimo global**.
- El **gradiente** $\nabla L$ apunta hacia donde la pendiente **sube** más rápido. Por eso se camina en la dirección **contraria**:
$$w_{nuevo} = w_{viejo} - \eta \frac{\partial L}{\partial w} \qquad b_{nuevo} = b_{viejo} - \eta \frac{\partial L}{\partial b}$$
- Si $\eta$ es muy grande te pasas del valle y subes por la otra pared. Si es muy pequeña necesitas miles de iteraciones.
- El perceptrón con escalón **no puede** usar gradiente porque la derivada del escalón es 0 en casi todo punto e indefinida en 0. Por eso las redes multicapa usan activaciones derivables.
- **Backpropagation** calcula esos gradientes hacia atrás con la **regla de la cadena**.

**Cuaderno 2, MSE en función de un solo peso.** Con $b=-5$ y $w_2=1$ fijos, se varía $w_1$ en $[-6, 10]$. La curva MSE–$w_1$ es una **parábola**, y la **tangente** en $w_1=0$ tiene pendiente igual al gradiente:
$$\frac{\partial L}{\partial w_1} = \frac{2}{n}\sum (b + w_1x_{1i} + w_2x_{2i} - y_i)\,x_{1i} = \frac{2}{n}X_1^\top(\hat y - y)$$
Si la pendiente es negativa hay que **aumentar** $w_1$; si es positiva, **disminuirlo**.

---

## 3. Funciones de activación (tema central)

Las activaciones introducen **no linealidad**. Sin ellas, varias capas equivalen a una sola regresión lineal: $W_3(W_2(W_1x))$ es otra transformación lineal.

| Función | Fórmula | Rango | Uso recomendado | Problema |
|---|---|---|---|---|
| **Escalón** | 1 si $z\ge0$, si no 0 | {0, 1} | Perceptrón clásico (AND, OR) | Derivada 0 o indefinida, así que no admite gradiente |
| **Sigmoide** | $\frac{1}{1+e^{-z}}$ | (0, 1) | **Salida de clasificación binaria** (probabilidad) | *Vanishing gradient* con \|z\| grande; se evita en capas ocultas profundas |
| **Tanh** | $\frac{e^z-e^{-z}}{e^z+e^{-z}}$ | (−1, 1) | Capas ocultas con datos **centrados en 0**; RNN y LSTM | También sufre *vanishing gradient*, aunque tiene gradientes más fuertes que la sigmoide cerca de 0 |
| **ReLU** | $\max(0, z)$ | [0, ∞) | **Opción por defecto en capas ocultas**; eficiente y reduce el *vanishing gradient* | "Neuronas muertas"; Leaky ReLU y ELU lo corrigen |
| **Lineal / identidad** | $f(z)=z$ | (−∞, ∞) | **Solo en la salida de regresión** | No aporta no linealidad; derivada constante = 1 |
| **Softmax** | $\frac{e^{z_i}}{\sum_j e^{z_j}}$ | (0, 1) y suma 1 | **Salida multiclase exclusiva** | — |

**Ejemplo de softmax** con $z=[2.0,\ 1.0,\ 0.1]$:
- $e^z \approx [7.389,\ 2.718,\ 1.105]$ y la suma es $\approx 11.212$.
- Resultado: $\approx [0.659,\ 0.242,\ 0.099]$, que suma 1.

**En Keras, `Dense` sin `activation` usa `linear` por defecto.** `layers.Dense(64)` es igual a `Dense(64, activation='linear')`. Si pones varias capas así sin activación, la red se comporta como una regresión lineal.

**Resumen según el tipo de problema:**
- **Regresión:** capas ocultas ReLU o Tanh y salida **Linear**.
- **Clasificación binaria:** capas ocultas ReLU o Tanh y salida **Sigmoid** con 1 neurona.
- **Multiclase exclusiva:** capas ocultas ReLU y salida **Softmax** con N neuronas, una por clase.

---

## 4. Funciones de pérdida y métricas

**La función de pérdida guía el aprendizaje.** Cuantifica la diferencia $L(y_{real}, y_{pred})$ y backpropagation la minimiza. Ejemplo: si el valor real es 100 y la predicción 90, el error es 10. El MSE lo eleva al cuadrado, el MAE toma el valor absoluto y la cross-entropy usa logaritmos.

| Problema | Activación de salida | Pérdida | En Keras |
|---|---|---|---|
| Regresión | Linear | **MSE** $\frac1n\sum(y-\hat y)^2$: castiga fuerte los errores grandes y es sensible a outliers | `'mse'`, `MeanSquaredError()` |
| Regresión | Linear | **MAE** $\frac1n\sum\lvert y-\hat y\rvert$: menos sensible a outliers y se interpreta directo | `'mae'`, `MeanAbsoluteError()` |
| Binaria | Sigmoid | **Binary Crossentropy** $-[y\log p + (1-y)\log(1-p)]$: castiga mucho los errores cometidos con alta confianza | `'binary_crossentropy'` |
| Multiclase con etiquetas **one-hot** | Softmax | **Categorical Crossentropy** $-\sum y_i\log p_i$ | `'categorical_crossentropy'` |
| Multiclase con etiquetas **enteras** (0, 1, 2…) | Softmax | **Sparse Categorical Crossentropy** | `'sparse_categorical_crossentropy'` |

**Pérdida vs métrica:**

| Pérdida | Métrica |
|---|---|
| Se usa durante el entrenamiento | Se usa para evaluar el desempeño |
| **Debe ser diferenciable** | No necesariamente diferenciable |
| Se minimiza con backprop y decide cómo cambian los pesos | No participa en el gradiente |
| Ej.: MSE, crossentropy | Ej.: accuracy, precision, recall, AUC, MAE |

Si eliges mal la pérdida, el modelo puede no aprender aunque la arquitectura sea buena. La pérdida debe coincidir con el **tipo de problema**, la **activación de salida** y el **formato de las etiquetas**.

---

## 5. Frameworks: Keras, TensorFlow, PyTorch y scikit-learn (Cuaderno 2)

- **Keras:** API de alto nivel escrita en Python para **prototipar rápido**. Corre sobre TensorFlow, que calcula los gradientes por debajo. Sus principios son interfaces simples y consistentes, pocos pasos, mensajes de error claros, **divulgación progresiva de la complejidad** y código conciso.
- **TensorFlow:** desarrollado por **Google**. Es una librería de cómputo numérico de **más bajo nivel** que **calcula gradientes automáticamente**. Da mucho control y conviene en proyectos grandes; lo usan Airbnb, Uber y Snapchat, entre otras.
- **PyTorch:** usa **grafos computacionales dinámicos** y tiene fuerte aceleración por GPU. Construye las redes y calcula los gradientes.
- **Scikit-learn:** clasificación, regresión y agrupamiento (clustering). Trabaja con NumPy y SciPy; lo usan Spotify y Booking, entre otras.
- **Buenas prácticas de entorno:** crear un entorno virtual con `python -m venv tf_env` y activarlo (`tf_env\Scripts\activate` en Windows). Usa **la misma versión de TensorFlow al entrenar y al desplegar**. Para revisar la GPU: `!nvidia-smi` y `tf.config.list_physical_devices('GPU')` (en Colab: Entorno de ejecución → T4 GPU).

### 5.1 Tensores
- Un tensor es un arreglo multidimensional parecido a los de NumPy, pero que puede correr en GPU. Sus atributos son **`shape`** (tamaño de cada eje) y **`dtype`** (float32, int32…).
- `tf.constant(...)` crea tensores **inmutables**. `x.numpy()` convierte a NumPy.
- Operaciones con `x = [[1,2,3],[4,5,6]]`, de shape (2, 3):
  - `x + x`, `x - x` y `5 * x` operan elemento a elemento.
  - `tf.transpose(x)` da shape (3, 2).
  - `x @ tf.transpose(x)` es el producto matricial: $[[14, 32],[32, 77]]$.
  - `tf.concat([x, x, x], axis=0)` da (6, 3), apilando filas. Con `axis=1` da (2, 9), pegando columnas.
  - `tf.reduce_mean(x, axis=1)` da la media **por muestra (fila)**: $[2, 5]$. Con `axis=0` da la media **por feature (columna)**: $[2.5, 3.5, 4.5]$.
  - `tf.reduce_sum(x)` da la suma total, 21.
  - `tf.nn.softmax(x, axis=1)` aplica softmax por fila.
- **`tf.Variable`** es **mutable** y sirve para los pesos. Se modifica con `.assign([1,2,3])`, `.assign_add(...)` y `.assign_sub(...)`.

### 5.2 Diferenciación automática: `tf.GradientTape`
- Es una "cinta" que **graba** las operaciones hechas sobre `tf.Variable`. Después, `tape.gradient(y, x)` las recorre **hacia atrás aplicando la regla de la cadena**.
- Si la variable es una constante hay que llamar a `tape.watch(t)`.
- Ejemplo con $f(x)=3x^3+2x^2-4x-2$:
  - $f'(x)=9x^2+4x-4$ y $f''(x)=18x+4$.
  - $f'(2)=40$; con $x=7.5$: $f=1346.125$ y $f'=532.25$.
  - Con $x=0.481$: $f'\approx 0.006 \approx 0$ y $f''\approx 12.66 > 0$, así que ahí hay un **mínimo local**. Si $f''>0$ la curva es cóncava hacia arriba (mínimo); si $f''<0$ es cóncava hacia abajo (máximo).
- La **segunda derivada** se calcula con **dos cintas anidadas**. `sympy.diff` da la expresión simbólica.

### 5.3 Grafos: `@tf.function`
- Convierte una función de Python en un **grafo computacional** optimizado que puede correr en CPU, GPU o TPU.
- El **tracing** ocurre la primera vez: por eso el `print('Tracing')` sale solo una vez, ya que `print` es de Python y no forma parte del grafo.
- Con **otra forma o dtype de entrada** se vuelve a trazar y se crea un grafo nuevo.
- Ventajas: más velocidad y la posibilidad de **exportar** el grafo con `tf.saved_model` para usarlo sin Python (servidor o móvil).

### 5.4 `tf.Module`, guardado y entrenamiento manual
- `tf.Module` administra variables y `tf.function`. Permite `tf.train.Checkpoint` (guardar y restaurar el estado) y `tf.saved_model.save(mod, './saved')` / `tf.saved_model.load(...)`.
- Un SavedModel es **independiente del código** y sirve para TF Serving, **TensorFlow Lite** (móvil) y **TensorFlow.js** (navegador).
- **Entrenamiento manual.** Los datos son $y = x^2 + 2x - 5 + \text{ruido}$, con 201 puntos en [−2, 2]. Se construye `X = [x, x²]` de shape (201, 2), y la neurona es `tf.matmul(X, W) + b`, con W de (2, 1) y **sin activación**:
```python
for epoch in range(200):
    with tf.GradientTape() as tape:
        loss = mse(modelo(X), y)
    gW, gb = tape.gradient(loss, [modelo.W, modelo.b])
    modelo.W.assign_sub(0.05 * gW); modelo.b.assign_sub(0.05 * gb)
```
- **El mismo modelo en Keras:**
```python
Sequential([Lambda(lambda x: tf.stack([x, x**2], axis=1)),
            Dense(1, kernel_initializer=RandomNormal(seed=42))])
compile(loss=MSE, optimizer=SGD(learning_rate=0.01)); fit(x, y, epochs=100, batch_size=32)
model.save('modelo_keras.keras')
```
  La capa **Lambda** hace la transformación de features ($x$ → $[x, x^2]$). Es una forma de aprender relaciones no lineales con un modelo lineal.
- **Keras = capas + modelos.** Una **capa** (`tf.keras.layers.Layer`) encapsula pesos y un `call`. Un **modelo** es un **DAG (grafo acíclico dirigido) de capas**. Los métodos del modelo son `fit`, `predict` y `evaluate`. Los *callbacks* sirven para EarlyStopping, checkpoints y TensorBoard. También admite entrenamiento distribuido y `steps_per_execution`.

---

## 6. Construcción de redes con Keras (Cuaderno 3, **examen**)

### 6.1 Las 3 formas de declarar un modelo
```python
# A) Lista en el constructor de Sequential (arquitectura conocida)
model_a = models.Sequential([
    layers.Input(shape=(20,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')], name="Modelo_Lista")

# B) Incremental con .add() (útil con bucles o if/else)
model_b = models.Sequential(name="Modelo_Add")
model_b.add(layers.Input(shape=(20,)))
model_b.add(layers.Dense(64, activation='relu'))
model_b.add(layers.Dropout(0.2))      # apaga el 20% de las neuronas al entrenar
model_b.add(layers.Dense(32, activation='relu'))
model_b.add(layers.Dense(1, activation='sigmoid'))

# C) API Funcional: x = Capa(...)(x_anterior)
entradas = layers.Input(shape=(20,), name="Entrada")
c1 = layers.Dense(64, activation='relu')(entradas)
c2 = layers.Dropout(0.2)(c1)
c3 = layers.Dense(32, activation='relu')(c2)
salida = layers.Dense(1, activation='sigmoid')(c3)
model_c = models.Model(inputs=entradas, outputs=salida, name="Modelo_Funcional")
```
- **Sequential** apila capas en forma lineal: la salida de una capa es la entrada de la siguiente.
- La **API funcional** es la más flexible y la más usada en la industria. Permite **varias entradas, varias salidas y ramificaciones** (ResNet, Inception). Se reconoce porque llama a cada capa con la variable anterior entre paréntesis y termina con `models.Model(inputs=..., outputs=...)`.

### 6.2 Inspección
- `model.summary()` muestra capas, *output shape* y número de parámetros.
- `for w in model.weights: w.name, w.shape` lista los pesos.
- `W, b = model.layers[0].get_weights()`: en un Sequential, `Input` **no cuenta como capa**, así que `layers[0]` es la primera Dense. W tiene shape (20, 64) y b tiene (64,).
- `tf.keras.utils.plot_model(model, show_shapes=True, show_layer_names=True)` dibuja la red; necesita `pydot` y `graphviz`.

### 6.3 Cómo contar parámetros (pregunta típica)
$$\text{parámetros de una Dense} = (\text{entradas} \times \text{neuronas}) + \text{neuronas (bias)}$$
- **model_a, b y c**: $20\cdot64+64=1344$, $64\cdot32+32=2080$ y $32\cdot1+1=33$, en total **3457**. **Dropout no tiene parámetros**, por eso los tres modelos suman lo mismo.
- **Celsius→Fahrenheit** (`Dense(1)` con 1 entrada): **2** parámetros ($w$ y $b$).
- **Círculos, 1 capa** (`Dense(1)` con 2 entradas): **3**. **Red profunda** 2→8→4→1: $24+36+5$ = **65**. **Con 3 features** 3→8→4→1: $32+36+5$ = **73**.
- **Fashion-MNIST**: Flatten convierte 28×28 en 784 (0 parámetros). Luego $784\cdot64+64=50240$, $64\cdot32+32=2080$, $32\cdot16+16=528$ y $16\cdot10+10=170$, en total **53 018**.
- **Auto-MPG** (7 features, 7→32→16→8→1): $256+528+136+9$ = **929**.

### 6.4 Datos sintéticos y `input_shape`
- `X = np.random.randn(1000, 20)` son 1000 muestras con 20 features, por eso `Input(shape=(20,))`. La forma de entrada **no incluye el número de muestras**.
- La etiqueta es `y = (X.sum(axis=1) > 0)`. La división es manual: 800 para train y 200 para validación (80/20).
- En Celsius→Fahrenheit hay 12 valores, de shape `(12,)`, y cada uno es **una** feature, por eso `Input(shape=(1,))`.

### 6.5 `.compile()`: optimizador, learning rate, pérdida y métricas
```python
model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
              loss=losses.BinaryCrossentropy(), metrics=['accuracy'])
```
- **Learning rate alto:** el modelo se vuelve inestable o diverge. **Bajo:** el entrenamiento es lento o se queda en un mínimo local.
- Optimizadores que aparecen en el repo:
  - **SGD** (descenso de gradiente estocástico).
  - **Adam**: tasa adaptativa y el más usado; `'adam'` usa lr = 0.001 por defecto.
  - **RMSprop**: usado en el problema de gasolina con lr = 0.005.

### 6.6 `.fit()`
```python
history = model.fit(x=X_train, y=y_train, epochs=30, batch_size=32,
                    validation_data=(X_val, y_val), verbose=1)
```
- `epochs` es el número de pasadas completas y `batch_size` las muestras que se procesan antes de actualizar los pesos.
- `validation_data=(X_val, y_val)` evalúa al final de cada época **sin entrenar con esos datos**. La alternativa es `validation_split=0.2`, que toma el último 20 % de train.
- **verbose:** 0 es silencioso, **1 muestra una barra de progreso** por época y **2 muestra una línea por época**.
- Cada paso de `fit` hace *forward pass*, cálculo de la pérdida, *backward pass* y actualización de pesos.
- `history.history` es un diccionario con `'loss'`, `'val_loss'`, `'accuracy'` y `'val_accuracy'` por época, y sirve para graficar las curvas.

### 6.7 Evaluar y predecir
```python
loss, acc = model.evaluate(X_val, y_val, verbose=0)
prob = model.predict(X_val[:5]); clase = (prob > 0.5).astype(int)   # umbral de 0.5 con sigmoide
# multiclase: np.argmax(model.predict(x)[0])  → índice de la clase más probable
```
Para guardar: `model.save('modelo.keras')` y `keras.models.load_model(...)`.

### 6.8 Ejercicio Celsius → Fahrenheit
- La fórmula es $F = 1.8\,C + 32$. La red tiene **una neurona**, `Input(shape=(1,))` y `Dense(1)`, y debe aprender $w \approx 1.8$ y $b \approx 32$.
- Los datos son 12 valores de Celsius, de −35 a 40. **No se preprocesan**, porque son valores simples.
- Se compila con `Adam(learning_rate=0.01)` y `loss='mean_squared_error'`, y se entrena con `fit(celsius, fahrenheit, epochs=1000)`. La pérdida se estabiliza cerca de la **época 400**.
- `model.predict(np.array([16]))` da ≈ **60.8 °F**. `model.get_weights()` muestra $w$ y $b$.
- Es un problema de **regresión lineal simple** resuelto con un perceptrón **sin activación**, es decir, lineal.

---

## 7. Diseño y entrenamiento de redes densas (Cuaderno 4)

### 7.1 Batch
- El **batch** es el número de muestras procesadas antes de actualizar los pesos. Hay tres variantes: **Batch GD** (todo el dataset), **SGD** (1 muestra) y **Mini-batch** (punto intermedio, el que se usa normalmente).
- Con 10 000 datos y batch 32 hay ≈ **312** actualizaciones por época. Con 500 datos y batch 32 hay ⌈15.6⌉ = **16** lotes por época.
- **¿Por qué usar batches?** Usan menos memoria, dan un entrenamiento más estable, se paralelizan en GPU y el "ruido" ayuda a escapar de mínimos locales.
- Tamaños típicos: **16, 32, 64 y 128**. Con un dataset pequeño conviene un batch pequeño; con mucha memoria de GPU, uno más grande. Un batch muy grande **generaliza peor**; uno muy pequeño da un entrenamiento **inestable**. Se suele empezar con 32 o 64.

### 7.2 Tamaño de las capas ocultas
- No hay fórmula exacta. Se usa un tamaño entre el de la entrada y el de la salida, preferiblemente **potencias de 2** (32, 64, 128, 256). Se empieza simple y se agranda si hay underfitting.
- **Embudo** (p. ej. 128 → 64 → 32 → salida): obliga a comprimir la información (*feature compression*) y reduce el sobreajuste. Es una buena práctica, pero **no es obligatoria**.

### 7.3 Train, Validation y Test
| Conjunto | ¿Aprende de él? | ¿Ajusta hiperparámetros? | ¿Evalúa el desempeño final? |
|---|---|---|---|
| Train | Sí | No | No |
| Validation | No (solo indirectamente) | **Sí**: épocas, arquitectura, learning rate, early stopping | No |
| Test | No | No | **Sí** |

- En resumen: **train entrena, validation ajusta y test evalúa**. Si usas el test repetidamente para tomar decisiones, deja de ser un test.
- Porcentajes sugeridos:
  - Dataset **grande**: 70/15/15.
  - Dataset **mediano**: 70/20/10.
  - Dataset **pequeño**: 80/10/10, o **K-Fold Cross Validation**.
- **K-Fold** divide los datos en K partes y entrena K veces, rotando la parte de validación. La evaluación es más robusta, pero cuesta más cómputo.
- Reglas:
  - No mezclar el test con el entrenamiento ni ajustar nada con él.
  - Hacer divisiones aleatorias (y estratificadas en clasificación).
  - En **series temporales**, no mezclar datos futuros con pasados.
  - Evitar el **data leakage**.

### 7.4 Overfitting (sobreajuste)
- El modelo aprende de memoria el entrenamiento, incluido el ruido, y **no generaliza**.
- **Señal:** $Loss_{train}\downarrow$ mientras $Loss_{val}\uparrow$, con una brecha creciente. Ejemplo: accuracy de train 98 % y de validación 82 %. En la gráfica, la curva de validación baja y luego **empieza a subir**: ahí comienza el sobreajuste.
- **Causas:** modelo demasiado complejo (muchas capas o neuronas), pocos datos, demasiadas épocas o falta de regularización.
- **Soluciones:**
  1. **EarlyStopping**.
  2. **Regularización L2**: `Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))`.
  3. **Dropout**: `layers.Dropout(0.5)`, que reduce la dependencia entre neuronas.
  4. **Reducir la complejidad** del modelo.
  5. **Más datos** o **data augmentation**.
- **Underfitting** es lo contrario: el modelo es muy simple o entrenó poco, y rinde mal tanto en train como en validación.
- Un error bajo en train **no** significa que el modelo sea bueno; lo que importa es el desempeño en datos no vistos.

### 7.5 EarlyStopping en Keras
```python
from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', patience=5, min_delta=0.001,
                           restore_best_weights=True, mode='min')
model.fit(X, y, validation_split=0.2, epochs=100, batch_size=32, callbacks=[early_stop])
```
- **monitor**: la métrica que se vigila. Se recomienda `'val_loss'`.
- **patience**: cuántas épocas sin mejora se esperan antes de parar. Se recomienda entre 3 y 10, y más si los datos son ruidosos.
- **min_delta**: la mejora mínima que cuenta como mejora real.
- **restore_best_weights=True**: al parar, vuelve a los pesos de la **mejor época**. Úsalo **siempre en producción**.
- **mode**: `'min'` para pérdidas y `'max'` para accuracy.
- **Ejemplo:** si la mejor val_loss fue en la época 15 y las épocas 16 a 20 no mejoran, con patience=5 se detiene en la 20 y restaura los pesos de la 15.
- Se combina con **ReduceLROnPlateau**. EarlyStopping **no mejora el modelo por sí mismo**: solo evita el sobreentrenamiento y conserva el mejor punto.

### 7.6 Práctica: círculos concéntricos con `make_circles`
- `make_circles(n_samples=500, factor=0.5, noise=0.05)` genera dos clases, una en el círculo interior y otra en el exterior. **No son linealmente separables.**
- **Modelo de una capa**, `Dense(1, 'sigmoid')`: es equivalente a una regresión logística y traza una frontera **recta**, así que el accuracy queda cerca de **50 %**, como el azar. **Conclusión: una sola capa no resuelve problemas no lineales.**
- **Modelo profundo** 2 → Dense(8, ReLU) → Dense(4, ReLU) → Dense(1, Sigmoid), con `adam` y `binary_crossentropy` durante 500 épocas: separa bien porque las capas ocultas con ReLU **transforman el espacio de características**.
- **Con EarlyStopping** (`validation_split=0.2`, patience=5, min_delta=1e-3) se para antes. Con `layer.get_weights()` se inspeccionan W y b de cada capa.
- **Ingeniería de características** (similar al truco de kernel polinomial de SVM): se agrega una tercera feature $(X_1 \cdot X_2)^2$. Al llevar los datos a 3D se separan mejor, y el modelo pasa a tener `Input(shape=(3,))`. Para graficar la frontera, **la grilla también necesita la feature nueva**. ⚠️ El texto del cuaderno la llama "X1²", pero el código calcula $(X_1X_2)^2$.
- **Mapa de decisión:** `np.meshgrid`, luego `np.c_[xx.ravel(), yy.ravel()]`, luego `predict` y `reshape`, y por último `plt.contourf`.
- Con `sigmoid` en la salida, se clasifica con `(pred > 0.5).astype(int)`.

---

## 8. Regresión: consumo de gasolina con Auto-MPG (Cuaderno 5)

- **Objetivo:** predecir `mpg` (millas por galón), una variable **continua**, así que es un problema de **regresión**.
- **Pasos:**
  1. Descargar con `kagglehub.dataset_download("uciml/autompg-dataset")` y leer `auto-mpg.csv`.
  2. Revisar nulos con `isnull().sum()` y **eliminar `car name`**, que es texto sin valor predictivo directo.
  3. Hacer EDA con `sns.pairplot(diag_kind="hist")` y un **heatmap de correlaciones** con `df.corr(numeric_only=True)`.
  4. `horsepower` trae `'?'`: se convierte con `pd.to_numeric(errors='coerce')`, que deja NaN, y se **imputa con la media**.
  5. `X = df.drop("mpg", axis=1)` y `y = df["mpg"]`, con división 80/20.
  6. **StandardScaler**: `fit_transform(X_train)` y **solo `transform(X_test)`**.
  7. Red: `Input(7)` → Dense(32, ReLU) → Dense(16, ReLU) → Dense(8, ReLU) → **Dense(1, linear)**.
  8. Compilar con `RMSprop(learning_rate=0.005)`, `loss='mse'` y `metrics=['mae']`.
  9. Entrenar con `fit(..., epochs=500, batch_size=32, validation_split=0.2)` y graficar loss vs val_loss.
  10. Guardar con `save_model("model_mpg.keras")` y evaluar en test con `mean_absolute_error` y `mean_squared_error`.
- ⚠️ Observaciones:
  - El cuaderno hace `train_test_split` en una celda **anterior** a la que define X e y, así que las celdas están desordenadas.
  - Importa `OneHotEncoder` pero no lo usa, aunque `origin` es categórica y lo ideal sería codificarla con one-hot.

---

## 9. Clasificación multiclase: Fashion-MNIST (Reconocimiento de prendas)

- **Datos:** 60 000 imágenes de train y 10 000 de test, de **28×28 en escala de grises** (0 a 255), en **10 clases**: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag y Ankle boot.
- **Normalización:** `images / 255.0`, que lleva los valores a [0, 1].
```python
model = models.Sequential([
    layers.Input(shape=(28, 28)),
    layers.Flatten(),                 # 28x28 -> vector de 784 (las Dense solo reciben vectores planos)
    layers.Dense(64, 'relu'), layers.Dense(32, 'relu'), layers.Dense(16, 'relu'),
    layers.Dense(10, 'softmax')])     # 10 clases, así que 10 neuronas con softmax
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=10)
```
- Se usa **`sparse_`** porque las etiquetas son **enteros** del 0 al 9, no one-hot.
- `model.evaluate(test_images, test_labels)` da el accuracy en test.
- `predictions[0]` es un vector de 10 probabilidades, y `np.argmax` devuelve el índice de la clase predicha, que se busca en `class_names[...]`.
- `model.save('modelo_reconocimiento_prendas.keras')`. En Colab también se puede guardar en Drive con `drive.mount('/content/drive')`.

---

## 10. Procesamiento digital de imágenes y convolución (Cuaderno 6)

### 10.1 Qué es una imagen digital
- Es una representación **discreta** de una imagen analógica: se **muestrean** las coordenadas y se **cuantizan** las intensidades. Se guarda como una **matriz de píxeles**.
- **Tipos:**
  - **Escala de grises** (intensidad): un valor por píxel, de 0 a 255 en 8 bits, en un arreglo 2D (alto, ancho).
  - **Binaria**: solo dos valores, 0 y 1 (blanco y negro).
  - **Color RGB**: tres canales, en un arreglo **3D** (alto, ancho, 3).
- **Representaciones:** 1D es la imagen aplanada (poco común, pero es lo que recibe una capa Dense). 2D corresponde a grises y 3D a color.
- Atributos de NumPy: `ndim` (número de dimensiones), `shape` (tupla con las dimensiones), `size` (número de elementos) y `dtype` (normalmente `uint8`).
- **`flatten()` vs `ravel()`**: los dos aplanan el arreglo. `flatten` devuelve una **copia**; `ravel` devuelve una **vista** cuando puede, así que gasta menos memoria.
- `np.concatenate(..., axis=0)` apila filas. `np.nditer` sirve para iterar sobre un canal.

### 10.2 Librerías y operaciones básicas
- **OpenCV (`cv2`)** hace operaciones de bajo nivel de visión: leer, escribir, modificar y analizar. **NumPy** manipula los arreglos, **Matplotlib** visualiza y **Pillow (PIL)** hace transformaciones de alto nivel. También se usan scikit-image y scipy.ndimage.
- Funciones principales:
  - `cv.imread('logo.jpg')` para leer y `cv.imwrite('out.jpg', img)` para escribir.
  - `cv.resize(img, (w, h))` para redimensionar.
  - `cv.line(img, p1, p2, color, grosor)` para dibujar una línea.
  - En Colab se muestra con `cv2_imshow`, porque `cv2.imshow` no funciona sin interfaz gráfica.
- ⚠️ **OpenCV lee en formato BGR**, no RGB. Hay que convertir con `cv.cvtColor(img, cv.COLOR_BGR2RGB)`; para pasar a grises se usa `cv.COLOR_BGR2GRAY`.
- **Binarización:** `cv.threshold(gray, 127, 255, cv.THRESH_BINARY)`.
- **Canales:** `im[:, :, 0]` es un canal y se ve en gris con `cmap='gray'`, porque es solo una matriz de intensidades. Para verlo "a color" se crea `np.zeros_like` y se copia solo ese canal. Recortar es slicing: `im[200:600, 500:800, 0]`.
- `imshow` interpreta los valores según el dtype: los **float** se mapean a 0–1, así que para ver valores de 0 a 255 hay que convertir a `uint8`.
- **Escala de grises ponderada:** $Gris = 0.2989R + 0.5870G + 0.1140B$. El ojo humano es más sensible al **verde**, luego al rojo y por último al azul.
- **Reducir el tamaño** (`cv.resize` o `scipy.ndimage.zoom(im, (0.2, 0.2, 1))`) abarata la convolución. El 1 en la tercera posición deja intactos los canales.
- **Rotación de color:** se trata cada píxel como un punto 3D. La imagen se normaliza con una transformación tipo sigmoide/log, se rota con una matriz de rotación en torno al eje X usando `np.einsum`, y se desnormaliza.

### 10.3 Convolución y kernels (base de las CNN)
- La **convolución** combina dos funciones para producir una tercera. En imágenes, **cada píxel se reemplaza por una suma ponderada de sus vecinos**: $C(x,y)=\sum I(x+x',y+y')\,W(x',y')$.
- El **kernel o filtro** es una matriz pequeña, normalmente 3×3 (también 5×5 o 7×7), que se desliza sobre la imagen. En cada posición se multiplica elemento a elemento con la vecindad y se suma. Según el kernel se suaviza, se detectan bordes o se realzan formas.
- **Kernels típicos:**
  - **Suavizado (blur) o media**: una matriz de $\frac19$ (3×3) o $\frac1{25}$ (5×5). Promedia los vecinos y reduce ruido y detalle. ⚠️ El cuaderno usa `np.ones((5,5))/15`, que no suma 1 y **aclara** la imagen; el texto dice 25.
  - **Sobel** (bordes horizontales): $\begin{bmatrix}-1&0&1\\-2&0&2\\-1&0&1\end{bmatrix}$. Aproxima el **gradiente** de intensidad. Combinando Sobel en X y en Y y calculando la magnitud se obtiene un mapa de bordes.
  - **Nitidez (sharpening)**: $\begin{bmatrix}0&-1&0\\-1&5&-1\\0&-1&0\end{bmatrix}$. Aumenta el contraste en las zonas de alta frecuencia.
  - **Laplaciano**: otro detector de bordes.
- Funciones: `cv2.filter2D(img, -1, kernel)` (el −1 mantiene el mismo tipo de dato) y `scipy.signal.convolve2d` aplicada **canal por canal** a los tres canales RGB.
- **Filtros de desenfoque:**
  - **Uniforme (media):** todos los píxeles pesan igual.
  - **Gaussiano:** el centro pesa más y el desenfoque es más suave.
  - **Mediana:** toma la mediana de la vecindad. **Elimina ruido conservando los bordes.**
  - Una ventana más grande produce más desenfoque.

### 10.4 Segmentación
- **Umbral simple:** se pasa a gris y los píxeles por encima del umbral quedan en 255 (blanco) y los de abajo en 0 (negro). Separa fondo y primer plano.
- **Otsu:** elige el umbral **automáticamente** a partir del histograma. Minimiza la **varianza intra-clase**, lo que equivale a maximizar la separación entre clases. Aplicado por canal R, G y B y luego combinado funciona mejor que sobre la imagen en gris.
- **K-Means:** agrupa los píxeles por color con `reshape` a `(h*w, 3)`. Cada píxel toma el color de su **centroide** (k = 3). Es más directo que buscar un umbral.
- **Vectorización y contornos:** se aísla un clúster y todo lo demás se pone en negro. Luego `skimage.measure.find_contours` encuentra los contornos y se **aproximan con polígonos**.
- **Relación con las CNN:** en una red convolucional los kernels **no se diseñan a mano; la red los aprende**. Las primeras capas terminan aprendiendo filtros parecidos a Sobel o blur.

---

## 11. Redes neuronales convolucionales para imágenes (Cuaderno 7)

### 11.1 Objetivo y datos de CIFAR-10

Una **CNN** (*Convolutional Neural Network*, red neuronal convolucional) es una red pensada para datos con estructura espacial, especialmente imágenes. Una red densa conectaría cada píxel con muchas neuronas y trataría el vector como una lista. Una CNN aprovecha que los píxeles vecinos forman patrones: bordes, texturas y partes de objetos. Conserva la distribución espacial al principio y aprende filtros que detectan patrones útiles.

El cuaderno entrena un clasificador de **CIFAR-10**. Contiene 50.000 imágenes para entrenamiento y 10.000 para prueba, con resolución de 32×32 y tres canales de color. Cada imagen tiene una etiqueta entre diez clases (avión, automóvil, pájaro, gato, ciervo, perro, rana, caballo, barco y camión). La forma `(32, 32, 3)` significa alto, ancho y canales; con el lote completo se añade antes el número de imágenes, por ejemplo `(50000, 32, 32, 3)`.

Los píxeles originales toman valores entre 0 y 255. Dividir por 255 los lleva al intervalo `[0, 1]`. Esta **normalización** mantiene el contenido visual y hace que las magnitudes de entrada sean más manejables para el entrenamiento.

### 11.2 Convolución y mapas de características

Una convolución desliza una matriz pequeña llamada **filtro** o **kernel** sobre la imagen. En cada posición multiplica sus valores por los píxeles vecinos y suma los productos. El resultado indica cuánto coincide esa zona con el patrón que el filtro representa.

En el procesamiento digital clásico, se elige el kernel manualmente (por ejemplo, Sobel para bordes). En una CNN entrenable, los valores de los filtros se **aprenden a partir de los datos** mediante retropropagación y descenso del gradiente. Cada filtro produce un **mapa de características**; una capa con 32 filtros produce 32 mapas o canales de salida.

La misma matriz de pesos se reutiliza en distintas posiciones de la imagen. A esto se le llama **compartición de pesos**. Reduce parámetros y permite detectar un patrón aunque aparezca en lugares diferentes. En capas iniciales la red suele representar bordes y texturas; capas posteriores combinan esas respuestas en patrones más complejos. Es una jerarquía aprendida, no una lista fija de objetos que la red entiende como una persona.

La operación de convolución también depende de **stride** (cuántos píxeles avanza el filtro) y **padding** (si se agregan píxeles en los bordes). El cuaderno usa las opciones usuales por defecto de `Conv2D`, sin padding para conservar el tamaño: con filtro 3×3 y stride 1, una entrada 32×32 pasa a 30×30. Añadir padding `same` puede conservar el alto y el ancho; aumentar el stride los reduce más rápidamente.

### 11.3 ReLU y MaxPooling

Después de las convoluciones se aplica **ReLU**, que conserva las activaciones positivas y convierte las negativas en cero. Su aporte central es introducir **no linealidad**: así, varias capas pueden modelar relaciones complejas que una sola transformación lineal no representaría.

**MaxPooling 2×2** divide el mapa en bloques y retiene el máximo de cada bloque. Reduce a la mitad aproximadamente el ancho y el alto, baja el costo de capas posteriores y conserva respuestas fuertes de cada zona. No aprende filtros: aplica una regla fija de reducción.

En el modelo del cuaderno, las dimensiones espaciales avanzan aproximadamente así: `32×32 → 30×30 → 15×15 → 13×13 → 6×6 → 4×4`. Al mismo tiempo aumenta la cantidad de mapas: de 32 a 64. Las capas más profundas trabajan con mapas más pequeños pero con más canales de características.

### 11.4 De mapas a clasificación

Tras las capas convolucionales, la salida tiene forma `4×4×64`. **Flatten** reorganiza esos valores en un vector de 1.024 elementos; no aprende una característica, solo cambia la forma. Una capa densa combina las características extraídas y la capa final entrega diez puntuaciones, una por cada clase.

Esas puntuaciones finales se llaman **logits**. Todavía no son probabilidades. La pérdida `SparseCategoricalCrossentropy(from_logits=True)` recibe logits y etiquetas enteras, compara la predicción con la clase correcta y produce el error que guiará la actualización de pesos. Si las etiquetas fueran vectores *one-hot*, se usaría la variante categórica correspondiente.

La función **softmax** puede convertir logits en probabilidades cuya suma es 1. Aunque el cuaderno no pone softmax explícitamente en la última capa, la pérdida configurada con `from_logits=True` realiza el cálculo apropiado internamente. La clase predicha es la de mayor puntuación (equivalente a la de mayor probabilidad después de softmax).

### 11.5 Entrenamiento, evaluación y sobreajuste

Una **época** recorre todas las muestras de entrenamiento. El cuaderno usa Adam para ajustar los pesos y compara la pérdida y la precisión durante el entrenamiento. Al final, la precisión de entrenamiento llega a aproximadamente 91 %, mientras la precisión reportada sobre el conjunto usado para validar queda alrededor de 70 %. También aumenta la pérdida de validación hacia las últimas épocas. Esa brecha sugiere **sobreajuste**: el modelo se adapta cada vez mejor a entrenamiento, pero no mejora al mismo ritmo en imágenes no vistas.

Advertencia metodológica: el código pasa el conjunto de prueba como `validation_data` durante `.fit()` y luego vuelve a evaluarlo como prueba. Al observar esa métrica durante el entrenamiento, se reutiliza información del conjunto reservado. Para estimar generalización de forma imparcial, se separan tres conjuntos: **train** ajusta pesos; **validation** orienta decisiones; **test** se reserva para la evaluación final.

El cuaderno también predice “avión” para una foto de un bus. No demuestra que la arquitectura sea incapaz de reconocer buses: la imagen se reduce a 32×32 y proviene de un contexto distinto a CIFAR-10. El resultado muestra que una predicción depende del entrenamiento, de la resolución y de que la nueva imagen se parezca a los datos con los que se aprendió.

### 11.6 Términos y relación con el Cuaderno 6

- **Clasificación de imágenes:** asignar una clase a una imagen completa.
- **Detección de objetos:** localizar objetos y clasificarlos, normalmente con recuadros.
- **Segmentación:** predecir una clase para cada píxel.
- **Aumento de datos:** crear variaciones realistas de entrenamiento para enseñar invariancia y reducir memorización.
- **CNN y kernels clásicos:** Sobel o blur son filtros elegidos por una persona; una CNN aprende los valores de sus propios filtros.

## 12. Aprendizaje por transferencia (Cuaderno 8)

### 12.1 La idea de reutilizar un modelo

Una red **preentrenada** ya ajustó sus pesos con un conjunto de datos grande. El **aprendizaje por transferencia** usa ese conocimiento como punto de partida para una nueva tarea relacionada. En este cuaderno se adapta una CNN preentrenada para distinguir fotos de gatos y perros.

La base de una CNN suele haber aprendido características visuales reutilizables: bordes, texturas y formas. En vez de aprender todos esos patrones desde cero con pocas imágenes, se aprovechan los pesos existentes y se agrega un clasificador que aprenda las categorías nuevas.

### 12.2 Extracción de características

En la estrategia básica se **congela** la base preentrenada: sus pesos no se actualizan. La base sigue procesando imágenes y extrayendo características, mientras solo se entrena el nuevo clasificador. Esto reduce tiempo y costo, y disminuye el riesgo de sobreajustar una base grande con pocos ejemplos.

Conviene cuando el conjunto nuevo es pequeño o parecido al dominio original. La salida de la base es una representación compacta de cada imagen, que las capas nuevas convierten en la predicción de gato o perro.

### 12.3 Ajuste fino (*fine-tuning*)

Una vez que el clasificador nuevo ya aprendió, se pueden **descongelar algunas capas superiores** de la base y entrenarlas junto con el clasificador. Esto permite adaptar características especializadas a la tarea nueva. Las capas iniciales suelen capturar patrones más generales, mientras que las finales reflejan más la tarea original; por eso se conservan a menudo las iniciales y se ajustan solo algunas finales.

El ajuste fino suele usar una **tasa de aprendizaje pequeña**. Una tasa alta puede cambiar demasiado deprisa pesos que ya eran útiles. Es más seguro entrenar primero la cabeza clasificadora y luego afinar gradualmente la base. Congelar y descongelar capas son decisiones sobre qué pesos pueden actualizarse, no cambios en el hecho de que el modelo procese la imagen.

**Transfer learning** nombra la estrategia amplia de reutilizar conocimiento. **Fine-tuning** es una modalidad en la que también se actualizan ciertas capas preentrenadas. Por ello, hacer aprendizaje por transferencia no necesariamente significa hacer ajuste fino.

### 12.4 Ejemplo con VGG16

El taller carga **VGG16** con pesos de ImageNet. En la adaptación a gatos y perros se usa `include_top=False`: se retira el clasificador original de ImageNet y se conserva la base convolucional como extractor. La entrada se prepara con el tamaño y la transformación de píxeles que espera VGG16; no hay un preprocesamiento universal para todas las arquitecturas.

La base produce mapas de características de tamaño `7×7×512` para la entrada 224×224 de este ejemplo. **GlobalAveragePooling2D** promedia cada mapa espacialmente y produce 512 valores por imagen. Es más compacto que aplanar los 7×7×512 valores: `Flatten` habría generado 25.088 elementos antes de la capa densa y habría aumentado el número de parámetros del clasificador.

Encima se añaden una capa densa intermedia, **Dropout** y una salida con **sigmoid**. Como es una clasificación binaria, la salida escalar representa una puntuación para una de las clases; un umbral como 0,5 convierte esa puntuación en una decisión. **BinaryCrossentropy** cuantifica el error frente a la etiqueta binaria.

### 12.5 Aumento y preparación de datos

Las imágenes deben redimensionarse al tamaño esperado por la base y prepararse con la normalización que corresponde a sus pesos. VGG16, MobileNetV2 y otras arquitecturas pueden esperar escalas diferentes. Aplicar la transformación equivocada cambia las entradas que reciben los filtros y puede reducir el rendimiento.

El **aumento de datos** del taller aplica variaciones como volteo, rotación, zoom o contraste. Aporta variedad artificial solo al entrenamiento y puede reducir el sobreajuste si las transformaciones conservan la etiqueta. El conjunto de validación y el de prueba deben reflejar datos sin esas variaciones aleatorias de entrenamiento.

El pipeline también usa lotes (*batches*) para procesar varias imágenes a la vez y *prefetch* para preparar datos mientras el modelo trabaja. Estas herramientas mejoran el uso del tiempo y la memoria; no cambian por sí solas la lógica de clasificación.

### 12.6 Evaluación y límites del ejemplo

El flujo separa entrenamiento, validación y prueba. El entrenamiento ajusta los pesos; la validación orienta el desarrollo; la prueba estima el rendimiento final. El cuaderno comunica una precisión de validación cercana al 98 % después del ajuste fino. Esa cifra no garantiza el mismo resultado con cualquier foto: depende del tamaño y representatividad de los datos, de su separación y del dominio al que se aplique el modelo.

El notebook mezcla en algunos comentarios los nombres **MobileNetV2** y **VGG16**. En el taller concreto de gatos y perros, el código crea `tf.keras.applications.VGG16`; sigue ese código y las formas observadas en el resumen de modelo. MobileNetV2 aparece como recomendación o ejemplo en otras explicaciones. **BERT** también figura en una tabla general, pero es un modelo de lenguaje, no una CNN de imágenes.

### 12.7 Cuándo elegir cada estrategia

| Situación | Estrategia razonable | Motivo |
|---|---|---|
| Pocos datos etiquetados y dominio parecido | Congelar la base y entrenar el clasificador | Aprovecha características generales y entrena pocos parámetros |
| Datos suficientes y tarea muy relacionada | Extracción inicial y luego ajuste fino | Permite adaptar algunas características superiores |
| Dominio muy distinto o preprocesamiento incompatible | Evaluar otra base, más datos o entrenamiento propio | El conocimiento previo puede transferirse mal |

## 13. Taller: del dato a la app (`taller/`)

**`procesamiento.py`:**
1. `read_csv("pacientes.csv")` y `dropna()`.
2. **Filtrar rangos válidos:** edad de 0 a 120 y colesterol de 100 a 600, para descartar valores imposibles u outliers.
3. Cambiar la etiqueta 0 por **−1**, de modo que las clases queden en {−1, 1}. Esto es compatible con una salida **tanh**.
4. Aplicar `StandardScaler` (**Z-score**: $z=\frac{x-\mu}{\sigma}$) y **multiplicar por 2**.
5. Guardar el scaler en `modelo_estandarizacion.joblib`.
6. Exportar `datos_procesados.json` en formato `{x, y, label}`, que es el que usa el **playground** (playground.scienxlab.org, similar al TensorFlow Playground).
7. Hacer un scatter de las dos clases y guardarlo como PNG.

**`app.py` (Streamlit):**
- La red está entrenada en el playground y **escrita a mano** con los pesos fijos en la función `forward`.
- **Entradas:** $X_1$, $X_2$ y la feature cruzada $X_1 X_2$.
- **Arquitectura:** 3 entradas → **8** → **6** → **4** → **3** neuronas ocultas con **ReLU** (`max(0, ·)`) → **1 salida con tanh**.
- Cada neurona calcula `bias + Σ w·entrada` seguido de su activación, que es exactamente el perceptrón de la sección 1.
- **Inferencia:** `scaler.transform([[edad, col]]) * 2` aplica **el mismo preprocesamiento que en el entrenamiento** (esta es la idea clave), luego `forward` y luego la clase: 1 si la salida es ≥ 0 y −1 en otro caso.
- La "probabilidad" se calcula como $(salida+1)/2$, que reescala el rango de tanh [−1, 1] a [0, 1]. No es una probabilidad calibrada.
- **UI de Streamlit:** `st.sidebar.slider`, `st.button`, `st.error` / `st.success` y recomendaciones según el riesgo.
- **`requirements.txt`**: pandas, scikit-learn, streamlit, matplotlib y joblib.

---

## 14. Persistencia de modelos (resumen)
| Qué | Cómo |
|---|---|
| Modelos y scalers de sklearn | `joblib.dump(obj, 'x.joblib' / 'x.pkl')` y `joblib.load` |
| Modelo de Keras | `model.save('x.keras')` y `keras.models.load_model` |
| Módulo o grafo de TensorFlow | `tf.saved_model.save` / `load`; se puede convertir a TF Lite (móvil) o TF.js (web) |
| Checkpoints durante el entrenamiento | `tf.train.Checkpoint` (y `ModelCheckpoint` en Keras) |
| **Regla** | Guardar también el **scaler** y aplicar el mismo preprocesamiento en producción, con la **misma versión** de las librerías |

---

## 15. Preguntas de repaso (con respuesta corta)

1. **¿Por qué un perceptrón no resuelve XOR?** Porque XOR no es linealmente separable y el perceptrón solo traza una frontera lineal. Hace falta un MLP con capas ocultas y activación no lineal.
2. **¿Para qué sirve el bias?** Para desplazar la frontera de decisión y que la neurona pueda activarse aunque todas las entradas sean 0.
3. **¿Qué es una época?** Una pasada completa por todo el conjunto de entrenamiento.
4. **Con 1000 datos y batch 50, ¿cuántas actualizaciones hay por época?** 20. Con SGD serían 1000 y con Batch GD, 1.
5. **¿Por qué el gradiente se resta?** Porque apunta hacia donde el error crece más rápido, y queremos bajar.
6. **¿Qué pasa si η es muy grande o muy pequeña?** Muy grande: oscila o diverge. Muy pequeña: converge muy lento.
7. **¿Qué activación y qué pérdida usar en la salida?**
   - Regresión: linear con MSE o MAE.
   - Binaria: sigmoid con binary_crossentropy.
   - Multiclase: softmax con (sparse_)categorical_crossentropy.
8. **¿Cuándo usar `sparse_categorical` y cuándo `categorical`?** Sparse cuando las etiquetas son enteros; categorical cuando son one-hot.
9. **¿Qué activación usa `Dense` si no se especifica?** Linear. Si todas las capas son lineales, la red equivale a un modelo lineal.
10. **¿Por qué ReLU en las capas ocultas?** Es barata de calcular, no se satura para z > 0 (reduce el vanishing gradient) y aporta no linealidad.
11. **¿Qué es el vanishing gradient?** Con sigmoide o tanh saturadas, la derivada es casi 0, los gradientes se vuelven diminutos al retropropagarse y las primeras capas casi no aprenden.
12. **¿Diferencia entre pérdida y métrica?** La pérdida es diferenciable y guía la actualización de los pesos. La métrica solo mide el desempeño.
13. **¿Cómo se detecta el overfitting?** Cuando la loss de train baja y la val_loss sube, o cuando hay una brecha grande entre las métricas de train y validación.
14. **¿Cómo se combate el overfitting?** Con EarlyStopping, L2, Dropout, un modelo más simple, más datos o data augmentation.
15. **Explica `patience` y `restore_best_weights`.** `patience` son las épocas sin mejora que se esperan antes de parar. `restore_best_weights` hace que el modelo vuelva a los pesos de la mejor época.
16. **¿Para qué sirve cada conjunto de datos?** Train para aprender, validation para ajustar hiperparámetros y detectar overfitting, y test para la evaluación final. El test nunca se usa para tomar decisiones.
17. **¿Por qué hacer `fit` del scaler solo con train?** Para evitar el data leakage: el test debe simular datos nunca vistos.
18. **¿Qué hace `stratify=y`?** Mantiene la proporción de clases en train y en test.
19. **¿Por qué el recall es crítico en medicina?** Porque minimiza los falsos negativos, que son enfermos no detectados.
20. **¿Cuántos parámetros tiene `Dense(32)` si recibe 64 entradas?** 64·32 + 32 = 2080.
21. **¿Cuáles son las 3 formas de crear un modelo en Keras?** Una lista en `Sequential([...])`, `Sequential()` con `.add()`, y la API Funcional con `Model(inputs, outputs)`. La funcional es la única que permite varias entradas o salidas y ramificaciones.
22. **¿Qué hace `Flatten`?** Convierte la imagen 28×28 en un vector de 784, porque las capas Dense reciben vectores.
23. **¿Para qué sirven `GradientTape` y `@tf.function`?** `GradientTape` graba las operaciones y calcula derivadas con la regla de la cadena. `@tf.function` convierte una función de Python en un grafo optimizado y exportable.
24. **¿Diferencia entre `tf.constant` y `tf.Variable`?** La constante es inmutable. La variable es mutable (`assign`) y es donde se guardan los pesos.
25. **¿En qué formato lee OpenCV las imágenes?** En BGR, así que para matplotlib hay que convertir a RGB.
26. **¿Cómo funciona un kernel de convolución?** Es una matriz pequeña que se desliza sobre la imagen: se multiplica elemento a elemento con la vecindad y se suma. Blur suaviza, Sobel detecta bordes y sharpen da nitidez.
27. **Otsu vs K-Means:** Otsu busca un umbral automático que minimiza la varianza intra-clase en el histograma. K-Means agrupa los píxeles por color en k clústeres.
28. **¿Por qué el taller multiplica el Z-score por 2 también en la app?** Porque en inferencia hay que repetir exactamente el preprocesamiento del entrenamiento; si no, los pesos no corresponden a la escala de los datos.
29. **¿Por qué una capa `Dense(1, sigmoid)` falla con `make_circles`?** Porque traza una frontera lineal y el problema es radial. Hacen falta capas ocultas con ReLU o una feature nueva como $(x_1x_2)^2$.
30. **¿Qué es el tracing en `@tf.function`?** Es la construcción del grafo en la primera llamada, o cuando cambia la forma o el dtype de la entrada. Por eso los `print` de Python solo aparecen esa vez.
