# Apuntes de Ciencia de Datos

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?logo=tensorflow&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-F37626?logo=jupyter&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)

Repositorio de apuntes, cuadernos y talleres del curso de **Ciencia de Datos** de la Universidad Autónoma de Bucaramanga (UNAB). Reúne el material trabajado en clase sobre redes neuronales artificiales, desde el perceptrón hasta las redes convolucionales y el aprendizaje por transferencia, junto con un libro de estudio que resume cada cuaderno.

**Autor:** Daniel Enrique Villamizar Ramírez  
**Docente:** Alfredo Antonio Díaz Claro

---

## Contenido

- [Estructura del repositorio](#estructura-del-repositorio)
- [Ruta de estudio](#ruta-de-estudio)
- [Libro de apuntes](#libro-de-apuntes)
- [Cómo ejecutar el material](#cómo-ejecutar-el-material)
- [Tecnologías](#tecnologías)
- [Fuentes y créditos](#fuentes-y-créditos)

---

## Estructura del repositorio

```text
ciencia_de_datos_apuntes/
├── apuntes/                      Libro de estudio (Markdown y Word)
├── Perceptron/                   Cuadernos 1, 1.1 y 2 + MLP con scikit-learn
├── Redes densas/                 Cuadernos 3, 4 y 5
│   └── Reconocimiento de prendas/    Fashion-MNIST: cuaderno, modelo y app
├── procesamiento de imagenes/    Cuadernos 6, 7 y 8 (imágenes y CNN)
├── taller/                       Taller: del dato a una app en Streamlit
└── imagenes/                     Figuras usadas en las guías
```

## Ruta de estudio

| # | Tema | Material | Qué se aprende |
|---|------|----------|----------------|
| 1 | El perceptrón | [Cuaderno 1](Perceptron/Avanzada_Cuaderno_1_ANN_El_Perceptron.ipynb) · [Guía](Perceptron/Readme.md) | Neurona artificial, pesos, sesgo, función escalón, regla de actualización y entrenamiento manual con NumPy |
| 1.1 | Perceptrón con scikit-learn | [Cuaderno 1.1](<Perceptron/Cuanderno_1_1_Perceptron_con_Sklearn_ipynb (1).ipynb>) | `Perceptron` de scikit-learn, escalado con `MinMaxScaler`, accuracy y recall |
| — | Perceptrón multicapa | [MLP círculos concéntricos](<Perceptron/MLP Circuitos concentricos.ipynb>) | `MLPClassifier` sobre datos que no son linealmente separables |
| 2 | Frameworks | [Cuaderno 2](Perceptron/Avanzada_Cuaderno_2_ANN_Red_Neuronal_sklearn_keras_tensorflow.ipynb) | Tensores, sigmoide y softmax, diferenciación automática, compilación y entrenamiento en TensorFlow y Keras |
| 3 | Redes secuenciales | [Cuaderno 3](<Redes densas/Avanzada_Cuaderno_3_ANN_Red_neuronal_básica_de_regresion_lineal_Ejemplo.ipynb>) · [Guía](<Redes densas/Readme.md>) | Tres formas de declarar un modelo, `compile` → `fit` → `evaluate`, `input_shape`, regresión Celsius → Fahrenheit |
| 4 | Redes densas | [Cuaderno 4](<Redes densas/Avanzada_Cuaderno_4_ANN_Red_Neuronal_Clasificación_(Redes_densas).ipynb>) | Funciones de activación y de pérdida, tamaño de lote, sobreajuste, `EarlyStopping`, clasificación con `make_circles` |
| 5 | Regresión aplicada | [Cuaderno 5](<Redes densas/Cuaderno_Avanzado_5_Regresion_Aplicacion_Gasolina.ipynb>) | Predicción del consumo de gasolina con el conjunto Auto-MPG |
| — | Clasificación multiclase | [Reconocimiento de prendas](<Redes densas/Reconocimiento de prendas>) | Fashion-MNIST con `Flatten` + `Dense` + softmax y una app para dibujar y predecir |
| 6 | Procesamiento de imágenes | [Cuaderno 6](<procesamiento de imagenes/Avanzada Cuaderno 6  CNN  Procesamiento digital de imágenes.ipynb>) | Imágenes como arreglos, canales de color, convolución y kernels, filtros, Sobel, segmentación y contornos |
| 7 | Redes convolucionales | [Cuaderno 7](<procesamiento de imagenes/Avanzada_Cuaderno_7_CNN_Redes_Neuronales_Convolucionales_.ipynb>) | CNN desde cero sobre CIFAR-10: capas `Conv2D`, pooling, aumento de datos, evaluación y predicción |
| 8 | Transferencia de aprendizaje | [Cuaderno 8](<procesamiento de imagenes/Avanzada Cuaderno 8 CNN Transferencia de aprendizaje.ipynb>) | Modelos preentrenados (VGG16), extracción de características, congelar y descongelar capas, *fine-tuning* |
| — | Taller integrador | [taller/](taller) | Limpieza y estandarización de datos, exportación a JSON y despliegue de una red en Streamlit |

## Libro de apuntes

La carpeta [`apuntes/`](apuntes) contiene el resumen teórico de todos los cuadernos, pensado para repasar antes del parcial:

| Archivo | Descripción |
|---------|-------------|
| [Resumen_Parcial_Ciencia_de_Datos.md](apuntes/Resumen_Parcial_Ciencia_de_Datos.md) | Versión para leer en GitHub: mapa temático, explicación de cada cuaderno y preguntas de repaso con respuesta corta |
| [Resumen_Cuadernos_Ciencia_de_Datos.docx](apuntes/Resumen_Cuadernos_Ciencia_de_Datos.docx) | Versión en Word, con índice, lista para imprimir |

Además, cada carpeta de cuadernos incluye su propia guía (`Readme.md`).

## Cómo ejecutar el material

### Cuadernos

Los cuadernos están preparados para **Google Colab**: basta con abrirlos desde GitHub o subirlos a Colab y ejecutar las celdas en orden. Para trabajar en local:

```bash
git clone https://github.com/Daniverd15/ciencia_de_datos_apuntes.git
cd ciencia_de_datos_apuntes
python -m venv .venv
.venv\Scripts\activate        # En Linux/macOS: source .venv/bin/activate
pip install jupyter numpy pandas matplotlib seaborn scikit-learn tensorflow opencv-python scikit-image
jupyter notebook
```

> El cuaderno 6 usa las imágenes que están en su misma carpeta. Algunas celdas de los cuadernos 6, 7 y 8 apuntan a rutas de Colab (`/content/...`); en local hay que ajustarlas a la ruta del archivo.

### Aplicaciones en Streamlit

Taller de predicción de riesgo cardíaco:

```bash
cd taller
pip install -r requirements.txt
streamlit run app.py
```

Reconocimiento de prendas (Fashion-MNIST):

```bash
cd "Redes densas/Reconocimiento de prendas"
pip install -r requirements.txt
streamlit run app.py
```

## Tecnologías

| Área | Herramientas |
|------|--------------|
| Lenguaje y entorno | Python, Jupyter Notebook, Google Colab |
| Datos y visualización | NumPy, pandas, Matplotlib, Seaborn |
| Aprendizaje automático | scikit-learn |
| Aprendizaje profundo | TensorFlow, Keras |
| Imágenes | OpenCV, Pillow, scikit-image |
| Despliegue | Streamlit, joblib |

## Fuentes y créditos

- Los cuadernos numerados y las guías de `Perceptron/` y `Redes densas/` son material del curso, elaborado por el profesor Alfredo Antonio Díaz Claro: [adiacla/Apuntes-Ciencia-de-Datos](https://github.com/adiacla/Apuntes-Ciencia-de-Datos).
- Los cuadernos 6, 7 y 8 y sus archivos de apoyo corresponden a los compartidos en clase.
- Para contrastar los apuntes también se consultó el repositorio de un compañero del curso: [AndiLinUnab/ciencia-de-datos-apuntes](https://github.com/AndiLinUnab/ciencia-de-datos-apuntes).
- Los conjuntos de datos pertenecen a sus respectivos autores: Auto-MPG (UCI Machine Learning Repository), Fashion-MNIST (Zalando Research) y CIFAR-10 (Universidad de Toronto).

Este repositorio tiene fines exclusivamente académicos.
