# Reconocimiento de prendas (Fashion-MNIST)

Clasificación multiclase de imágenes de ropa de 28 × 28 píxeles en 10 categorías con una red densa en Keras.

| Archivo | Descripción |
|---------|-------------|
| `introduccion keras.ipynb` | Cuaderno de entrenamiento: carga de Fashion-MNIST, normalización, red `Flatten` + `Dense` + softmax y evaluación |
| `prendas.keras` | Modelo entrenado |
| `app.py` | App en Streamlit para dibujar una prenda y obtener la predicción del modelo |
| `requirements.txt` | Dependencias de la app |

## Ejecución

```bash
pip install -r requirements.txt
streamlit run app.py
```
