# Reconocimiento de prendas (Fashion-MNIST)

Clasificación multiclase de imágenes de ropa de 28 × 28 píxeles en 10 categorías con una red densa en Keras, y una app de Streamlit para **dibujar** o **subir** una prenda y ver la predicción.

> Este proyecto también está como repositorio independiente, con el README completo (métricas por clase, matriz de confusión, script de entrenamiento local e imágenes de prueba): **[Daniverd15/Prendas](https://github.com/Daniverd15/Prendas)**.

| Archivo | Descripción |
|---------|-------------|
| `introduccion keras.ipynb` | Cuaderno de entrenamiento: carga de Fashion-MNIST, normalización, red `Flatten` + `Dense` + softmax y evaluación |
| `prendas.keras` | Modelo entrenado (exactitud en prueba: **88 %**) |
| `app.py` | App en Streamlit para dibujar una prenda o subir una imagen y obtener la predicción |
| `requirements.txt` | Dependencias de la app (versiones exactas) |

## Modelo

```text
Entrada 28×28 ─► Flatten (784) ─► Dense 64 ReLU ─► Dense 32 ReLU ─► Dense 16 ReLU ─► Dense 10 Softmax
```

Adam · `sparse_categorical_crossentropy` · 30 épocas · píxeles divididos entre 255.

## Ejecución

Requiere **Python 3.10 – 3.13** (TensorFlow 2.20 no está disponible para 3.14).

```bash
cd "Redes densas/Reconocimiento de prendas"
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
streamlit run app.py
```

Se abre en **http://localhost:8501**. Para reentrenar, abre el cuaderno en Google Colab y ejecútalo en orden; el modelo se guarda en Google Drive (`/content/drive/MyDrive/prendas/prendas.keras`) y hay que copiarlo junto a `app.py`.

> **Consejo:** el modelo aprendió con prendas claras sobre fondo negro. Al dibujar, usa trazos gruesos y centra la prenda.
