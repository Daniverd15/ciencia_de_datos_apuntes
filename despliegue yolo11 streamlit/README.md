# Despliegue en Streamlit de YOLO11 (n | s) × (detección | seg | pose)

Actividad de Ciencia de Datos (UNAB). Basada en el cuaderno *Avanzada Cuaderno 9C – train_yolo11_instance_segmentation*.
Se entrenaron **6 modelos** con el dataset de daños en carros [`cicatriz/danocarro` v1](https://universe.roboflow.com/cicatriz/danocarro) (Roboflow, CC BY 4.0): 8 clases, 3024 imágenes de entrenamiento, 318 de validación y 150 de prueba.

Clases: Abolladura, Agrietado, Corrosión, Descascaramiento, Desportillado de pintura, Parte faltante, Parte rota, Rayón.

## Modelos (`modelos/`)

| Modelo | Tarea | mAP50 (test) | mAP50-95 (test) | Métrica |
|---|---|---|---|---|
| `yolo11n.pt` | Detección | 0.346 | 0.260 | caja |
| `yolo11s.pt` | Detección | 0.360 | 0.273 | caja |
| `yolo11n-seg.pt` | Segmentación | 0.316 | 0.202 | máscara |
| `yolo11s-seg.pt` | Segmentación | 0.337 | 0.217 | máscara |
| `yolo11n-pose.pt` | Pose | 0.341 | 0.225 | puntos clave |
| `yolo11s-pose.pt` | Pose | 0.365 | 0.252 | puntos clave |

Todas las métricas están en [`metricas.json`](metricas.json). El modelo `s` supera al `n` en las tres tareas.

## Cómo se hizo

1. **Entorno:** entrenamiento local en GPU (RTX 5070, PyTorch cu128) en lugar de Colab; el procedimiento es el mismo del cuaderno.
2. **Datos:** el dataset original es de segmentación (polígonos). [`preparar_datasets.py`](preparar_datasets.py) deriva los otros dos:
   - *detección*: caja que encierra cada polígono;
   - *pose*: caja + 5 puntos clave **derivados del contorno** (superior, inferior, izquierdo, derecho y centro). El dataset no trae puntos clave anotados a mano, por lo que la pose es una adaptación didáctica.
3. **Entrenamiento:** [`entrenar_todos.py`](entrenar_todos.py) — transfer learning desde COCO, 30 épocas máx., `imgsz=640`, `batch=16`, early stopping (`patience=8`). Evaluación final sobre el conjunto de *test*.
4. **Despliegue:** [`app.py`](app.py) con Streamlit.

## Ejecutar la app

```bash
pip install -r requirements.txt
streamlit run app.py
```

En la barra lateral se elige el tamaño (n/s), la tarea y los umbrales de confianza e IoU; la imagen se sube o se toma con la cámara. Se muestra el resultado anotado y una tabla con clase, confianza, caja y área de máscara.

Para reentrenar: descargar el dataset de Roboflow en formato YOLOv11 (con tu propia API key como variable de entorno, nunca en el código), correr `preparar_datasets.py` y luego `entrenar_todos.py`.

## Limitaciones

Los mAP son moderados (recall ≈ 0.3): hay clases difíciles y poco frecuentes (rayones y desportillados pequeños). Más épocas o un modelo mayor probablemente mejorarían los resultados.
