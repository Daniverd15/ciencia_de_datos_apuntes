# Detección de objetos, segmentación y puntos clave

Cuadernos 9 a 9D del curso: de clasificar una imagen completa a **localizar** lo que hay dentro de ella. Se trabaja con la familia YOLO (Ultralytics) y con aprendizaje por transferencia sobre datasets etiquetados en Roboflow.

| Cuaderno | Tema | Modelo base | Datos |
|----------|------|-------------|-------|
| [9 · Detección de objetos con YOLOv8](<Avanzada_Cuaderno_9_CNN_Detección_de_Objetos_con_YOLOv8_y_Transfer_LearningL_.ipynb>) | Teoría de detección (modelos y datasets comunes, formatos de *bounding boxes*), entrenamiento desde cero vs. transferencia, evaluación, predicción en imágenes y video, exportación a ONNX, SavedModel y TFLite | `yolov8n.pt` | PPE Detection (Roboflow Universe) |
| [9A · Matrículas de vehículos](Avanzada_Cuaderno_9A_CNN_matriculas_de_vehiculos.ipynb) | Proyecto completo: detectar la placa con YOLOv8, recortarla y leerla con **EasyOCR**, también sobre video | `yolov8n.pt` | `car_detect` y `placa-de-carro` (Roboflow) |
| [9B · Elementos de protección personal](Avanzada_Cuaderno_9B_PPE_CNN_Entrenar_Detección_de_objeto_yolov8_Transfer_L.ipynb) | Plantilla de entrenamiento con un `data.yaml` propio: cascos y chalecos, métricas en TensorBoard, prueba sobre imágenes y video | `yolov8n.pt` | Hard Hat / PPE Detection (Roboflow) |
| [9C · Segmentación de instancias con YOLO11](Avanzada_Cuaderno_9C_train_yolo11_instance_segmentation.ipynb) | Anotación con polígonos en Roboflow, inferencia con un modelo de COCO (CLI y SDK), entrenamiento con *early stopping*, validación, despliegue y exportación | `yolo11m-seg.pt` | `cicatriz/danocarro`: daños en carros (Roboflow) |
| [9D · Detección de puntos clave](Avanzada_Cuaderno_9D_Deteccion_puntos_clave2.ipynb) | Regresión de 24 puntos del cuerpo de perros con un generador de datos y aumento con `imgaug`, *backbone* MobileNetV2 congelado y cabeza totalmente convolucional, pérdida MSE | MobileNetV2 (ImageNet) | StanfordExtra (Stanford Dogs) |

## Conceptos clave

| Concepto | En pocas palabras |
|----------|-------------------|
| **Detección** | Predice para cada objeto una caja (`x, y, w, h`), una clase y una confianza |
| **Segmentación de instancias** | Además de la caja, predice la máscara de píxeles de cada objeto; por eso se anota con polígonos |
| **Puntos clave** | Predice coordenadas de partes concretas (ojos, patas, cola); es un problema de regresión |
| **Transferencia** | Se parte de pesos entrenados en COCO o ImageNet y se ajustan a las clases propias: se necesitan muchas menos imágenes y épocas |
| **mAP50 / mAP50-95** | Precisión media con un solapamiento (IoU) mínimo de 0.5, o promediada entre 0.5 y 0.95: la métrica estándar de detección |
| **`data.yaml`** | Archivo que le dice a YOLO dónde están `train/`, `valid/` y `test/` y cuáles son las clases |

## Cómo ejecutarlos

Todos están pensados para **Google Colab con GPU** (*Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU T4*): instalan sus dependencias (`ultralytics`, `roboflow`, `easyocr`, `supervision`, `imgaug`) y descargan los datos en `/content/`.

- Los cuadernos que usan Roboflow piden una **API key** personal: créala gratis en [roboflow.com](https://roboflow.com) (*Settings → API Keys*) y pégala donde lo indica el cuaderno. No la subas al repositorio.
- El 9C entrena con `yolo11m-seg.pt` durante 50 épocas: en una T4 tarda bastante; para probar rápido baja las épocas o cambia a `yolo11n-seg.pt`.
- El 9D descarga las imágenes de Stanford Dogs (~750 MB).

## Relación con el despliegue

Los modelos de YOLO11 entrenados con el mismo dataset de daños en carros del cuaderno 9C están desplegados en una app de Streamlit en [`despliegue yolo11 streamlit/`](<../despliegue yolo11 streamlit>), con detección, segmentación y pose.
