# Autoencoders y redes generativas (GAN)

Cuadernos 14, 15 y 19 del curso: modelos que aprenden a **reconstruir** o a **generar** datos en vez de clasificarlos. Se aplican a imágenes y a señales de electrocardiograma (ECG).

| Cuaderno | Tema | Datos |
|----------|------|-------|
| [14 · Introducción a los autoencoders](<Avanzada_Cuaderno_14_Introducción_a_los_Autoencoder.ipynb>) | Encoder, espacio latente y decoder; autoencoder híbrido (`Conv2D` + `Dense`) que reconstruye prendas y autoencoder convolucional para **eliminar ruido** | Fashion-MNIST |
| [15A · Introducción al procesamiento de ECG](<Avanzada_Cuaderno_15A_ECG_Introducción_procesamiento.ipynb>) | Lectura de registros con `wfdb`, frecuencia de muestreo, anotaciones de latidos, intervalos RR, frecuencia cardíaca promedio e instantánea, clasificación de latidos y simulación de monitoreo en tiempo real | Registro 23 de la LTAFDB (PhysioNet) |
| [15B · Detección de anomalías cardiacas](Avanzada_Cuaderno_15B_ECG_Autoencoder.ipynb) | Autoencoder denso entrenado solo con latidos normales; el error de reconstrucción sobre un umbral marca un latido anómalo | ECG5000 (`ecg.csv` de TensorFlow) |
| [15C · Autoencoder sobre la LTAFDB](Avanzada_Cuaderno_15C_ECG_Autoencoder_LTAFDB_es.ipynb) | Segmentación de la señal en latidos, entrenamiento del autoencoder y detección de anomalías sobre un registro real | Registro 23 de la LTAFDB (PhysioNet) |
| [15D · Limpieza de imágenes](Avanzada_Cuaderno_15D_Limpieza_imagenes_2_autoencoders.ipynb) | Autoencoder convolucional que quita manchas y ruido de documentos escaneados | *Noisy documents* |
| [19 · GAN en Keras](Avanzada_Cuaderno_19__GAN_keras.ipynb) | Generador y discriminador que compiten: funciones de pérdida, optimizadores y bucle de entrenamiento que genera dígitos nuevos | MNIST |

## Conceptos clave

| Concepto | En pocas palabras |
|----------|-------------------|
| **Encoder** | Comprime la entrada en un vector pequeño (el código latente) |
| **Espacio latente** | Representación comprimida que conserva lo esencial de los datos |
| **Decoder** | Reconstruye la entrada a partir del código latente |
| **Error de reconstrucción** | Diferencia entre la entrada y la salida (MSE o MAE); si es grande, el dato no se parece a lo que el modelo aprendió |
| **Detección de anomalías** | Se entrena solo con datos normales y se fija un umbral (por ejemplo, media + k · desviación del error) |
| **Autoencoder de eliminación de ruido** | Recibe la imagen con ruido y se entrena para devolver la imagen limpia |
| **GAN** | El **generador** crea datos falsos a partir de ruido y el **discriminador** intenta distinguirlos de los reales; ambos mejoran compitiendo |

## Cómo ejecutarlos

Funcionan en Google Colab o en local con las dependencias del repositorio. Los cuadernos de ECG instalan `wfdb` y descargan el registro de [PhysioNet](https://physionet.org/content/ltafdb/1.0.0/); el resto descarga sus datos con Keras o desde GitHub.

```bash
pip install jupyter numpy pandas matplotlib scikit-learn tensorflow wfdb
cd "Autoencoders Gans"
jupyter notebook
```

> El 15A está pensado para leerse por partes: cada paso tiene una celda de explicación antes del código. El 15C reutiliza esa misma lectura del registro, así que conviene hacerlo después del 15A.
