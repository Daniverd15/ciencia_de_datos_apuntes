# Taller: del dato a la aplicación

Flujo completo de un problema de clasificación: predecir el riesgo de problemas cardíacos a partir de la **edad** y el **colesterol** de un paciente.

## Archivos

| Archivo | Descripción |
|---------|-------------|
| `pacientes.csv` | Datos originales |
| `procesamiento.py` | Limpia los datos, los estandariza y genera el resto de archivos |
| `datos_procesados.json` | Datos estandarizados, listos para cargar en el *playground* de redes neuronales |
| `modelo_estandarizacion.joblib` | Escalador ajustado, para transformar datos nuevos igual que en el entrenamiento |
| `grafico_dispersion.png` | Gráfico de dispersión de los datos procesados |
| `app.py` | App en Streamlit con la red escrita a mano (pesos tomados del *playground*) |
| `app2.py` | Variante de la app que carga un modelo entrenado desde `modelo.joblib` |

## Ejecución

```bash
pip install -r requirements.txt
python procesamiento.py
streamlit run app.py
```

> `app2.py` necesita un archivo `modelo.joblib` con el modelo entrenado en la misma carpeta; no se incluye en el repositorio.
