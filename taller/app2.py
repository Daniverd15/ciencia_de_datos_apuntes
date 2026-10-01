import streamlit as st
import numpy as np
import joblib
from PIL import Image
import requests
from io import BytesIO

# Configuración de la página
st.set_page_config(page_title="Predicción Cardíaca", layout="centered")

# Título principal
st.title("🫀 Predicción Majestuosa de Problemas Cardíacos")

# Cargar y mostrar imagen del corazón
try:
    heart_image_url = "https://tecolotito.elsiglodetorreon.com.mx/i/2011/12/345983.jpeg"
    response = requests.get(heart_image_url)
    heart_image = Image.open(BytesIO(response.content))
    st.image(heart_image, width=300)
except Exception as e:
    st.warning(f"No se pudo cargar la imagen principal: {e}")

# Objetivo de la aplicación
st.markdown("""
### Objetivo de la Aplicación
Esta herramienta utiliza un modelo de red neuronal avanzado para predecir el riesgo de desarrollar 
problemas cardíacos basándose en dos factores clave: **edad** y **nivel de colesterol**. 
El análisis integrado proporciona una evaluación personalizada del riesgo cardiovascular, permitiendo 
a los usuarios obtener recomendaciones preventivas específicas según su perfil de edad y colesterol 
para mantener una salud cardíaca óptima.
""")

st.divider()

# Sección de entrada de datos
st.subheader("📋 Ingresa tus Datos")

col1, col2 = st.columns(2)

with col1:
    edad = st.slider("Edad (años)", min_value=20, max_value=100, value=50, step=1)

with col2:
    colesterol = st.slider("Nivel de Colesterol (mg/dL)", min_value=200, max_value=400, value=300, step=5)

st.divider()

# Botón para realizar predicción
if st.button("🔍 Realizar Predicción", use_container_width=True):
    try:
        # Cargar modelo y estandarizador
        modelo = joblib.load("modelo.joblib")
        modelo_estandarizacion = joblib.load("modelo_estandarizacion.joblib")
        
        # Preparar datos
        datos_entrada = np.array([[edad, colesterol]])
        
        # Normalizar datos
        datos_normalizados = modelo_estandarizacion.transform(datos_entrada) * 2
        
        # Realizar predicción
        prediccion = modelo.predict(datos_normalizados)[0]
        prediccion_proba = modelo.predict_proba(datos_normalizados)[0]
        
        st.divider()
        st.subheader("📊 Resultado de la Predicción")
        
        if prediccion == -1:
            # Resultado negativo (sin problemas cardíacos)
            st.success("✅ ¡Buenas noticias! No sufrirás problemas del corazón")
            
            # Mostrar imagen de corazón sano
            try:
                healthy_image_url = "https://cardiologoenguadalajarachavolla.com/portalweb/wp-content/uploads/2020/06/bigstock-People-Chest-Pain-From-Heart-A-279003418-768x512-1.webp"
                response = requests.get(healthy_image_url)
                healthy_image = Image.open(BytesIO(response.content))
                st.image(healthy_image, width=400)
            except Exception as e:
                st.warning(f"No se pudo cargar la imagen de corazón sano: {e}")
            
            # Recomendaciones para mantener salud cardíaca
            st.markdown("""
            ### 💚 Recomendaciones para Mantener tu Salud Cardíaca
            
            1. **Actividad Física Regular**: Realiza al menos 150 minutos de ejercicio moderado por semana
            2. **Alimentación Saludable**: Consume frutas, verduras y alimentos bajos en grasa saturada
            3. **Control de Estrés**: Practica meditación, yoga o actividades relajantes
            4. **Evita el Tabaco**: No fumes ni te expongas al humo del cigarrillo
            5. **Mantén un Peso Saludable**: Controla tu IMC regularmente
            6. **Duerme Adecuadamente**: Procura dormir 7-8 horas diarias
            7. **Controles Médicos**: Realiza revisiones periódicas con tu cardiólogo
            """)
            
        else:  # prediccion == 1
            # Resultado positivo (con riesgo de problemas cardíacos)
            porcentaje_riesgo = prediccion_proba[1] * 100
            
            st.error(f"⚠️ Atención: Existe riesgo de problemas cardíacos")
            st.metric("Porcentaje de Riesgo", f"{porcentaje_riesgo:.2f}%")
            
            # Mostrar imagen de advertencia cardíaca
            try:
                risk_image_url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRThzEhIFesPZWe9LeQ42f1oNo89GzPrWzdA-MiXc0MqA&s=10"
                response = requests.get(risk_image_url)
                risk_image = Image.open(BytesIO(response.content))
                st.image(risk_image, width=400)
            except Exception as e:
                st.warning(f"No se pudo cargar la imagen de riesgo: {e}")
            
            # Recomendaciones personalizadas según edad y colesterol
            st.markdown(f"""
            ### ❤️ Recomendaciones Personalizadas
            
            **Basado en tu edad ({edad} años) y colesterol ({colesterol} mg/dL):**
            """)
            
            if edad < 40:
                st.info("⏰ **Para tu edad:** Es fundamental actuar ahora para prevenir problemas futuros.")
            elif edad < 60:
                st.warning("⏰ **Para tu edad:** Requiere atención inmediata en estilo de vida.")
            else:
                st.error("⏰ **Para tu edad:** Necesitas seguimiento médico cercano y cambios urgentes.")
            
            if colesterol > 350:
                st.error(f"🔴 Tu colesterol ({colesterol} mg/dL) está muy elevado. Consulta a un cardiólogo inmediatamente.")
                recomendaciones_colesterol = """
                - **Medicación**: Considera medicamentos para reducir colesterol (estatinas)
                - **Dieta estricta**: Elimina grasas saturadas y trans
                - **Reducir sodio**: Máximo 2,300 mg diarios
                """
            elif colesterol > 300:
                st.warning(f"🟡 Tu colesterol ({colesterol} mg/dL) está elevado. Requiere intervención.")
                recomendaciones_colesterol = """
                - **Cambios dietéticos agresivos**: Aumenta fibra, reduce grasas
                - **Ejercicio intenso**: 5 días por semana, 30-45 minutos
                - **Consulta médica**: Evalúa necesidad de medicación
                """
            else:
                recomendaciones_colesterol = """
                - **Control moderado**: Mantén una dieta balanceada
                - **Ejercicio regular**: 3-4 días por semana
                """
            
            st.markdown(f"""
            {recomendaciones_colesterol}
            
            ### 🏥 Acciones Inmediatas Recomendadas:
            1. **Cita con Cardiólogo**: Agendar evaluación en los próximos 7 días
            2. **Pruebas Complementarias**: ECG, ecocardiograma si es necesario
            3. **Cambio de Hábitos**: Implementar cambios desde hoy
            4. **Monitoreo**: Revisiones periódicas cada 3 meses
            5. **Medicación**: Seguir prescripciones médicas al pie de la letra
            """)
        
    except FileNotFoundError:
        st.error("❌ Error: No se encontraron los archivos del modelo. Asegúrate de tener 'modelo.joblib' y 'modelo_estandarizacion.joblib' en la carpeta.")
    except Exception as e:
        st.error(f"❌ Error en la predicción: {str(e)}")

st.divider()

# Firma y acreditación
st.markdown("""
---
### ™️ Daniel Villamizar
**ES UN TRABAJO EXPERIMENTAL UNAB 2026**
""")
