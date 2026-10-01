from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from streamlit_drawable_canvas import st_canvas


CLASS_NAMES = [
	"Camiseta/top",
	"Pantalón",
	"Suéter",
	"Vestido",
	"Abrigo",
	"Sandalia",
	"Camisa",
	"Zapatilla",
	"Bolso",
	"Botín",
]
MODEL_PATH = Path(__file__).parent / "prendas.keras"
CANVAS_SIZE = 280


st.set_page_config(
	page_title="Predictor de prendas con TensorFlow",
	page_icon="👕",
	layout="centered",
)


@st.cache_resource
def load_model():
	if not MODEL_PATH.is_file():
		raise FileNotFoundError(f"No se encontró el modelo en {MODEL_PATH}")
	return tf.keras.models.load_model(MODEL_PATH, compile=False)


def prepare_image(image: Image.Image) -> np.ndarray:
	grayscale = image.convert("L").resize((28, 28), Image.Resampling.LANCZOS)
	pixels = np.asarray(grayscale, dtype=np.float32) / 255.0
	return pixels[np.newaxis, ...]


def predict(image: Image.Image):
	probabilities = load_model().predict(prepare_image(image), verbose=0)[0]
	predicted_index = int(np.argmax(probabilities))
	return predicted_index, probabilities


st.title("Predictor de prendas con TensorFlow")
st.write("Dibuja una prenda o carga una imagen para identificar su categoría.")

try:
	load_model()
except Exception as error:
	st.error(f"No se pudo cargar el modelo prendas.keras: {error}")
	st.stop()


input_mode = st.radio("Selecciona una opción", ["Dibujar", "Subir imagen"], horizontal=True)
image_to_predict = None

if input_mode == "Dibujar":
	canvas_result = st_canvas(
		fill_color="rgba(0, 0, 0, 1)",
		stroke_width=12,
		stroke_color="#FFFFFF",
		background_color="#000000",
		width=CANVAS_SIZE,
		height=CANVAS_SIZE,
		drawing_mode="freedraw",
		key="prenda_canvas",
	)
	if canvas_result.image_data is not None:
		canvas_pixels = np.asarray(canvas_result.image_data, dtype=np.uint8)
		if np.any(canvas_pixels[:, :, :3] > 0):
			image_to_predict = Image.fromarray(canvas_pixels[:, :, :3]).convert("L")
else:
	uploaded_file = st.file_uploader(
		"Carga una imagen",
		type=["png", "jpg", "jpeg", "bmp", "webp"],
	)
	if uploaded_file is not None:
		image_to_predict = Image.open(uploaded_file)
		st.image(image_to_predict, caption="Imagen cargada", use_container_width=True)


if image_to_predict is not None:
	predicted_index, probabilities = predict(image_to_predict)
	st.subheader(f"Predicción: {CLASS_NAMES[predicted_index]}")
	st.progress(float(probabilities[predicted_index]))
	st.caption(f"Confianza: {probabilities[predicted_index] * 100:.2f}%")

	results = {
		CLASS_NAMES[index]: float(probability)
		for index, probability in enumerate(probabilities)
	}
	st.bar_chart(results, horizontal=True)


st.divider()
st.subheader("Instrucciones")
st.markdown(
	"""
	1. En **Dibujar**, traza la prenda con el lápiz blanco sobre el fondo negro.
	2. También puedes seleccionar **Subir imagen** y cargar un archivo.
	3. La imagen se convierte automáticamente a escala de grises y se redimensiona a 28 x 28 píxeles.
	4. Para obtener mejores resultados, usa imágenes similares a las utilizadas durante el entrenamiento: prendas claras sobre fondo negro.
	5. El modelo fue entrenado con valores de píxel normalizados dividiendo entre 255 y devuelve probabilidades mediante `softmax`.
	"""
)
