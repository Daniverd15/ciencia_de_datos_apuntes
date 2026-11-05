"""Despliegue en Streamlit de los modelos YOLO11 entrenados para detectar danos en carros.

Permite elegir tamano (n | s) y tarea (deteccion | segmentacion | pose),
subir una imagen o usar la camara y ver el resultado.
Ejecutar: streamlit run app.py
"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

AQUI = Path(__file__).parent
MODELOS = AQUI / "modelos"
TAREAS = {"Deteccion": "", "Segmentacion": "-seg", "Pose": "-pose"}
CLAVE_METRICA = {
    "Deteccion": ("metrics/mAP50(B)", "metrics/mAP50-95(B)"),
    "Segmentacion": ("metrics/mAP50(M)", "metrics/mAP50-95(M)"),
    "Pose": ("metrics/mAP50(P)", "metrics/mAP50-95(P)"),
}

st.set_page_config(page_title="YOLO11 - Danos en carros", page_icon="🚗", layout="wide")


@st.cache_resource
def cargar(nombre: str) -> YOLO:
    return YOLO(str(MODELOS / f"{nombre}.pt"))


def metricas() -> dict:
    f = AQUI / "metricas.json"
    return json.loads(f.read_text()) if f.exists() else {}


st.title("🚗 YOLO11: deteccion de danos en carros")
st.caption("Ciencia de Datos - UNAB. Modelos YOLO11 n y s entrenados con el dataset danocarro (Roboflow) en deteccion, segmentacion y pose.")

with st.sidebar:
    st.header("Modelo")
    tamano = st.radio("Tamano", ["n", "s"], format_func=lambda x: {"n": "n (nano)", "s": "s (small)"}[x], horizontal=True)
    tarea = st.radio("Tarea", list(TAREAS))
    nombre = f"yolo11{tamano}{TAREAS[tarea]}"
    conf = st.slider("Confianza minima", 0.05, 0.95, 0.25, 0.05)
    iou = st.slider("IoU (NMS)", 0.1, 0.9, 0.7, 0.05)
    st.divider()
    m = metricas().get(nombre)
    st.subheader(f"`{nombre}`")
    if m:
        k50, k5095 = CLAVE_METRICA[tarea]
        st.metric("mAP50 (test)", m.get(k50, "-"))
        st.metric("mAP50-95 (test)", m.get(k5095, "-"))
    if tarea == "Pose":
        st.info("Los 5 puntos clave se derivan del contorno del dano (superior, inferior, izquierdo, derecho y centro); el dataset no trae puntos anotados a mano.")

if not (MODELOS / f"{nombre}.pt").exists():
    st.error(f"Falta el archivo modelos/{nombre}.pt. Ejecuta entrenar_todos.py primero.")
    st.stop()

modelo = cargar(nombre)

origen = st.radio("Origen de la imagen", ["Subir archivo", "Camara"], horizontal=True)
if origen == "Subir archivo":
    archivo = st.file_uploader("Imagen del vehiculo", type=["jpg", "jpeg", "png", "webp"])
else:
    archivo = st.camera_input("Toma una foto")

if archivo is None:
    st.info("Sube o captura una imagen para analizarla.")
    st.stop()

imagen = Image.open(archivo).convert("RGB")
with st.spinner("Analizando..."):
    r = modelo.predict(imagen, conf=conf, iou=iou, verbose=False)[0]

c1, c2 = st.columns(2)
c1.subheader("Original")
c1.image(imagen, use_container_width=True)
c2.subheader(f"Resultado ({tarea.lower()})")
c2.image(r.plot()[:, :, ::-1], use_container_width=True)

n = len(r.boxes)
st.subheader(f"Danos detectados: {n}")
if n:
    filas = []
    for i, b in enumerate(r.boxes):
        x1, y1, x2, y2 = (round(v) for v in b.xyxy[0].tolist())
        fila = {"#": i + 1, "Clase": r.names[int(b.cls)], "Confianza": round(float(b.conf), 3),
                "Caja (x1,y1,x2,y2)": f"{x1},{y1},{x2},{y2}"}
        if r.masks is not None:
            fila["Area mascara (% imagen)"] = round(float(r.masks.data[i].mean()) * 100, 2)
        filas.append(fila)
    st.dataframe(pd.DataFrame(filas), use_container_width=True, hide_index=True)
else:
    st.success("No se detectaron danos con esta confianza. Prueba a bajar el umbral.")
