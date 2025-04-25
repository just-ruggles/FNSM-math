import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import os

# 🧠 Función de predicción
def predictDigit(image):
    model = tf.keras.models.load_model("model/handwritten.h5")
    image = ImageOps.grayscale(image)
    img = image.resize((28, 28))
    img = np.array(img, dtype='float32') / 255.0
    img = img.reshape((1, 28, 28, 1))
    pred = model.predict(img)
    return np.argmax(pred[0])

# 🌐 Configuración general
st.set_page_config(page_title='🧮 Reconocimiento de Dígitos', layout='wide')
st.markdown("""
    <style>
    .stApp {
        background-color: #0e0e0e;
        color: white;
    }
    h1 {
        font-size: 42px;
        color: #FF3131;
        text-shadow: 1px 1px 4px black;
        text-align: center;
    }
    h3 {
        color: #FFAB00;
        text-align: center;
    }
    .stSidebar {
        background-color: rgba(30, 30, 30, 0.95);
    }
    .stButton > button {
        background-color: #FF3131;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        font-size: 16px;
        padding: 8px 20px;
    }
    </style>
""", unsafe_allow_html=True)

# 🖋️ Título
st.title("✍️ Reconocimiento de Dígitos Escritos a Mano")
st.subheader("Cuando los que nos piden ayuda son niños muy pequeños, a veces dibujar los hace sentir mejor, pero no siempre entendemos qué quieren decir.")

# 🎨 Parámetros del canvas
stroke_width = st.slider("🖊️ Ancho del trazo", 1, 30, 15)
stroke_color = "#FFFFFF"
bg_color = "#000000"

canvas_result = st_canvas(
    fill_color="rgba(0, 0, 0, 0)",  # Sin fondo en el canvas
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    height=200,
    width=200,
    drawing_mode="freedraw",
    key="canvas",
)

# 🧠 Botón de predicción
if st.button("🔍 Predecir"):
    if canvas_result.image_data is not None:
        input_numpy_array = np.array(canvas_result.image_data)
        input_image = Image.fromarray(input_numpy_array.astype('uint8'), 'RGBA')
        os.makedirs("prediction", exist_ok=True)
        input_image.save("prediction/img.png")
        img = Image.open("prediction/img.png")
        res = predictDigit(img)
        st.success(f"🧠 El dígito es: {res}")
    else:
        st.warning("✋ Por favor dibuja un dígito antes de predecir.")

# ℹ️ Sidebar
st.sidebar.title("ℹ️ Acerca de esta app:")
st.sidebar.markdown("""
Esta aplicación utiliza una Red Neuronal entrenada con TensorFlow
para reconocer dígitos escritos a mano.

Aplicar esto en la app de FNSM fue una de nuestras más brillantes ideas.
""")
