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

# 🎨 Estilos personalizados
st.set_page_config(page_title='🧮 Reconocimiento de Dígitos a Mano', layout='wide')

st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://pbs.twimg.com/media/F2sr38KWYAAj0bc.jpg:large");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("""
    <style>
    .stApp {
        background-color: #0e0e0e;
        color: white;
    }
    h1 {
        font-size: 48px;
        color: #FF3131;
        text-align: center;
        text-shadow: 2px 2px 8px #000;
    }
    h3 {
        color: #FFAB00;
        text-align: center;
    }
    .stSlider > div {
        color: white;
    }
    .stButton > button {
        background-color: #FF3131;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        font-size: 16px;
        padding: 10px 24px;
    }
    .stSidebar {
        background-color: rgba(30, 30, 30, 0.9);
    }
    </style>
""", unsafe_allow_html=True)

# 🖋️ Encabezado
st.title("✍️ Reconocimiento de Dígitos Escritos a Mano")
st.subheader("Cuando los que nos piden ayuda son niños muy pequeños, a veces dibujar los hace sentir mejor, pero no siempre entendemos qué quieren decir.")

# 📏 Ajustes del canvas
stroke_width = st.slider("🖊️ Ancho del trazo", 1, 30, 15)
stroke_color = "#FFFFFF"
bg_color = "#000000"

# 🎨 Canvas de dibujo
canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 0.2)",  # color del relleno
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    height=200,
    width=200,
    drawing_mode="freedraw",
    key="canvas",
)

# 📤 Botón para predecir
if st.button("🔍 Predecir"):
    if canvas_result.image_data is not None:
        input_numpy_array = np.array(canvas_result.image_data)
        input_image = Image.fromarray(input_numpy_array.astype('uint8'), 'RGBA')
        os.makedirs("prediction", exist_ok=True)
        input_image.save("prediction/img.png")
        img = Image.open("prediction/img.png")
        result = predictDigit(img)
        st.success(f"🧠 El dígito es: {result}")
    else:
        st.warning("✋ Por favor dibuja un dígito antes de predecir.")

# ℹ️ Sidebar informativa
st.sidebar.title("ℹ️ Acerca de esta app:")
st.sidebar.markdown("""
Esta aplicación utiliza una Red Neuronal Artificial (RNA)
entrenada para reconocer dígitos escritos a mano usando TensorFlow.

Aplicar esto en la app de FNSM fue una de nuestras más brillantes ideas.
""")
