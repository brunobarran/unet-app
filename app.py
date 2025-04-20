import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from skimage.transform import resize
import io

# --- Constantes (ajusta según tu script original) ---
MODEL_PATH = 'model-oxford-pets-1.h5'
IMG_HEIGHT = 128
IMG_WIDTH = 128

# --- Cargar Modelo (usando cache de Streamlit para eficiencia) ---
@st.cache_resource # Cache para no recargar en cada interacción
def load_keras_model():
    print("Intentando cargar el modelo...")
    try:
        # Asegúrate que tu archivo .h5 está en el repo junto a este script
        model = load_model(MODEL_PATH, compile=False)
        print("Modelo cargado.")
        return model
    except Exception as e:
        st.error(f"Error cargando el modelo: {e}")
        return None

model = load_keras_model()

# --- Funciones de Procesamiento (las mismas de tu app Flask) ---
def preprocess_image(image_pil):
    if image_pil.mode != 'RGB':
        image_pil = image_pil.convert('RGB')
    img_resized = resize(np.array(image_pil), (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    img_array = img_resized.astype(np.float32) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)
    return img_batch

def apply_mask_to_image(original_pil, mask_pred):
    original_rgba = original_pil.convert("RGBA")
    # Redimensionar máscara a tamaño original (o aplicar en pequeño y redimensionar resultado)
    # Aquí aplicamos en pequeño y redimensionamos resultado para simplicidad
    mask_resized_small = resize(mask_pred, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True, order=0)
    img_resized_pil_small = original_pil.resize((IMG_WIDTH, IMG_HEIGHT))
    img_resized_rgba_small = img_resized_pil_small.convert("RGBA")
    img_resized_np_small = np.array(img_resized_rgba_small)
    binary_mask = (mask_resized_small > 0.5).squeeze()
    alpha_channel = np.where(binary_mask, 255, 0).astype(np.uint8)
    img_resized_np_small[:, :, 3] = alpha_channel
    result_img_pil = Image.fromarray(img_resized_np_small, 'RGBA')
    # Redimensionar resultado al tamaño original para mejor visualización
    result_img_pil = result_img_pil.resize(original_pil.size, Image.Resampling.LANCZOS)
    return result_img_pil

# --- Interfaz de Streamlit ---
st.set_page_config(layout="wide") # Opcional: usar ancho completo
st.title("🐾 Removedor de Fondo para Mascotas 🐾")
st.write("Sube una imagen de una mascota (perro o gato) para quitarle el fondo.")

uploaded_file = st.file_uploader("Elige una imagen...", type=["jpg", "jpeg", "png"])

if model is None:
    st.error("El modelo no está disponible. No se puede procesar.")
elif uploaded_file is not None:
    # Mostrar imagen original
    image_original = Image.open(uploaded_file)

    col1, col2 = st.columns(2) # Crear dos columnas para mostrar lado a lado

    with col1:
        st.header("Original")
        st.image(image_original, caption="Imagen Subida", use_column_width=True)

    # Procesar y mostrar resultado
    with st.spinner('Procesando imagen... esto puede tardar un momento.'):
        try:
            img_batch = preprocess_image(image_original)
            pred_mask = model.predict(img_batch)[0]
            result_img = apply_mask_to_image(image_original, pred_mask)

            with col2:
                st.header("Resultado (Fondo Eliminado)")
                # Convertir a bytes para mostrar correctamente transparencia en st.image
                buf = io.BytesIO()
                result_img.save(buf, format="PNG")
                byte_im = buf.getvalue()
                st.image(byte_im, caption="Fondo Eliminado", use_column_width=True)
        except Exception as e:
             st.error(f"Ocurrió un error durante el procesamiento: {e}")

st.sidebar.info(f"Modelo: {MODEL_PATH} (Imagen espera {IMG_WIDTH}x{IMG_HEIGHT})")