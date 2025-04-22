import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import BinaryIoU # Necesario si se guardó con métrica personalizada nombrada
import numpy as np
from PIL import Image
import io
from skimage.transform import resize # Usado para redimensionar como en el entrenamiento
import os

# --- Configuración de la página (¡MOVIDO AQUÍ!) ---
# Debe ser el PRIMER comando de Streamlit
st.set_page_config(layout="wide", page_title="Removedor de Fondo (Gatos/Perros)")

# --- Configuración de Constantes ---
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3
MODEL_PATH = 'model-oxford-pets-1.h5' # Asegúrate que esta ruta sea correcta

# --- Carga del Modelo (Cacheado para eficiencia) ---
@st.cache_resource # Usar cache_resource para objetos no serializables como modelos Keras
def load_keras_model(model_path):
    """Carga el modelo Keras preentrenado."""
    try:
        # Intenta cargar especificando el custom object si fue guardado con ese nombre
        # Revisa el nombre de la métrica en model.compile() y ModelCheckpoint -> 'iou'
        custom_objects = {'iou': BinaryIoU(threshold=0.5)}
        model = load_model(model_path, custom_objects=custom_objects)
        # Ya no se puede usar st.success/error aquí si se llama antes que otros st commands fuera de la función
        # Lo manejaremos después de la carga
        return model
    except Exception as e:
        # Captura el error para mostrarlo después
        st.session_state['model_load_error'] = f"Error al cargar el modelo desde {model_path}: {e}\nAsegúrate de que el archivo del modelo existe y que la métrica 'iou' (o BinaryIoU) está definida correctamente si es necesario."
        # Intenta cargar sin custom_objects como último recurso
        try:
            model = load_model(model_path)
             # Si tiene éxito, limpia el error
            if 'model_load_error' in st.session_state:
                del st.session_state['model_load_error']
            return model
        except Exception as e2:
             # Si el segundo intento también falla
             st.session_state['model_load_error'] += f"\nError secundario al cargar sin objetos personalizados: {e2}"
             return None

# Inicializa el estado de error si no existe
if 'model_load_error' not in st.session_state:
    st.session_state['model_load_error'] = None

# Carga el modelo una vez al inicio
model = load_keras_model(MODEL_PATH)

# Muestra mensajes de éxito/error DESPUÉS de set_page_config y la carga
if st.session_state['model_load_error']:
    st.error(st.session_state['model_load_error'])
    model = None # Asegurarse de que el modelo es None si hubo error
elif model is not None:
    st.success(f"Modelo cargado exitosamente desde {MODEL_PATH}")


# --- Funciones Auxiliares ---
def preprocess_image(image_pil, target_size=(IMG_HEIGHT, IMG_WIDTH)):
    """Preprocesa la imagen PIL para la entrada del modelo."""
    img = image_pil.convert('RGB') # Asegura 3 canales
    img_array = np.array(img)

    # Redimensionar como en el script de entrenamiento
    img_resized = resize(img_array, target_size, mode='constant', preserve_range=True)

    # El modelo espera uint8 [0-255] o float [0-255] debido a la capa Lambda(x/255.0)
    img_resized = img_resized.astype(np.uint8)

    # Añadir dimensión de batch
    img_batch = np.expand_dims(img_resized, axis=0)
    return img_batch

def remove_background(original_image_pil, predicted_mask, threshold=0.5):
    """Aplica la máscara predicha para quitar el fondo."""
    original_image_np = np.array(original_image_pil.convert('RGBA'))
    original_height, original_width, _ = original_image_np.shape

    # Redimensionar la máscara predicha al tamaño original
    mask_resized = resize(predicted_mask, (original_height, original_width),
                          mode='constant', preserve_range=True, order=0)

    # Umbralizar la máscara redimensionada
    binary_mask = (mask_resized > threshold).astype(np.uint8) # 0 o 1

    # Crear canal alfa
    alpha_channel = binary_mask * 255
    alpha_channel = np.squeeze(alpha_channel)

    # Aplicar el canal alfa
    original_image_np[:, :, 3] = alpha_channel

    # Convertir de nuevo a imagen PIL
    result_pil = Image.fromarray(original_image_np.astype(np.uint8), 'RGBA')
    return result_pil


# --- Interfaz de Streamlit ---
# st.set_page_config ya se llamó arriba

st.title("🐾 Removedor de Fondo para Gatos y Perros 🐶")
st.markdown("""
Sube una imagen de un gato o un perro y esta aplicación intentará quitar el fondo
utilizando un modelo U-Net entrenado en el dataset Oxford-IIIT Pet.
""")

# Solo procede si el modelo se cargó correctamente
if model is not None:
    uploaded_file = st.file_uploader("Elige una imagen...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Mostrar la imagen original
        image_original = Image.open(uploaded_file)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Imagen Original")
            st.image(image_original, caption="Imagen Subida", use_column_width=True)

        # Procesar la imagen y predecir
        with st.spinner('Procesando imagen y prediciendo máscara...'):
            # Preprocesar para el modelo
            img_for_model = preprocess_image(image_original, target_size=(IMG_HEIGHT, IMG_WIDTH))

            # Predecir la máscara
            predicted_mask_raw = model.predict(img_for_model) # Salida es (1, 128, 128, 1) con valores 0-1

            # Quitar la dimensión del batch y canal para postprocesado
            predicted_mask_squeezed = np.squeeze(predicted_mask_raw) # Ahora es (128, 128)

            # Quitar el fondo usando la máscara
            image_no_bg = remove_background(image_original, predicted_mask_squeezed, threshold=0.5)

        with col2:
            st.subheader("Resultado (Fondo Eliminado)")
            st.image(image_no_bg, caption="Imagen con fondo eliminado", use_column_width=True)

            # Añadir botón de descarga
            buf = io.BytesIO()
            image_no_bg.save(buf, format="PNG")
            byte_im = buf.getvalue()
            st.download_button(
                label="Descargar Imagen sin Fondo",
                data=byte_im,
                file_name=f"nobg_{uploaded_file.name.split('.')[0]}.png",
                mime="image/png"
            )

        # Opcional: Mostrar la máscara predicha (útil para depuración)
        with st.expander("Ver Máscara Predicha (redimensionada)"):
             # Redimensionamos la máscara 0-1 para visualización
             mask_display = resize(predicted_mask_squeezed, (image_original.height, image_original.width),
                                   mode='constant', preserve_range=True, order=0)
             st.image(mask_display, caption="Máscara Predicha por el Modelo (0=Negro, 1=Blanco)", use_column_width=True, clamp=True)


    elif uploaded_file is None:
        st.info("Por favor, sube un archivo de imagen.")

elif model is None:
     # El mensaje de error ya se mostró arriba si la carga falló
     st.warning("El modelo no pudo ser cargado. La funcionalidad principal está deshabilitada.")