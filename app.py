import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os

MODEL_PATH = "unet_model.keras"
FILE_ID = "1yR9Gar8U0VDRq9gbHiO_W8oZFupf3sjO"

def download_model():
    import gdown
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    output = MODEL_PATH
    gdown.download(url, output, quiet=False)

# Check and download model if necessary
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model..."):
        download_model()

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

st.title("🧠 Brain Tumor Segmentation App (MRI)")

uploaded_file = st.file_uploader("Upload an MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded MRI Image", use_column_width=True)

    img_resized = image.resize((128, 128))  # Adapt if needed
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0]
    prediction_mask = (prediction > 0.5).astype(np.uint8) * 255

    st.image(prediction_mask.squeeze(), caption="🧠 Tumor Segmentation Mask", use_column_width=True)
