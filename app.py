import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os

# Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("unet_model.keras")  # change name if needed
    return model

model = load_model()

st.title("🧠 Brain Tumor Segmentation - MRI")

uploaded_file = st.file_uploader("Upload an MRI image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded MRI", use_column_width=True)

    # Preprocessing
    image_resized = image.resize((128, 128))
    img_array = np.array(image_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)[0]
    prediction_mask = (prediction > 0.5).astype(np.uint8) * 255

    st.image(prediction_mask.squeeze(), caption="Tumor Segmentation", use_column_width=True)
