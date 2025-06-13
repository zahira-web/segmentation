import streamlit as st
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os

MODEL_PATH = "unet_model.keras"  # your model filename here

def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

@st.cache_resource
def load_my_model():
    model = load_model(MODEL_PATH, custom_objects={'dice_coef': dice_coef, 'dice_loss': dice_loss})
    return model

if not os.path.exists(MODEL_PATH):
    st.error(f"Model file {MODEL_PATH} not found. Please download or upload it.")
else:
    model = load_my_model()
    st.title("Brain Tumor Segmentation")

    uploaded_file = st.file_uploader("Upload MRI Image", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("L")  # grayscale for 1 channel
        image_resized = image.resize((128, 128))
        st.image(image_resized, caption="Input Image", use_column_width=True)

        img_array = np.array(image_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=(0, -1))  # shape: (1, 128, 128, 1)

        pred_mask = model.predict(img_array)[0, :, :, 0]
        pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255

        st.image(pred_mask, caption="Predicted Tumor Mask", use_column_width=True)
