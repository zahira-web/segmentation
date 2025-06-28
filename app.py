import os
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
import gdown

# -------- Constants --------
MODEL_DRIVE_ID = "1hPOZ90Uc054k7uygkQKlL0feIQ6UaiYt"
MODEL_PATH = "unet_model.h5"
IMG_SIZE = (128, 128)

# -------- Custom dice functions --------
def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

# -------- Download model if needed --------
def download_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("📥 Téléchargement du modèle depuis Google Drive..."):
            gdown.download(id=MODEL_DRIVE_ID, output=MODEL_PATH, quiet=False)
        st.success("✅ Modèle téléchargé avec succès!")

# -------- Load model --------
@st.cache_resource(show_spinner=True)
def load_unet_model():
    model = load_model(MODEL_PATH, custom_objects={'dice_loss':
