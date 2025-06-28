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
        with st.spinner("Downloading model from Google Drive..."):
            gdown.download(id=MODEL_DRIVE_ID, output=MODEL_PATH, quiet=False)
        st.success("Model downloaded successfully!")

# -------- Load model --------
@st.cache_resource(show_spinner=True)
def load_unet_model():
    model = load_model(MODEL_PATH, custom_objects={'dice_loss': dice_loss, 'dice_coef': dice_coef})
    return model

# -------- Preprocess uploaded image --------
def preprocess_image(img: Image.Image):
    img = img.convert("L")  # grayscale
    img = img.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, IMG_SIZE[0], IMG_SIZE[1], 1)
    return img_array

# -------- Main app --------
def main():
    st.title("Brain Tumor Segmentation with U-Net")

    download_model()
    model = load_unet_model()

    uploaded_file = st.file_uploader("Upload a grayscale brain MRI image (.jpg, .png, .tif)", type=["jpg", "jpeg", "png", "tif"])
    if uploaded_file is not None:
        input_image = Image.open(uploaded_file)
        st.image(input_image, caption="Input Image", use_column_width=True)

        img_processed = preprocess_image(input_image)

        with st.spinner("Predicting tumor mask..."):
            pred_mask = model.predict(img_processed)[0, :, :, 0]
            pred_mask = (pred_mask > 0.5).astype(np.uint8)  # Threshold

        st.image(pred_mask, caption="Predicted Mask", clamp=True, use_column_width=True)

if __name__ == "__main__":
    main()
