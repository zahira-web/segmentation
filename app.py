import streamlit as st
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os
import subprocess

MODEL_PATH = "unet_model.keras"
GDRIVE_ID = "1cVa9hNetMAQZyQUUoVM5y9AA4yjM048K"  

def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

@st.cache_resource
def download_model():
    if not os.path.exists(MODEL_PATH):
        st.info("Downloading model from Google Drive...")
        # Install gdown if not present
        subprocess.run(["pip", "install", "gdown"], check=True)
        # Download model
        subprocess.run(["gdown", "--id", GDRIVE_ID, "-O", MODEL_PATH], check=True)
        st.success("Model downloaded successfully!")

@st.cache_resource
def load_unet_model():
    download_model()
    model = load_model(MODEL_PATH, custom_objects={'dice_coef': dice_coef, 'dice_loss': dice_loss})
    return model

def preprocess_image(image, target_size=(128, 128)):
    image = image.convert("L").resize(target_size)
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=(0, -1))
    return img_array

def postprocess_mask(mask):
    mask = (mask > 0.5).astype(np.uint8) * 255
    return mask

def main():
    st.title("Brain Tumor Segmentation with U-Net")

    model = load_unet_model()

    uploaded_file = st.file_uploader("Upload a brain MRI image (png, jpg, jpeg)", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        img_array = preprocess_image(image)
        prediction = model.predict(img_array)[0, :, :, 0]

        mask_img = postprocess_mask(prediction)
        st.image(mask_img, caption="Predicted Tumor Mask", use_column_width=True)

if __name__ == "__main__":
    main()
