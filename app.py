import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, Activation, BatchNormalization
from tensorflow.keras.models import Model
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

def unet(input_size=(128,128,1)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, (3,3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, (3,3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2,2))(conv1)
    conv2 = Conv2D(128, (3,3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, (3,3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2,2))(conv2)
    conv3 = Conv2D(256, (3,3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, (3,3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2,2))(conv3)
    conv4 = Conv2D(512, (3,3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, (3,3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2,2))(conv4)
    conv5 = Conv2D(1024, (3,3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(1024, (3,3), activation='relu', padding='same')(conv5)
    up6 = Conv2DTranspose(512, (2,2), strides=(2,2), padding='same')(conv5)
    up6 = concatenate([up6, conv4])
    conv6 = Conv2D(512, (3,3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(512, (3,3), activation='relu', padding='same')(conv6)
    up7 = Conv2DTranspose(256, (2,2), strides=(2,2), padding='same')(conv6)
    up7 = concatenate([up7, conv3])
    conv7 = Conv2D(256, (3,3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(256, (3,3), activation='relu', padding='same')(conv7)
    up8 = Conv2DTranspose(128, (2,2), strides=(2,2), padding='same')(conv7)
    up8 = concatenate([up8, conv2])
    conv8 = Conv2D(128, (3,3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(128, (3,3), activation='relu', padding='same')(conv8)
    up9 = Conv2DTranspose(64, (2,2), strides=(2,2), padding='same')(conv8)
    up9 = concatenate([up9, conv1])
    conv9 = Conv2D(64, (3,3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(64, (3,3), activation='relu', padding='same')(conv9)
    conv10 = Conv2D(1, (1,1), activation='sigmoid')(conv9)
    model = Model(inputs=inputs, outputs=conv10)
    model.compile(optimizer='adam', loss=dice_loss, metrics=[dice_coef])
    return model

@st.cache_resource
def download_weights():
    if not os.path.exists(MODEL_WEIGHTS_PATH):
        st.info("Downloading model weights from Google Drive...")
        subprocess.run(["pip", "install", "gdown"], check=True)
        subprocess.run(["gdown", "--id", GDRIVE_ID, "-O", MODEL_WEIGHTS_PATH], check=True)
        st.success("Model weights downloaded!")

@st.cache_resource
def load_model_weights():
    download_weights()
    model = unet()
    model.load_weights(MODEL_WEIGHTS_PATH)
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

    model = load_model_weights()

    uploaded_file = st.file_uploader("Upload a brain MRI image", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        img_array = preprocess_image(image)
        prediction = model.predict(img_array)[0, :, :, 0]
        mask_img = postprocess_mask(prediction)
        st.image(mask_img, caption="Predicted Tumor Mask", use_column_width=True)

if __name__ == "__main__":
    main()
