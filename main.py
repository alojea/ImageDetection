import streamlit as st
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load pre-trained model
model = MobileNetV2(weights='imagenet')

st.title("🐶 Animal Classifier")
st.write("Upload an image and I will tell you which animal it is!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image.', use_column_width=True)

    # Preprocess
    img_resized = img.resize((224, 224))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Predict
    preds = model.predict(img_array)
    decoded = decode_predictions(preds, top=3)[0]

    st.subheader("Prediction:")
    for i, (imagenet_id, label, prob) in enumerate(decoded):
        st.write(f"**{label}**: {prob*100:.2f}%")