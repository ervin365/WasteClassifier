import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np

img_height, img_width = 180, 180

# Load your pre-trained model (replace 'model_path' with the actual path)
model = tf.keras.models.load_model('my_model.keras')

class_names = ['Biodegradable_Waste', 'Non_Biodegradable'] 

import streamlit as st
st.title("Image Classification App")

# Upload an image
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_image:
    # Display the uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    img = image.resize((img_height, img_width))
    # Preprocess the image
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    # Make predictions
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])

    # Display the result
    st.write(f"This image most likely belongs to {class_names[predicted_class]} with a {100 * np.max(predictions[0]):.2f}% confidence.")
