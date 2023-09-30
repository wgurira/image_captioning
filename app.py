import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import pickle
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Dense, Embedding, LSTM, Reshape, concatenate, Dropout, add

# Load your model
model = load_model('ficmodel.h5')

# Load your tokenizer
with open('fictoken.pk1', 'rb') as tokenizer_file:
    tokenizer = pickle.load(tokenizer_file)

# Load your feature extraction model (fe)
fe = DenseNet201(weights='imagenet')
img_size = 224

# Function to preprocess and generate captions
def generate_caption(image):
    img = load_img(image, target_size=(img_size, img_size))
    img = img_to_array(img)
    img = img / 255.
    img = np.expand_dims(img, axis=0)

    # Extract image features
    feature = fe.predict(img, verbose=0)

    # Generate caption
    in_text = 'startseq'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([feature, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = word_for_id(yhat, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break

    caption = in_text.split()[1:-1]  # Remove 'startseq' and 'endseq'
    return ' '.join(caption)

# Function to check the file size
def check_file_size(file, max_size):
    if file is None:
        return True
    if len(file.read()) > max_size:
        return False
    return True

# Streamlit app
st.title("R204433P Image Captioning Model")

# Upload an image with a size limit of 2MB
image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if image:
    # Check the file size
    max_size = 2 * 1024 * 1024  # 2MB limit
    if check_file_size(image, max_size):
        # Display the uploaded image
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Generate caption when the user clicks the button
        if st.button("Generate Caption"):
            try:
                # Generate and display the caption
                caption = generate_caption(image)
                st.write("Generated Caption:", caption)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    else:
        st.error("File size exceeds the 2MB limit.")
