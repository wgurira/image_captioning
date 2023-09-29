import streamlit as st
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import pickle
import numpy as np
import cv2

# Load the image captioning model and tokenizer
caption_model = load_model("myicmodel.h5")
tokenizer = pickle.load(open("myictokenizer.pkl", "rb"))

# Function to preprocess frames and generate captions
def preprocess_frame(frame):
    # Preprocess the frame (resize, normalize, etc.)
    # ...

    # Generate caption using the image captioning model
    # ...

    # Return the generated caption
    # ...

# Function to generate next word predictions
def generate_predictions(seed_text, model, tokenizer, max_sequence_len=8, next_words=5):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted_probs = model.predict(token_list)
        predicted_word = tokenizer.index_word[np.argmax(predicted_probs)]
        seed_text += " " + predicted_word
    return seed_text

# Streamlit UI
st.title("Video Captioning and Word Prediction App")

# Allow user to upload a video file
video_file = st.file_uploader("Upload a video (max 2MB)", type=["mp4"])

if video_file:
    try:
        # Read the video file using OpenCV
        video = cv2.VideoCapture(video_file)

        # Iterate over each frame
        while True:
            # Read the next frame
            ret, frame = video.read()

            if not ret:
                break

            # Preprocess the frame and generate caption
            caption = preprocess_frame(frame)

            # Generate word predictions based on caption
            predictions = generate_predictions(caption, caption_model, tokenizer)

            # Display the frame and its caption with word predictions
            st.image(frame, caption=f"Caption: {caption}")
            st.write("Word Predictions:", predictions)

    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.warning("Please upload a video file.")
