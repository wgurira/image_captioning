import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import pickle

# Load the image captioning model and tokenizer
caption_model = load_model('myicmodel.h5')
tokenizer = pickle.load(open('myictokenizer.pkl', 'rb'))

# Define function to preprocess frames and generate captions
def preprocess_frame(frame):
    # Preprocess the frame (resize, normalize, etc.)
    # ...

    # Generate caption using the image captioning model
    # ...

    # Return the generated caption
    # ...

# Streamlit UI
st.title("Video Captioning App")

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

            # Display the frame and its caption
            st.image(frame, caption=caption)

    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.warning("Please upload a video file.")
