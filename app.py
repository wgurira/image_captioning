import streamlit as st
from keras.models import load_model
import numpy as np
from keras.applications import ResNet50
from keras.preprocessing import image, sequence
from keras.preprocessing.sequence import pad_sequences
import cv2

vocab = np.load('mine_vocab.npy', allow_pickle=True)
vocab = vocab.item()
inv_vocab = {v: k for k, v in vocab.items()}

embedding_size = 128
vocab_size = len(vocab)
max_len = 40

image_model = Sequential()
image_model.add(Dense(embedding_size, input_shape=(2048,), activation='relu'))
image_model.add(RepeatVector(max_len))

language_model = Sequential()
language_model.add(Embedding(input_dim=vocab_size, output_dim=embedding_size, input_length=max_len))
language_model.add(LSTM(256, return_sequences=True))
language_model.add(TimeDistributed(Dense(embedding_size)))

conca = Concatenate()([image_model.output, language_model.output])
x = LSTM(128, return_sequences=True)(conca)
x = LSTM(512, return_sequences=False)(x)
x = Dense(vocab_size)(x)
out = Activation('softmax')(x)
model = Model(inputs=[image_model.input, language_model.input], outputs=out)

model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])
model.load_weights('myicmodel.h5')

resnet = DenseNet201(include_top=False,weights='imagenet',input_shape=(224,224,3),pooling='avg')

st.title("Image Captioning App")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        image = image.load_img(uploaded_file, target_size=(224, 224))
        img_array = image.img_to_array(image)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        incept = resnet.predict(img_array).reshape(1, 2048)

        text_in = ['startofseq']
        final = ''
        count = 0

        while count < 20:
            count += 1

            encoded = [vocab[i] for i in text_in]
            padded = pad_sequences([encoded], maxlen=max_len, padding='post', truncating='post').reshape(1, max_len)

            sampled_index = np.argmax(model.predict([incept, padded]))
            sampled_word = inv_vocab[sampled_index]

            if sampled_word != 'endofseq':
                final = final + ' ' + sampled_word

            text_in.append(sampled_word)

        st.write("Image Caption:", final)

    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.warning("Please upload an image.")
