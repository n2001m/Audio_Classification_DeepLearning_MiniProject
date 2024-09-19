import streamlit as st
import numpy as np
import librosa
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('Deployment/model.h5')

# Define the label mapping
label_mapping = {0: 'others', 1: 'siren', 2: 'traffic'}

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
    spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=6, fmin=50).T, axis=0)
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr).T, axis=0)
    feature_vector = np.hstack([mfccs, chroma, mel, spectral_contrast, spectral_centroid])
    return feature_vector

# Streamlit app interface
st.title('Audio Classification App')

# File uploader
uploaded_file = st.file_uploader("Choose an audio file...", type=["wav"])

if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    file_path = "uploaded_file.wav"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Extract features from the audio file
    features = extract_features(file_path).reshape(1, -1)

    # Make prediction
    prediction = model.predict(features)
    predicted_label = label_mapping[np.argmax(prediction)]

    # Display result
    if predicted_label == 'siren':
        st.warning('Warning: Siren Detected!')
    elif predicted_label == 'traffic':
        st.info('Traffic Noise Detected!')
    else:
        st.success('Normal: No Siren or Traffic Noise Detected.')
