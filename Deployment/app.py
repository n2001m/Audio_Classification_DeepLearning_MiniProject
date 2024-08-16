from flask import Flask, request, jsonify, render_template
import numpy as np
import librosa
import tensorflow as tf

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('modelFullData.h5')

# Define the label mapping
label_mapping = {0: 'siren', 1: 'traffic', 2: 'others'}

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
    spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=6, fmin=50).T, axis=0)
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr).T, axis=0)
    feature_vector = np.hstack([mfccs, chroma, mel, spectral_contrast, spectral_centroid])
    return feature_vector

@app.route('/')
def index():
    return render_template('W4Project.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    # Save the uploaded file
    file_path = "uploaded_file.wav"
    file.save(file_path)

    # Extract features
    features = extract_features(file_path).reshape(1, -1)

    # Make prediction
    prediction = model.predict(features)
    predicted_label = label_mapping[np.argmax(prediction)]

    # Generate result based on the predicted label
    if predicted_label == 'siren':
        result = 'Warning: Siren Detected!'
    elif predicted_label == 'traffic':
        result = 'Traffic Noise Detected!'
    else:
        result = 'Normal: No Siren or Traffic Noise Detected.'

    return jsonify({'result': result})

if __name__ == "__main__":
    app.run(debug=True)
