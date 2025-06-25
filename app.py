import os
import numpy as np
import librosa
from keras.models import load_model
import streamlit as st
import tempfile
import streamlit.components.v1 as components

# Load trained model
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, 'Emotion_Audio.h5')
emotion_classifier = load_model(model_path, compile=False)

# Emotion labels
emotion_labels = ['angry', 'calm', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Feature extraction function
def extract_features(data, sample_rate):
    result = np.array([])
    result = np.hstack((result, np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)))
    stft = np.abs(librosa.stft(data))
    result = np.hstack((result, np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)))
    result = np.hstack((result, np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)))
    result = np.hstack((result, np.mean(librosa.feature.rms(y=data).T, axis=0)))
    result = np.hstack((result, np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)))
    return result

# Predict emotion
def predict_emotion(file_path):
    data, sr = librosa.load(file_path, sr=44100)
    features = extract_features(data, sr)
    features_resized = np.resize(features, (162, 1))
    input_data = np.reshape(features_resized, (1, 162, 1))
    predictions = emotion_classifier(input_data)[0]
    dominant_index = np.argmax(predictions)
    return emotion_labels[dominant_index], predictions

# Streamlit Page Config
st.set_page_config(page_title="🔊 Emotion Detection", layout="centered")

# Custom CSS
st.markdown("""
<style>
html, body {
    background: linear-gradient(to right, #f3e5f5, #ede7f6);
    font-family: 'Segoe UI', sans-serif;
}
h1, h2, h3 {
    color: #6a1b9a;
    text-align: center;
}
.stButton>button {
    background-color: #8e24aa;
    color: white;
    border-radius: 8px;
    height: 3em;
    font-size: 16px;
    font-weight: bold;
}
.stButton>button:hover {
    background-color: #6a1b9a;
}
.stFileUploader {
    border: 2px dashed #ba68c8;
    padding: 1.5em;
    border-radius: 10px;
    background-color: #f3e5f5;
    text-align: center;
    color: #4a148c;
}
.audio-box {
    text-align: center;
    margin-top: 1em;
}
.result-card {
    margin-top: 2em;
    background-color: #ede7f6;
    padding: 2em;
    border-radius: 15px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# Page UI
st.title("🎤 Audio Emotion Analyzer")
st.markdown("##### Upload a `.wav` audio file to analyze the speaker's emotion.")

audio_file = st.file_uploader("🎵 Drop WAV file here or click to browse", type=["wav"])
submit = st.button("✨ Analyze")

if audio_file and submit:
    with st.spinner("Analyzing... please wait."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_file.read())
            temp_path = tmp.name

        emotion, _ = predict_emotion(temp_path)

        st.markdown(f"""
        <div class="result-card">
            <h2>🧠 Detected Emotion</h2>
            <h1 style="font-size: 3em;">{emotion.upper()}</h1>
        </div>
        """, unsafe_allow_html=True)

        st.audio(audio_file, format='audio/wav')

        components.html("""
        <script>
        const anchor = window.parent.document.getElementsByClassName("result-card")[0];
        if(anchor) {
            anchor.scrollIntoView({behavior: 'smooth'});
        }
        </script>
        """, height=0)
