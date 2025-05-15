import streamlit as st
import librosa
import numpy as np
import joblib
import os


# Load the Trained Model

@st.cache_resource
def load_model():
    return joblib.load('tess_ser_mlp_model.pkl')

model = load_model()


# Feature Extraction

def extract_features(file_path):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast', duration=3.0, offset=0.5)

        mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T, axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sample_rate).T, axis=0)
        mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=sample_rate).T, axis=0)

        feature_vector = np.hstack((mfccs, chroma, mel))
        return feature_vector

    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None


# Streamlit App UI

st.title("🎤 Speech Emotion Recognition App (TESS Model)")
st.write("Upload a `.wav` audio file and get emotion prediction.")

uploaded_file = st.file_uploader("Upload Audio File (.wav)", type=["wav"])

if uploaded_file is not None:
    file_path = f"uploads/{uploaded_file.name}"
    
    # Ensure uploads directory exists
    os.makedirs('uploads', exist_ok=True)
    
    # Save uploaded file
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.audio(uploaded_file, format='audio/wav')
    
    # Extract Features & Predict
    with st.spinner('Analyzing...'):
        feature_vector = extract_features(file_path)
        if feature_vector is not None:
            prediction = model.predict([feature_vector])[0]
            st.success(f"🎯 Predicted Emotion: **{prediction.upper()}**")

