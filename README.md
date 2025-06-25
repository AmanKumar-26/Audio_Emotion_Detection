# 🎵 Audio Emotion Detection using Deep Learning

This project detects human emotions from voice recordings using a deep learning model trained on audio features. It supports both a **web UI (Streamlit)** and a **REST API (Flask-based)** for flexibility.

---

## 📌 Features

- 🎙 Upload `.wav` audio to detect speaker emotion
- 🧠 Uses MFCC, chroma, ZCR, RMS, and mel-spectrogram for feature extraction
- 🧪 Includes interactive Jupyter Notebook for experimentation
- 🖥 Streamlit-based UI for real-time analysis
- 🔗 Flask-based API for integration

---

## 🚀 Demo
![Screenshot 2025-06-25 162543](https://github.com/user-attachments/assets/fc2a5ac9-0a2e-411a-9ab7-51dd53dc94d0)


---

## 📁 File Overview

| File                     | Description                              |
|--------------------------|------------------------------------------|
| `app.py`                 | Streamlit app for UI-based emotion detection |
| `classifier.py`          | Flask-compatible backend classifier        |
| `audio-emotion-detection.ipynb` | Notebook for model training/testing      |
| `Emotion_Audio.h5`       | Pre-trained Keras model (required to run) |

---

## 🛠️ Installation

1. Clone the repo:
   ```bash
   git clone https://github.com/AmanKumar-26/Audio_Emotion_Detection.git
   cd Audio_Emotion_Detection
2. Install requirements:
    pip install -r requirements.txt
3. Run Streamlit app:
    streamlit run app.py
4. Or run Flask API:
    python classifier.py

## 💡 Emotions Detected

- Angry  
- Calm  
- Disgust  
- Fear  
- Happy  
- Neutral  
- Sad  
- Surprise

## 🖼️ Suggested Assets

- `assets/demo-ui.png` – Screenshot of the Streamlit user interface
- `assets/emotion-diagram.png` – Optional: Diagram of the audio-to-emotion detection workflow
- Example `.wav` audio files – For demo or testing purposes


## ✨ Acknowledgements

- [Librosa](https://librosa.org/) – For audio signal processing
- [Keras](https://keras.io/) – Deep learning model framework
- [Streamlit](https://streamlit.io/) – UI for interactive web app
- [RAVDESS Dataset](https://zenodo.org/record/1188976) – Used for training and evaluation

## 📜 License

- This project is licensed under the **MIT License**.  
- See the [LICENSE](LICENSE) file for details.











      

