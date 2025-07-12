# ui/app.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import numpy as np
import pickle
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from src.preprocessing import clean_text  # Ensure this function is implemented

# --- Paths ---
MODEL_PATH = "../saved_model/mood_lstm_glove"
TOKENIZER_PATH = '../saved_model/mood_lstm_glove/tokenizer_glove.pkl'
LABEL_ENCODER_PATH = '../saved_model/mood_lstm_glove/label_map_glove.pkl'
MOOD_MAP_PATH = '../mood_to_music.json'

MAX_LEN = 35  # Keep same as training

# --- Load model and assets ---
model = load_model(MODEL_PATH)

with open(TOKENIZER_PATH, 'rb') as handle:
    tokenizer = pickle.load(handle)

with open(LABEL_ENCODER_PATH, 'rb') as handle:
    label_names = pickle.load(handle)

with open(MOOD_MAP_PATH, 'r') as f:
    mood_to_music = json.load(f)

# --- Mood Prediction Function ---
def predict_mood(text):
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post')
    pred = model.predict(padded)[0]
    top_2 = pred.argsort()[-2:][::-1]
    return [(label_names[i], round(pred[i] * 100, 2)) for i in top_2]

# --- Streamlit UI ---
st.set_page_config(page_title="Mood Music App", layout="centered")
st.title("🎵 Mood Detector + Music Recommender")

user_input = st.text_area("What's on your mind?", "")

if st.button("Detect Mood"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        top_moods = predict_mood(user_input)
        primary_mood = top_moods[0][0]
        st.success(f"🎯 Predicted Mood: **{primary_mood}**")

        st.markdown("#### 🔍 Mood Probabilities")
        for mood, prob in top_moods:
            st.write(f"- {mood}: {prob}%")

        st.markdown("#### 🎧 Top 3 Music Suggestions:")
        if primary_mood in mood_to_music:
            for i, song in enumerate(mood_to_music[primary_mood][:3]):
                st.markdown(f"{i + 1}. [{song['title']}]({song['url']})")
        else:
            st.write("No music suggestions found for this mood.")

# --- About Section ---
with st.expander("ℹ️ About This App", expanded=True):
    st.markdown("""
    **🎯 Mood Detection + Music Recommendation App**

    This AI tool analyzes your input text and predicts your emotional mood using a custom-trained LSTM model with GloVe embeddings.

    ---
    🛠️ **Current Version**:
    - Model: LSTM with GloVe (trained on emotion-labeled dataset)
    - Suggests 3 songs based on detected mood
    - Accuracy may vary; still improving!

    ---
    🚀 **Upcoming Enhancements**:
    - Switch to a BERT-based transformer model for deeper contextual understanding
    - Fine-tune on a larger, more diverse dataset
    - Smarter and more personalized music suggestions using Spotify API
    - Real-time mood tracking via voice/text input

    ---
    🤖 This app is part of an AI developer portfolio project — not a final product. Feedback is welcome!
    """)
