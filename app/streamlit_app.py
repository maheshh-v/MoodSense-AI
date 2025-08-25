import streamlit as st
import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
import numpy as np
import sys
import os

# Adding src folder to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))


from services.spotify_client import get_top_songs_for_mood

# --- Configuration 
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'saved_model')

# --- Caching Models for Performance
@st.cache_resource
def load_model_and_tokenizer():
    """Load the saved model and tokenizer from the directory  """
    try:
        model = TFAutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {e}. Make sure you have run main py to train and save the model first")
        return None, None

# --- Loading Assets 
model, tokenizer = load_model_and_tokenizer()

# --- Prediction Function ---
def predict_mood(sentence, model, tokenizer):
    """Predict the mood of a single sentence."""
    if model is None or tokenizer is None:
        return None

    # The label mapping from the dataset
    label_map = {0: "sadness", 1: "joy", 2: "love", 3: "anger", 4: "fear", 5: "surprise"}

    inputs = tokenizer(sentence, return_tensors="tf", truncation=True, padding=True)
    logits = model(**inputs).logits

    # apply softmax to convert logits to probabilities
    probabilities = tf.nn.softmax(logits, axis=-1).numpy()[0]
    pred_class_index = np.argmax(probabilities)
    
    predicted_mood = label_map[pred_class_index]
    confidence = probabilities[pred_class_index]
    
    return predicted_mood, confidence

# --- Streamlit App Interface ---
st.set_page_config(page_title="MoodSense AI", page_icon="🎵", layout="centered")
st.title("🎵 MoodSense AI")
st.markdown("Enter a sentence, and I'll detect the mood and recommend some songs from Spotify!")

user_input = st.text_area("How are you feeling today?", "I had a wonderful day, everything went perfectly.")

if st.button("Recommend Music"):
    if user_input:
        with st.spinner("Analyzing mood and fetching songs..."):
            # Predict mood
            predicted_mood, confidence = predict_mood(user_input, model, tokenizer)

            if predicted_mood:
                st.success(f"**Predicted Mood: {predicted_mood.capitalize()}** (Confidence: {confidence:.2%})")
                
                # Get recommendations
                recommended_tracks = get_top_songs_for_mood(predicted_mood)
                
                st.subheader("Here are some tracks for you:")
                if recommended_tracks and recommended_tracks[0]['title'] != "No Song Found":
                    for track in recommended_tracks:
                        st.markdown(f"[{track['title']}]({track['url']})")
                else:
                    st.warning("Could not fetch song recommendations at the moment.")
    else:
        st.warning("Please enter a sentence.")