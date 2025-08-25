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
st.set_page_config(page_title="MoodSense AI", page_icon="🎵", layout="wide")

# Custom CSS for better styling
st.markdown("""
<style>
.mood-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 20px;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin: 10px 0;
}
.song-card {
    background: #f8f9fa;
    padding: 15px;
    border-radius: 8px;
    border-left: 4px solid #1db954;
    margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)

st.title("🎵 MoodSense AI")
st.markdown("### Discover music that matches your emotions! 🎭")
st.markdown("Enter how you're feeling, and I'll analyze your mood and recommend perfect songs from Spotify.")

# Create a more engaging input section
with st.container():
    st.markdown("#### 💭 Share your thoughts:")
    user_input = st.text_area(
        "How are you feeling today?", 
        "I had a wonderful day, everything went perfectly.",
        height=100,
        help="Describe your current mood, feelings, or what happened in your day"
    )

if st.button("Recommend Music"):
    if user_input:
        with st.spinner("Analyzing mood and fetching songs..."):
            # Predict mood
            predicted_mood, confidence = predict_mood(user_input, model, tokenizer)

            if predicted_mood:
                # Create a beautiful mood display card
                mood_emoji = {
                    "joy": "😊", "sadness": "😢", "love": "❤️", 
                    "anger": "😠", "fear": "😨", "surprise": "😲"
                }
                
                st.markdown(f"""
                <div class="mood-card">
                    <h2>{mood_emoji.get(predicted_mood, "🎭")} {predicted_mood.capitalize()}</h2>
                    <p>Confidence: {confidence:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Get recommendations
                recommended_tracks = get_top_songs_for_mood(predicted_mood)
                
                st.subheader("🎵 Recommended Songs for You:")
                if recommended_tracks and recommended_tracks[0]['title'] != "No Song Found":
                    for i, track in enumerate(recommended_tracks, 1):
                        # Create a nice card-like display for each song
                        with st.container():
                            col1, col2 = st.columns([3, 1])
                            
                            with col1:
                                st.markdown(f"**{i}. {track['title']}**")
                                # Extract Spotify track ID for embedding
                                track_id = track['url'].split('/')[-1].split('?')[0]
                                
                                # Show embedded Spotify player
                                spotify_embed = f"""
                                <iframe src="https://open.spotify.com/embed/track/{track_id}" 
                                        width="100%" height="152" frameborder="0" 
                                        allowtransparency="true" allow="encrypted-media">
                                </iframe>
                                """
                                st.components.v1.html(spotify_embed, height=160)
                            
                            with col2:
                                st.markdown("")
                                st.markdown(f"[🎧 Open in Spotify]({track['url']})")
                            
                            st.markdown("---")  # Separator line
                else:
                    st.warning("Could not fetch song recommendations at the moment.")
    else:
        st.warning("Please enter a sentence.")