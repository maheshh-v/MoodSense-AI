
import json
import sys
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from dotenv import load_dotenv
load_dotenv()


from bert_inference import predict_emotion

from src.spotify_helper import get_top_songs_for_mood

import streamlit as st

# -Streamlit UI Setup 
st.set_page_config(page_title="MoodSense AI", layout="centered")
st.title(" MoodSense AI – Emotion Detection with BERT")


user_input = st.text_area(" What's on your mind?", "")
#Emotion Prediction Button ---
if st.button("Detect Emotion"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        predicted_emotion, probabilities = predict_emotion(user_input)
        st.success(f"Predicted Emotion: **{predicted_emotion.upper()}**")

      
        st.subheader("Emotion Confidence Scores")
        labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
        for label, prob in zip(labels, probabilities):
            st.write(f"- **{label}**: {prob:.2f}")



        #Music Suggestions 
        st.subheader("🎧 Real-time Spotify Songs")
        try:
            songs = get_top_songs_for_mood(predicted_emotion)


            for i, song in enumerate(songs):
                st.markdown(f"**{i+1}. {song['title']}**")

                # track ID from the Spotify URL -to extract the track ID
                try:
                    track_id = song['url'].split("/")[-1].split("?")[0]
                    embed_url = f"https://open.spotify.com/embed/track/{track_id}"
                    st.components.v1.iframe(embed_url, height=80)
                except:
                    st.warning("Could not load embedded player.")




        except Exception as e:
            st.warning("Could not fetch real songs. Falling back to mock suggestions.")


# About Section 
with st.expander("About This App", expanded=True):
    st.markdown("""
    **BERT-Powered Emotion Classifier**

    This AI tool uses a fine-tuned BERT transformer model to detect emotional tone from natural language text.

    ---
     **Current Capabilities**:
    - High-accuracy emotion detection using pre-trained BERT
    - 6 emotion classes: sadness, joy, love, anger, fear, surprise
    - Results powered by real deep learning — not keyword matching

    ---
     **Upcoming Additions**:
    - Switchable model comparison (GloVe vs BERT)
    - Better explainability & confidence visualization

    ---
    """)
