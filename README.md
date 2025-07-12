# AI-Based Mood Detection from Text with Music Suggestion 🎧

## Overview

This project uses deep learning (TensorFlow + Keras) to detect a person's mood from input text and recommend relevant music. It combines NLP, sequence modeling (LSTM), word embeddings (GloVe), and plans for Spotify API integration.

This is not just a one-time model — it's a **developing real-world AI app** built with clear vision: enabling intelligent emotional recognition from natural language and delivering personalized music feedback.

---

## 🔬 Problem Statement

Can we understand someone's emotion from their words? And if we can — can we respond empathetically?

This project answers:  
> “Given any user-written sentence, detect their underlying emotion and recommend relevant music.”

---

##  Key Features

- **Multi-class Emotion Classification** — (6 classes): `joy`, `sadness`, `anger`, `fear`, `love`, `surprise`
- **Two NLP Models Built from Scratch:**
  1. Baseline LSTM using `Embedding + LSTM + Dense`
  2. Advanced LSTM with `GloVe embeddings + Dropout regularization + Class weight tuning`
- **Per-class Accuracy & Confusion Matrix Evaluation**
- **Top-2 Predicted Moods with Confidence** — Enhances UX & explains model behavior
- **Music Recommendation Logic** using mood-to-song mapping
- **Plans for Live Spotify Integration** to fetch real songs from Spotify dynamically
- **Deployable Web App (in development)** — Streamlit-based UI to test mood + hear music

---

## 💼 Why This Project Stands Out

This is not a typical “run and done” ML notebook. This project is:

- ⚙ **Manually engineered from data cleaning to evaluation** (no copy-paste pipelines)
-  Every step (tokenization, embedding, padding, loss choice, overfitting checks, class balancing, evaluation) was carefully tested and improved
-  **Two trained models** were compared and saved professionally
-  Uses real GloVe word embeddings instead of shallow or random vectors
-  Critical thinking applied at each decision point, from LSTM use to dropout and model selection

---

## 🔗 App Demo Vision (In Development)

The upcoming deployable app will allow users to:
1. Enter a sentence (e.g., "I miss my best friend")
2. Instantly detect the top 2 mood predictions with confidence
3. Hear a matching playlist directly through **Spotify integration** (via official API)

> *Goal: Turn this into a professional, interactive portfolio app with Spotify-powered mood-based music suggestion.*

---

##  Model Performance

**Best Model: GloVe + Dropout + Class Weights**

- **Test Accuracy:** ~88.7%
- **Per-Class Accuracy**:

  sadness : 0.93

  joy : 0.85

  love : 0.88

  anger : 0.84

  fear : 0.73

   surprise : 0.91


- **Confusion Matrix Sample:**
![confusion-matrix](/notebooks/confusion_matrix.png) *(Confusion Matrix – GloVe + Dropout
Nails sadness and love, but mixes up close ones like joy vs. love and fear vs. surprise. Next up: smarter BERT-based model)*

---

## 🛠 Tech Stack

- Python (3.10)
- TensorFlow / Keras
- Pandas / NumPy / Sklearn
- GloVe Word Embeddings
- Streamlit (in progress)
- Spotify Web API (to be integrated)


---

## 📌 What's Next

-  **Spotify API Integration**: Fetch top 3 songs live using user's mood
- **Deploy Full Web App** (Streamlit + Hosted link for recruiters)
-  **Upgrade to BERT/DistilBERT**: For sentence-level semantic detection
- 🎵 **Auto-play music from dashboard**: Using audio player (Spotify embed)

---

##  Author

**Mahesh Vyas** — 4th-year CS (AI), passionate about deploying human-like AI systems that blend empathy and logic.  
Actively seeking AI/ML Internship roles where learning meets impact.

---

##  How to Use (Once Deployed)

1. Type any sentence: _"I just feel like crying today..."_
2. The model predicts mood: `sadness` (91%) + `fear` (5%)
3. System suggests songs: _"Relaxing Piano", "Sad Acoustic"_
4. With Spotify API → you can listen right there.

---

##  Recruiter Notes

> _“This project showcases strong practical understanding of deep learning, critical thinking in design, thoughtful evaluation, and real deployment mindset — not just copy-paste ML.”_


