# MoodSense AI

This is my emotion detection project that recommends Spotify music based on your mood. I built this to learn about NLP and transformers.

## What it does

- Detects emotions from text (joy, sadness, anger, fear, love, surprise)
- Uses DistilBERT model for classification 
- Recommends songs from Spotify based on detected mood
- Web interface with Streamlit

## How to run

1. Clone the repo:
```bash
git clone https://github.com/maheshh-v/MoodSense-AI.git
cd MoodSense-AI
```

2. Install stuff:
```bash
pip install -r requirements.txt
```

3. Add your Spotify API keys to `.env` file:
```
SPOTIFY_CLIENT_ID=your_id_here
SPOTIFY_CLIENT_SECRET=your_secret_here
```

4. Train the model:
```bash
python scripts/train_model.py
```

5. Run the app:
```bash
streamlit run app/streamlit_app.py
```

## Project Structure

```
src/
├── models/          # BERT and LSTM models
├── data/           # Data preprocessing 
├── services/       # Spotify API client
app/                # Streamlit web app
scripts/            # Training script
tests/              # Unit tests
```

## Tech Stack

- TensorFlow & Transformers for ML
- Streamlit for web interface  
- Spotify Web API for music
- Pytest for testing

## Results

The DistilBERT model gets around 85%+ accuracy on emotion classification.

## Author

Mahesh Vyas
- GitHub: [@maheshh-v](https://github.com/maheshh-v)
- LinkedIn: [mahesh-vyas](https://www.linkedin.com/in/mahesh-vyas-88ab41188)

## Credits

Thanks to Hugging Face for the transformers library and Spotify for their API.