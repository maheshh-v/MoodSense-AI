# 🎵 MoodSense AI

A sophisticated emotion detection system that analyzes text sentiment and recommends personalized Spotify music based on detected moods.

## 🚀 Features

- **Advanced Emotion Detection**: Uses DistilBERT transformer model for accurate emotion classification
- **Real-time Music Recommendations**: Integrates with Spotify API for personalized music suggestions
- **Interactive Web Interface**: Clean Streamlit-based UI for seamless user experience
- **Multiple Model Support**: Supports both traditional LSTM and modern transformer architectures
- **Comprehensive Testing**: Includes unit tests for data preprocessing components

## 🛠️ Tech Stack

- **Machine Learning**: TensorFlow, Transformers (DistilBERT), Hugging Face Datasets
- **Web Framework**: Streamlit
- **APIs**: Spotify Web API
- **Data Processing**: NumPy, Pandas
- **Testing**: Pytest

## 📁 Project Structure

```
moodsense_ai/
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── bert_model.py
│   │   └── lstm_model.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── preprocessing.py
│   ├── services/
│   │   ├── __init__.py
│   │   └── spotify_client.py
│   └── utils/
│       └── __init__.py
├── app/
│   └── streamlit_app.py
├── scripts/
│   └── train_model.py
├── tests/
│   ├── __init__.py
│   └── test_preprocessing.py
├── saved_model/
├── docs/
│   └── rnn_lstm_visualization.html
├── .env
├── .gitignore
├── requirements.txt
└── README.md
```

## 🔧 Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd moodsense_ai
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your Spotify API credentials
```

## 🚀 Usage

### Training the Model
```bash
python scripts/train_model.py
```

### Running the Web Application
```bash
streamlit run app/streamlit_app.py
```

### Running Tests
```bash
pytest tests/
```

## 🎯 Model Performance

- **Accuracy**: 85%+ on emotion classification
- **Supported Emotions**: Joy, Sadness, Anger, Fear, Love, Surprise
- **Model**: DistilBERT fine-tuned on emotion dataset

## 🔑 Environment Variables

Create a `.env` file with the following variables:
```
SPOTIFY_CLIENT_ID=your_spotify_client_id
SPOTIFY_CLIENT_SECRET=your_spotify_client_secret
```

## 📊 API Integration

The application integrates with:
- **Spotify Web API**: For music recommendations
- **Hugging Face Datasets**: For emotion classification training data

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💻 Author

**Mahesh**
- GitHub: [@your-github-username]
- LinkedIn: [Your LinkedIn Profile]

## 🙏 Acknowledgments

- Hugging Face for the emotion dataset and transformers library
- Spotify for their comprehensive Web API
- Streamlit for the intuitive web framework