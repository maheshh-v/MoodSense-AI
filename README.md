🎵 MoodSense AI
A high-performance emotion detection system that analyzes text and recommends personalized Spotify music. This project fine-tunes a DistilBERT Transformer model to achieve over 92% accuracy.

Live Demo
(It's highly recommended to add a GIF of your Streamlit app in action here. You can use a tool like Giphy Capture or ScreenToGif.)

Features
State-of-the-Art Emotion Detection: Fine-tunes a DistilBERT Transformer model for highly accurate emotion classification.

Real-time Music Recommendations: Integrates directly with the Spotify Web API for relevant, mood-based song suggestions.

Interactive Web Interface: A clean and intuitive user interface built with Streamlit.

Proven Performance: Achieves 92.95% accuracy on the test dataset.

🛠️ Tech Stack
Machine Learning: TensorFlow, Hugging Face (Transformers, Datasets)

Web Framework: Streamlit

API Integration: Spotify Web API (via Spotipy)

Data Processing: NumPy, Pandas

Code Quality: Pytest for unit testing

📁 Project Structure
A clean, modular structure for maintainability and clarity.

moodsense_ai/
├── app.py                 # The main Streamlit application file
├── main.py                # Script to train and save the model
├── preprocessing.py       # Data loading and preparation functions
├── model.py               # Model architecture functions (BERT/LSTM)
├── spotify_client.py      # Handles all Spotify API interactions
├── test_preprocessing.py  # Unit tests for the preprocessing functions
├── saved_model/           # Directory for the saved trained model & tokenizer
├── .env                   # File for environment variables (API keys)
├── requirements.txt       # Project dependencies
└── README.md              # You are here!

🔧 Installation & Setup
Clone the repository:

git clone https://github.com/maheshh-v/moodsense_ai.git
cd moodsense_ai

Create and activate a virtual environment:

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt

Set up environment variables: Create a .env file in the root directory and add your Spotify API credentials:

SPOTIFY_CLIENT_ID="your_spotify_client_id"
SPOTIFY_CLIENT_SECRET="your_spotify_client_secret"

 Usage
Train the Model: First, run the main script to fine-tune and save the BERT model.

python main.py

Run the Web Application: Once the model is saved, launch the Streamlit app.

streamlit run app.py

Run Tests (Optional): To verify the data preprocessing logic, run the unit tests.

pytest

 Model Performance
Model: DistilBERT (fine-tuned)

Accuracy: 92.95% on the test set

Emotions Classified: Joy, Sadness, Anger, Fear, Love, Surprise

🤝 Contributing
Contributions are welcome! Please feel free to fork the repository, create a feature branch, and open a Pull Request.

Fork the Project

Create your Feature Branch (git checkout -b feature/AmazingFeature)

Commit your Changes (git commit -m 'Add some AmazingFeature')

Push to the Branch (git push origin feature/AmazingFeature)

Open a Pull Request


👨‍💻 Author
Mahesh Vyas

GitHub: @maheshh-v

LinkedIn: www.linkedin.com/in/mahesh-vyas-88ab41188

🙏 Acknowledgments
Hugging Face for their incredible transformers and datasets libraries.

Spotify for providing a feature-rich Web API.

The Streamlit team for making web app development in Python so accessible.