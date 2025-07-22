

from transformers import BertTokenizer, BertForSequenceClassification
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
import os

load_dotenv()

# Load model and tokenizer
model_path = "saved_model/bert_emotion_model"
bert_tokenizer = BertTokenizer.from_pretrained(model_path)
bert_model = BertForSequenceClassification.from_pretrained(model_path)
bert_model.eval()


def predict_emotion(text):
    inputs = bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)

    with torch.no_grad():
        outputs = bert_model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()

    emotion_labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
    return emotion_labels[predicted_class], probs.squeeze().tolist()


