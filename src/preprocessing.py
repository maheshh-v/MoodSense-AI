# src/preprocessing.py

import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # remove punctuation
    text = re.sub(r'\s+', ' ', text).strip() # remove extra spaces
    return text
