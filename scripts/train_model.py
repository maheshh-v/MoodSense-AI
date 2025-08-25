import tensorflow as tf
import os
import sys
from transformers import AutoTokenizer

# Add src directory to path - took me a while to figure this out
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from data.preprocessing import prepare_data_for_bert
from models.bert_model import build_bert_model

# Setup model directory
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'saved_model')
os.makedirs(MODEL_DIR, exist_ok=True)

# Load and prepare data - this takes a while
print("Loading and preparing data for BERT...")
train_ds, val_ds, test_ds = prepare_data_for_bert()
print("Data preparation complete")

# Build the model
print("Building DistilBERT model...")
bert_model = build_bert_model()
print("Model build complete.")

# Compile with Adam optimizer
# tried different learning rates, 3e-5 works best
optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metrics = ['accuracy']
bert_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
print("Model compilation complete")

# Start training - this is the slow part
print("Starting model fine-tuning")
history = bert_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=3  # 3 epochs seems to work well, more might overfit
)
print("Model fine tuning complete")

# Test how good our model is
print("Evaluating model performance on the test set..")
loss, accuracy = bert_model.evaluate(test_ds)
print(f"Test Accuracy: {accuracy*100:.2f}%")

# Save everything so we can use it later
print(f"Saving model and tokenizer to {MODEL_DIR}...")
bert_model.save_pretrained(MODEL_DIR)

tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
tokenizer.save_pretrained(MODEL_DIR)
print("Save complete")