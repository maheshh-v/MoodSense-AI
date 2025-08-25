import tensorflow as tf
import os
import sys
from transformers import AutoTokenizer

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from data.preprocessing import prepare_data_for_bert
from models.bert_model import build_bert_model

# 1 Setup 
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'saved_model')
os.makedirs(MODEL_DIR, exist_ok=True)

# 2 Load and Prepare Data for BERT 
print("Loading and preparing data for BERT...")
train_ds, val_ds, test_ds = prepare_data_for_bert()
print("Data preparation complete")

#  3. Build the BERT Model 
print("Building DistilBERT model...")
bert_model = build_bert_model()
print("Model build complete.")

#  4 Compile the Model 
optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5) # Slightly lower learning rate is often better
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metrics = ['accuracy']
bert_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
print("Model compilation complete")

# 5. Fine tuning
print("Starting model fine-tuning")
history = bert_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=3
)
print("Model fine tuning complete")

# 6. Evaluate the Model 
print("Evaluating model performance on the test set..")
loss, accuracy = bert_model.evaluate(test_ds)
print(f"Test Accuracy: {accuracy*100:.2f}%")

#  7. Save the Fine-Tuned Model and Tokenizer ---
print(f"Saving model and tokenizer to {MODEL_DIR}...")
bert_model.save_pretrained(MODEL_DIR)

tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
tokenizer.save_pretrained(MODEL_DIR)
print("Save complete")