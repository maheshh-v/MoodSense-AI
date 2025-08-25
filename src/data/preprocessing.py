from seaborn import load_dataset
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import re


def clean_text(text):
    """
    Clean a text string by:
    - Lowercasing
    - Removing punctuation
    - Removing extra spaces

    Args:
        text (string): Input raw text.

    Returns:
        string: Cleaned text.
    """


    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # FOR removing punctuatition
    text = re.sub(r'\s+', ' ', text).strip() #removing unwanted gaps
    return text



def load_and_prepare_data():
    """
    Load and preprocess the 'emotion' dataset for text classification.

    This function:
    - Loads the dataset from Hugging Face
    - Cleans the text (lowercasing, removing punctuation and extra spaces). : uses clean_text function
    - Tokenizes text into sequences using a Keras Tokenizer.
    - Pads sequences to a fixed maximum length (95th percentile of training lengths).
    - Converts labels to NumPy arrays.
    - Prepares metadata (vocabulary size, max sequence length).

    Returns:
        dict: A dictionary with the following keys:
            - "train": (np.ndarray, np.ndarray), padded sequences and labels for training.
            - "vali": (np.ndarray, np.ndarray), padded sequences and labels for validation.
            - "test": (np.ndarray, np.ndarray), padded sequences and labels for testing.
            - "meta": dict with 'vocab_size' (int) and 'max_len' (int).
    """

    from datasets import load_dataset
    ds = load_dataset("emotion")

    train_text =[clean_text(t) for t in ds['train']['text']]
    vali_text = [clean_text(t) for t in ds['validation']['text']]
    test_text = [clean_text(t) for t in ds['test']['text']]


    #Initializing the Keras Tokenizer.
    tokenizer = Tokenizer(oov_token='<OOV>')

    tokenizer.fit_on_texts(train_text)

    train_sequences = tokenizer.texts_to_sequences(train_text)
    vali_sequences = tokenizer.texts_to_sequences(vali_text)
    test_sequences = tokenizer.texts_to_sequences(test_text)


    # Padding the sequences to get all the sequences of same length.
    seq_length = [len(s) for s in train_sequences]

    max_length = int(np.percentile(seq_length, 95))  
    print(f'Max length of the sequences: {max_length}')

    padded_train= pad_sequences(train_sequences, maxlen=max_length, padding='pre')
    padded_vali = pad_sequences(vali_sequences, maxlen=max_length, padding='pre')
    padded_test = pad_sequences(test_sequences, maxlen=max_length, padding='pre')
    

    # Converting the labels into numpy arrays
    train_labels = np.array(ds['train']['label'])   
    vali_labels = np.array(ds['validation']['label'])
    test_labels = np.array(ds['test']['label'])

    #vocab size
    vocab_size = len(tokenizer.word_index) + 1  # +1 for padding token

    
    return {
        "train": (padded_train, train_labels),
        "vali": (padded_vali, vali_labels),
        "test": (padded_test, test_labels),
        "meta": {
            "vocab_size": vocab_size,
            "max_len": max_length  #also converting float value to int
        }
    }



from transformers import AutoTokenizer, DataCollatorWithPadding
from datasets import load_dataset

def prepare_data_for_bert():
    """
    Load and preprocess the emotion dataset for a BERT-style model.
    This function manually restructures the dataset into the (inputs, labels)
    tuple format required by Keras.
    """
    # 1. Load the dataset and tokenizer
    ds = load_dataset("emotion")
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

    # 2. Tokenize the text
    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True)
    tokenized_datasets = ds.map(tokenize_function, batched=True)

    # 3.Prepare columns: Remove 'text', rename 'label' to 'labels'
    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

    # 4. create a Data Collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")

    # 5 create initial datasets that yield dictionaries
    train_ds_dict = tokenized_datasets["train"].to_tf_dataset(
        columns=['attention_mask', 'input_ids', 'labels'],
        shuffle=True,
        batch_size=16,
        collate_fn=data_collator
    )

    val_ds_dict = tokenized_datasets["validation"].to_tf_dataset(
        columns=['attention_mask', 'input_ids', 'labels'],
        shuffle=False,
        batch_size=16,
        collate_fn=data_collator
    )

    test_ds_dict = tokenized_datasets["test"].to_tf_dataset(
        columns=['attention_mask', 'input_ids', 'labels'],
        shuffle=False,
        batch_size=16,
        collate_fn=data_collator
    )

    # 6. manually creating the (features, labels) tuple
 
    def to_keras_format(batch):
        # The input features for the model must be a dictionary
        features = {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"]}
        # The labels are the second element of the tuple
        labels = batch["labels"]
        return (features, labels)

    train_ds = train_ds_dict.map(to_keras_format)
    val_ds = val_ds_dict.map(to_keras_format)
    test_ds = test_ds_dict.map(to_keras_format)

    return train_ds, val_ds, test_ds