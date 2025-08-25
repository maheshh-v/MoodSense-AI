from transformers import TFAutoModelForSequenceClassification

def build_bert_model(num_labels=6):
    """
    Builds a DistilBERT model for sequence classification.
    """
    model = TFAutoModelForSequenceClassification.from_pretrained(
        'distilbert-base-uncased', num_labels=num_labels
    )
    return model