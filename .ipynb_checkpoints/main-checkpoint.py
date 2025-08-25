from preprocessing import load_and_prepare_data
from model import build_model
from sklearn.utils.class_weight import compute_class_weight
import numpy as np


data = load_and_prepare_data()

train_data, vali_data, test_data , meta = data['train'], data['vali'], data['test'] , data['meta']

padded_train, train_labels = train_data
padded_vali, vali_labels = vali_data
padded_test, test_labels = test_data


#balancing class weights for imbalanced dataset


class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_labels),
    y=train_labels
)

class_weights = dict(enumerate(class_weights))
print("Class weights:", class_weights)

vocab_size = meta['vocab_size']
max_len = meta['max_len']   

model = build_model(vocab_size, max_len)

model.summary()

model.fit(
    padded_train,
    train_labels,
    epochs=10,
    validation_data=(padded_vali, vali_labels),
    class_weight=class_weights)





