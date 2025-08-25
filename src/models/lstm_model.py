from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

def build_model(vocab_size, max_len, embedding_dim=128):
    ''' 
    this function builds and returns a Sequential model with an 
    -LSTM layer, 
    -dense output layer, and 
    -embedding layer.
     
     it uses the following parameters:
      - vocab_size: size of the vocabulary
      - max_len: maximum length of input sequences
      - embedding_dim: dimension of the embedding layer (default is 128)
      
      
      The model is compiled with the Adam optimizer, sparse categorical crossentropy loss, and accuracy metric.
    '''
    
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len))

    
     # LSTM with dropout on inputs + recurrent state
    model.add(LSTM(64, dropout=0.3, recurrent_dropout=0.3))

    # Extra dropout before Dense
    model.add(Dropout(0.5))  

    model.add(Dense(6, activation='softmax'))

    # Compiling the model
     # using sparse categorical crossentropy loss for multi-class classification
    model.compile(
        
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model