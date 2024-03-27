from keras.datasets import imdb
from keras.preprocessing import sequence
import keras
import tensorflow as tf
import os
import numpy as np


VOCAB_SIZE = 88584

MAXLEN = 250
BATCH_SIZE = 64

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words = VOCAB_SIZE)

train_data[1]

# More Preprocessing
train_data = sequence.pad_sequences(train_data, MAXLEN)
test_data = sequence.pad_sequences(test_data, MAXLEN)

# Create the layers: First is Word Embedding Layer, Second is LSTM, Third is Dense for Prediction
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(VOCAB_SIZE, 32),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.summary()

# Train the model
model.compile(loss="binary_crossentropy",optimizer="rmsprop",metrics=['acc'])

history = model.fit(train_data, train_labels, epochs=10, validation_split=0.2)

# Evaluate the model
results = model.evaluate(test_data, test_labels)
print(results)

# Making Predictions
word_index = imdb.get_word_index()

def encode_text(text, word_index, maxlen):
    tokens = text.lower().split()  # Split text into tokens
    encoded_tokens = [word_index[word] if word in word_index else 0 for word in tokens]  # Convert tokens to indices

    # Pad sequences manually
    if len(encoded_tokens) < maxlen:
        padded_tokens = encoded_tokens + [0] * (maxlen - len(encoded_tokens))  # Pad with zeros
    else:
        padded_tokens = encoded_tokens[:maxlen]  # Truncate if longer than maxlen

    return padded_tokens

text = "that movie was just amazing, so amazing"
encoded = encode_text(text, word_index, MAXLEN)  # Assuming word_index and MAXLEN are defined appropriately
print(encoded)

# while were at it lets make a decode function

reverse_word_index = {value: key for (key, value) in word_index.items()}


def decode_integers(integers):
    PAD = 0
    text = ""
    for num in integers:
        if num != PAD:
            text += reverse_word_index[num] + " "

    return text[:-1]


print(decode_integers(encoded))

# now time to make a prediction

def predict(text):
    encoded_text = encode_text(text, word_index, MAXLEN)  # Pass word_index and MAXLEN to encode_text
    pred = np.zeros((1, MAXLEN))
    pred[0] = encoded_text
    result = model.predict(pred)
    print(result[0])

positive_review = "That movie was! really loved it and would great watch it again because it was amazingly great"
predict(positive_review)

negative_review = "that movie really sucked. I hated it and wouldn't watch it again. Was one of the worst things I've ever watched"
predict(negative_review)