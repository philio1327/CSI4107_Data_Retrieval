import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

print(f"TensorFlow Version: {tf.__version__}")
print(f"TensorFlow Hub Version: {hub.__version__}")

# Tokenization and conversion to IDs function (replace with your tokenizer)
def tokenize_and_convert_to_ids(texts):
    # Your tokenization code here
    return input_ids, segment_ids

# Softmax function
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

# Input sequences
input_sequences = ["I like chicken fried rice", "The sky is blue today"]

# Tokenize and convert to IDs
input_ids, segment_ids = tokenize_and_convert_to_ids(input_sequences)
print(input_ids)
inputs = {
  'input_ids': input_ids,
  'segment_ids': segment_ids
}

# Load BEM model from TFHub
bem = hub.load('https://www.kaggle.com/models/google/bert/frameworks/TensorFlow2/variations/answer-equivalence-bem/versions/1')

# The outputs are raw logits
raw_outputs = bem(inputs)

# Transform raw logits into probabilities using softmax
bem_score = softmax(np.squeeze(raw_outputs))[1]

print(bem_score)
