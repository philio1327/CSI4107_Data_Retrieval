##########################################################################################################
## Assignment 2 - Neural Network Retrieval System
## Created by: Philip Anderegg & Svetlana Esina & Daniel Dumitrescu
## Student Number: 8191716 & 300176419 & 300176257
## Course Name: Information Retrieval and Internet
## Course Code: CSI 4107
## Professor: Diana Inkpen
## Due Date: March 30th, 2024
##########################################################################################################

import time
import requests
import re
from nltk import PorterStemmer as porter
import string
import os
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers
from tensorflow.keras import losses
from transformers import BertTokenizer, TFBertModel
import numpy as np
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import models


##########################################################################################################
# PRE-PROCESSING FUNCTIONS FROM ASSIGNMENT 1
##########################################################################################################
# Get Stopwords from Professor's website
def get_words_from_webpage(url):
    try:
        # Send a GET request to the webpage
        response = requests.get(url)

        # Check if request was successful
        if response.status_code == 200:
            # Split the webpage content by lines to get individual words
            words = response.text.split()
            return words
        else:
            print("Failed to fetch webpage. Status code:", response.status_code)
            return None
    except Exception as e:
        print("An error occurred:", e)
        return None


# Function to Read the File based on filename
def read_file(filename):
    try:
        start = time.time()
        filepath = f"AP_collection\\coll\\{filename}"
        with open(filepath, "r") as file:
            content = file.read()
        end = time.time()
        return content, (end - start)
    except:
        print("Error in File Retrieval ")
        filename1 = input("Enter Name of File Here: ")
        return read_file(filename1)


# Read a file that is directly within this directory
def read_file_direct(filename):
    try:
        with open(filename, "r") as file:
            content = file.read()
        return content
    except:
        print('Error in File Retrieval')
        return None


def remove_html_tags(content):
    clean_content = re.sub(r"<[^>]*>", "", content)
    return clean_content


##########################################################################################################
# END CODE FROM ASSIGNMENT 1
##########################################################################################################
def extract_docno_head_text(document):
    docno_pattern = re.compile(r'<DOCNO>(.*?)</DOCNO>')
    head_pattern = re.compile(r'<HEAD>(.*?)</HEAD>')

    # Extract document number and headline
    docno_match = docno_pattern.search(document)
    head_match = head_pattern.search(document)
    if docno_match and head_match:
        docno = docno_match.group(1).strip()
        head = head_match.group(1).strip()
        # Extract Text content excluding <TEXT> tags
        text_content = document.split('<TEXT>', 1)[-1].split('</TEXT>', 1)[0].strip()
        return (docno, head, text_content.lower())
    return None


def read_all_files_in_directory():
    document_list = []
    # Get the directory containing the current Python script
    # script_directory = os.path.dirname(os.path.abspath(__file__))
    # Get the current working directory
    current_directory = os.getcwd()
    print("Current working directory:", current_directory)
    # Relative path to the folder containing your files
    folder_path = os.path.join(current_directory, 'AP_collection', 'coll')

    # Iterate over each file in the folder
    for filename in os.listdir(folder_path):
        # Check if the item is a file (not a directory)
        if os.path.isfile(os.path.join(folder_path, filename)):
            # Construct the full path to the file
            file_path = os.path.join(folder_path, filename)
            # Open the file and read its contents
            file_content = read_file_direct(file_path)
            # split content by <DOC> tag
            split_content = file_content.split("<DOC>")
            for doc in split_content:
                result = extract_docno_head_text(doc)
                if result:
                    document_list.append(result)
    return document_list


def transform_to_dictionary(document_list):
    document_dict = {}
    for docno, head, text_content in document_list:
        document_dict[docno] = (head, text_content)
    return document_dict


def preprocess_text(text):
    text = remove_html_tags(text)
    text = text.lower()  # Convert text to lowercase
    return text


if __name__ == "__main__":
    print(f"TensorFlow Version: {tf.__version__}")
    document_list = read_all_files_in_directory()
    document_dict = transform_to_dictionary(document_list)

    max_sequence_length = 1024

    # Step 1: Preprocess text
    preprocessed_texts = [preprocess_text(text)[:max_sequence_length] for _, (_, text) in document_dict.items()]
    print("Text Preprocessed!")

    # Step 2: Tokenization
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    print("Tokenizer Loaded!")
    tokenized_texts = tokenizer(preprocessed_texts, padding=True, truncation=True, return_tensors='tf')
    print("Tokenization Complete!")

    # Step 3: Convert text to input IDs
    input_ids = tokenized_texts['input_ids']
    print("Input IDs Created!")

    # Step 4: Load BERT Model
    bert_model = TFBertModel.from_pretrained('bert-base-uncased')
    print("BERT Model Loaded!")
    # Step 5: Pass input IDs through BERT Model
    outputs = bert_model(input_ids)
    print("Passed Input IDs through Model!")

    # Extract embeddings from BERT output
    embeddings = outputs.last_hidden_state
    print("Embeddings extracted!")

    # Define the model architecture for unsupervised learning
    max_sequence_length = 1024  # Define your max sequence length here

    input_layer = Input(shape=(max_sequence_length,), dtype='int32')
    bert_output = bert_model(input_layer)[0]
    print('Check BERT Output!')
    pooled_output = bert_output[:, 0, :]  # Use the pooled output for classification
    print("Pooled Output used")
    reconstruction_output = Dense(max_sequence_length)(pooled_output)  # Reconstruct the input

    autoencoder_model = models.Model(inputs=input_layer, outputs=reconstruction_output)
    autoencoder_model.compile(optimizer='adam', loss='mean_squared_error')
    print("Loss Compiled!")

    # Train the autoencoder model
    autoencoder_model.fit(input_ids, input_ids, epochs=5, batch_size=16, validation_split=0.2)

    # You can get all the keys by typing document_dict.keys()
    # if you type print(document_dict["AP880212-0001"]) you will get a tuple of the form (head, text)
    # Sample print out:

    # Print the document with new lines
    # docno = "AP880212-0001"
    # head, text_content = document_dict[docno]
    # print("Head:", head)
    # print("Text:")
    # # Split the text content by newline and print each line separately
    # for line in text_content.split("\n"):
    #     print(line)





