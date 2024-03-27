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
        return content, (end-start)
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
        return (docno, head, text_content)
    return None
def read_all_files_in_directory():
    document_list = []
    # Get the directory containing the current Python script
    script_directory = os.path.dirname(os.path.abspath(__file__))

    # Relative path to the folder containing your files
    folder_path = os.path.join(script_directory, 'AP_collection', 'coll')

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

if __name__ == "__main__":
    document_list = read_all_files_in_directory()
    document_dict = transform_to_dictionary(document_list)

    # You can get all the keys by typing document_dict.keys()
    # if you type print(document_dict["AP880212-0001"]) you will get a tuple of the form (head, text)
    # Sample print out:

    # Print the document with new lines
    docno = "AP880212-0001"
    head, text_content = document_dict[docno]
    print("Head:", head)
    print("Text:")
    # Split the text content by newline and print each line separately
    for line in text_content.split("\n"):
        print(line)





