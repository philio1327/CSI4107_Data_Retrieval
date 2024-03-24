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
        filepath = f"src\\AP_collection\\coll\\{filename}"
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

def process_text(text):
    # Remove punctuation marks
    text = remove_punctuation(text)
    # Convert text to lowercase
    text = text.lower()
    # Tokenize the text into words
    words = text.split()
    return words

def remove_punctuation(text):
    clean_text = re.sub(rf"[{string.punctuation}]", " ", text)
    return clean_text

def remove_stopwords(list_of_words, stop_words_list):
    return [word for word in list_of_words if word not in stop_words_list]

def stem_words(content):
  # Initialize PorterStemmer
  stems = [porter.stem(word) for word in content]
  return stems

def get_tokens(doc_content, stop_words_list):
    stop_words_list = set(stop_words_list)
    # To read the file from the command line
    # if len(sys.argv) != 2:
    #     print("Usage: python Assignment_1_Group15.py <filename>")
    #     print("Example: python Assignment_1_Group15.py AP880212")
    #     filename1 = input("Enter Name of File Here: ")

    #     file_content = read_file(filename1)
    # else:
    #     filename = sys.argv[1]
    #     file_content = read_file(filename)
    #file_content = read_file(filename)

    read_file_end = time.time()
    #print(f"Read File Time: {file_content[1]} Seconds")

    clean_content = remove_html_tags(doc_content)
    html_remove_end = time.time()
    #print(f"Removed HTML tags Time: {html_remove_end-read_file_end} Seconds")

    list_of_words = process_text(clean_content)
    punct_remove_time = time.time()
    #print(f"Removed Punctuation and Lowercase the words Time: {punct_remove_time-html_remove_end} Seconds")

    words_without_stopwords = remove_stopwords(list_of_words, stop_words_list)
    remove_stopwords_time = time.time()
    #print(f"Remove Stopwords Time: {remove_stopwords_time-punct_remove_time} Seconds")
    #print(words_without_stopwords)
    #print("Number of words: " + str(len(words_without_stopwords)))

    # apply Porter stemming
    stemmed_tokens = stem_words(words_without_stopwords)
    stemming_time = time.time()
    # print(f"Stem words Time: {stemming_time-remove_stopwords_time} Seconds")

    return stemmed_tokens

def get_docID_text(doc):
  # getting id
  docID = doc.split("</DOCNO>")[0]
  docID = docID.split("<DOCNO>")[1].strip()

  head_info = doc.split("<HEAD>")
  all_head_info = ' '.join([x.split("</HEAD>")[0].strip() for x in head_info[1:]])

  text_info = doc.split("<TEXT>")
  all_text_info = ' '.join([x.split("</TEXT>")[0].strip() for x in text_info[1:]])

  searchable = all_head_info + " " + all_text_info

  clean_doc = remove_html_tags(searchable).strip()
  clean_doc = clean_doc.replace("\n", " ")

  if len(clean_doc) <= 1: # there is no text in doc
    clean_doc = docID

  return (docID, clean_doc)
##########################################################################################################
# END CODE FROM ASSIGNMENT 1
##########################################################################################################

