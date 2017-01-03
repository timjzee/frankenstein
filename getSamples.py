import os
import nltk


def getFileNames():
    input_path = "/Users/tim/GitHub/frankenstein/clean_texts/"
    temp_list = os.listdir(input_path)
    name_list = [i for i in temp_list if i[-4:] == ".txt"]
    return name_list, input_path


def getInfo(raw_text):
    """Counts the number of words in each raw_text and returns them."""
    tokens = nltk.word_tokenize(raw_text)
    word_count = len(tokens)
    return word_count


def processTexts():
    """Loops through texts, analyses them and creates classification files."""
    file_names, file_path = getFileNames()
    for name in file_names:
        f = open(file_path + name, "r", encoding="utf-8")
        text = f.read()
        f.close
        num_words = getInfo(text)
        print(name + ": " + str(num_words) + " words")


processTexts()
