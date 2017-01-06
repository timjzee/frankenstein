import os
import re
from sklearn.feature_extraction.text import CountVectorizer

# SAMPLE_SIZES = [200, 600, 1000, 1400, 1800, 2200]
# MAX_SAMPLE_SIZE = max(SAMPLE_SIZES)
MAX_SAMPLE_SIZE = 2200
SAMPLE_SIZE = 200


def getFileNames():
    input_path = "/Users/tim/GitHub/frankenstein/author_known/"
    temp_list = os.listdir(input_path)
    name_list = [i for i in temp_list if i[-4:] == ".txt"]
    return name_list, input_path


def getInfo(raw_text):
    """Counts the number of words in each raw_text and returns them."""
    tkn_regex = re.compile(r'[A-Za-z0-9]+')
    count_vect = CountVectorizer(analyzer='word', token_pattern=tkn_regex)
    analyze = count_vect.build_analyzer()
    tokens = analyze(raw_text)
    return tokens


def getSamples(token_list, whole_text):
    """Takes token_list and based on SAMPLE_SIZE and MAX_SAMPLE_SIZE finds sample boundaries in whole_text and splits whole_text at those boundaries, returning a list of samples."""
    token_count = len(token_list)
    sampling_max = token_count - token_count % MAX_SAMPLE_SIZE
    list_of_samples = []
    sample_start_index = 0
    for i in range(SAMPLE_SIZE, sampling_max + 1, SAMPLE_SIZE):
        lower_boundary = i - SAMPLE_SIZE
        token_sublist = token_list[lower_boundary:i]
        sample_end_index = re.search(r'[\W]*' + r'[\W]+'.join(token_sublist), whole_text, re.I).end() + 1
        text_sample = whole_text[sample_start_index:sample_end_index]
        list_of_samples.append(text_sample)
        sample_start_index = sample_end_index
    return list_of_samples


def processTexts():
    """Loops through texts, analyses them and creates classification files."""
    file_names, file_path = getFileNames()
    all_samples = []
    category_nms = []
    category_lbls = []
    category_index = 0
    for name in file_names:
        category_nms.append(name[:len(file_names)])
        f = open(file_path + name, "r", encoding="utf-8")
        text = f.read()
        f.close
        word_list = getInfo(text)
        print(name + ": " + str(len(word_list)) + " words")
        text_samples = getSamples(word_list, text)
        all_samples.extend(text_samples)
        num_samples = len(text_samples)
        category_lbls.extend([category_index for i in range(num_samples)])
        category_index += 1
    return all_samples, category_lbls, category_nms


samples, category_labels, category_names = processTexts()
