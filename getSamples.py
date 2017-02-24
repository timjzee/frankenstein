import os
import re
import pickle
from sklearn.feature_extraction.text import CountVectorizer

SAMPLE_SIZES = [25, 50, 100, 200, 400]
MAX_SAMPLE_SIZE = max(SAMPLE_SIZES)


def getFileNames():
    input_path = "/Users/tim/GitHub/frankenstein/clean_texts/author_known/"
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
        if i % (10 * SAMPLE_SIZE) == 0:
            print("Working on sample {}".format(int(i / SAMPLE_SIZE)))
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
    for name in file_names:
        if name[:3] not in category_nms:
            category_nms.append(name[:3])
        category_index = len(category_nms) - 1
        f = open(file_path + name, "r", encoding="utf-8")
        text = f.read()
        f.close()
        word_list = getInfo(text)
        print("Start sampling {} ({} words) with a sample size of {}".format(name, len(word_list), SAMPLE_SIZE))
        text_samples = getSamples(word_list, text)
        all_samples.extend(text_samples)
        num_samples = len(text_samples)
        category_lbls.extend([category_index for i in range(num_samples)])
    return all_samples, category_lbls, category_nms


def saveSamples():
    """Save samples, category_labels and category_names to a pickle of a list."""
    samples_path = "/Users/tim/GitHub/frankenstein/sampled_texts/check2/"
    samples_structure = [samples, category_labels, category_names]
    g = open(samples_path + "samples_" + str(SAMPLE_SIZE) + ".pck", "wb")
    pickle.dump(samples_structure, g)
    g.close()


for SAMPLE_SIZE in SAMPLE_SIZES:
    samples, category_labels, category_names = processTexts()
    saveSamples()
