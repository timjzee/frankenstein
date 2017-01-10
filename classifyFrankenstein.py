import sys
import pickle
import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


SAMPLE_SIZES = [100, 200, 400, 800]
TESTING_METHOD = "groups"

if TESTING_METHOD == "shifts":
    SHIFT_PROPORTION = 0.1
    NUMBER_OF_SHIFTS = int(1 / SHIFT_PROPORTION)
elif TESTING_METHOD == "groups":
    NUMBER_OF_GROUPS = 10


def loadFrankenstein():
    franken_path = "/Users/tim/GitHub/frankenstein/clean_texts/author_unknown/"
    f = open(franken_path + "MWS-PBS-frankenstein-clean.txt", encoding='utf-8')
    text = f.read()
    f.close()
    return text


def getTokens():
    tkn_regex = re.compile(r'[A-Za-z0-9æëâêô]+')
    count_vect = CountVectorizer(analyzer='word', token_pattern=tkn_regex)
    analyze = count_vect.build_analyzer()
    franken_tokens = analyze(raw_text)
    return franken_tokens


def loadTrainingSamples():
    """Loads samples and labels from pickle and returns a list of samples, a list of sample labels and a list of label names."""
    samples_path = "/Users/tim/GitHub/frankenstein/sampled_texts/known_samples/"
    try:
        g = open("{}samples_{}.pck".format(samples_path, SAMPLE_SIZE), "rb")
    except:
        print("Can't find sample file.")
        sys.exit()
    sample_files = pickle.load(g)
    g.close()
    smpls, lbls, nms = sample_files
    return smpls, lbls, nms


def getShiftedSamples(shift_number):
    """Makes and shifts samples."""
    token_count = len(tokens)
    sampling_max = token_count - (token_count % max(SAMPLE_SIZES))
    absolute_shift = int(shift_number * SHIFT_PROPORTION * SAMPLE_SIZE)
    if absolute_shift == 0:
        sample_start_index = 0
    else:
        skipped_tokens = tokens[:absolute_shift]
        sample_start_index = re.search(r'[^\wæëâêô]*' + r'[^\wæëâêô]+'.join(skipped_tokens), raw_text, re.I).end() + 1
        skipped_text = raw_text[:sample_start_index]
    list_of_samples = []
    for i in range(0, sampling_max - SAMPLE_SIZE + 1, SAMPLE_SIZE):
        # if i % (10 * SAMPLE_SIZE) == 0:
        #     print("Working on sample {}".format(int(i / SAMPLE_SIZE)))
        lower_boundary = i + absolute_shift
        upper_boundary = lower_boundary + SAMPLE_SIZE
        if not upper_boundary > sampling_max:
            token_sublist = tokens[lower_boundary:upper_boundary]
            sample_end_index = re.search(r'[^\wæëâêô]*' + r'[^\wæëâêô]+'.join(token_sublist), raw_text, re.I).end() + 1
            text_sample = raw_text[sample_start_index:sample_end_index]
            list_of_samples.append(text_sample)
            sample_start_index = sample_end_index
        else:
            continue    # Remove statement to patch skipped start text to final sample (this results in equal amount of samples compared to list of unshifted samples)
            token_sublist = tokens[lower_boundary:sampling_max]
            sample_end_index = re.search(r'[^\wæëâêô]*' + r'[^\wæëâêô]+'.join(token_sublist), raw_text, re.I).end() + 1
            text_fragment = raw_text[sample_start_index:sample_end_index]
            text_sample = text_fragment + skipped_text
            list_of_samples.append(text_sample)
    return list_of_samples


def getGroupedSamples(group_num):
    """Makes and groups samples."""
    token_count = len(tokens)
    sampling_max = token_count - (token_count % max(SAMPLE_SIZES))
    total_num_samples = sampling_max / SAMPLE_SIZE
    print("Total number of samples:", int(total_num_samples))
    basic_group_size = int(total_num_samples / NUMBER_OF_GROUPS)
    num_leftover_samples = total_num_samples % NUMBER_OF_GROUPS
    size_list = []
    for i in range(NUMBER_OF_GROUPS):
        size = basic_group_size
        if num_leftover_samples != 0:
            size += 1
            num_leftover_samples -= 1
        size_list.append(size)
    token_boundaries = []
    lower_boundary = 0
    for j in size_list:
        upper_boundary = lower_boundary + j * SAMPLE_SIZE
        token_boundaries.append([lower_boundary, upper_boundary])
        lower_boundary = upper_boundary
    start_boundary, stop_boundary = token_boundaries[group_num - 1]
    group_size = size_list[group_num - 1]
    sample_list = []
    for k in range(group_size):
        end_boundary = start_boundary + SAMPLE_SIZE
        token_sublist = tokens[start_boundary:end_boundary]
        string_indices = re.search(r'[^\wæëâêô]*' + r'[^\wæëâêô]+'.join(token_sublist), raw_text, re.I).span()
        text_sample = raw_text[string_indices[0]:string_indices[1]]
        sample_list.append(text_sample)
        if end_boundary == stop_boundary and k != group_size - 1:
            print("Problem with group sampler!")
        start_boundary = end_boundary
    return sample_list


def loadSamples(identifier):
    """Checks whether samples have already been made and either loads them or calls a function to make them."""
    franken_smpls_path = "/Users/tim/GitHub/frankenstein/sampled_texts/franken_samples/"
    if TESTING_METHOD == "shifts":
        try:
            f = open("{}samples_{}-s{}.pck".format(franken_smpls_path, SAMPLE_SIZE, identifier), "rb")
            franken_samples = pickle.load(f)
        except:
            franken_samples = getShiftedSamples(identifier)
            f = open("{}samples_{}-s{}.pck".format(franken_smpls_path, SAMPLE_SIZE, identifier), "wb")
            pickle.dump(franken_samples, f)
        f.close()
    elif TESTING_METHOD == "groups":
        try:
            f = open("{}samples_{}-g{}.pck".format(franken_smpls_path, SAMPLE_SIZE, identifier), "rb")
            franken_samples = pickle.load(f)
        except:
            franken_samples = getGroupedSamples(identifier)
            f = open("{}samples_{}-g{}.pck".format(franken_smpls_path, SAMPLE_SIZE, identifier), "wb")
            pickle.dump(franken_samples, f)
        f.close()
    return franken_samples


def classifyFrankenstein(test_smpls):
    """Classifies Frankenstein samples using samples with known authors."""
    train_lbls = np.array(training_labels)
    token_regex = re.compile(r'[A-Za-z0-9æëâêô]+')
    classifier = Pipeline([('vect', CountVectorizer(analyzer='word', token_pattern=token_regex)), ('clf', MultinomialNB())])
    classifier = classifier.fit(training_samples, train_lbls)
    predicted_classes = classifier.predict(test_smpls)
    return predicted_classes


def writeOutput(predicted, identifier):
    if TESTING_METHOD == "shifts":
        if SAMPLE_SIZE == SAMPLE_SIZES[0] and identifier == 0:
            f = open("/Users/tim/GitHub/frankenstein/results/franken_results_shifts.csv", "w")
            f.write("sample_size,shift_percentage")
            for author in label_names:
                f.write(",%_" + author)
            f.write("\n")
            f.close()
        shift_percentage = SHIFT_PROPORTION * 100 * identifier
        f = open("/Users/tim/GitHub/frankenstein/results/franken_results_shifts.csv", "a")
        f.write("{},{}".format(SAMPLE_SIZE, shift_percentage))
        for name in label_names:
            name_index = label_names.index(name)
            absolute_count = (predicted == name_index).sum()
            percentage = (absolute_count / len(predicted)) * 100
            f.write("," + str(percentage))
        f.write("\n")
        f.close()
    elif TESTING_METHOD == "groups":
        if SAMPLE_SIZE == SAMPLE_SIZES[0] and identifier == 1:
            f = open("/Users/tim/GitHub/frankenstein/results/franken_results_groups.csv", "w")
            f.write("sample_size,group")
            for author in label_names:
                f.write(",%_" + author)
            f.write("\n")
            f.close()
        f = open("/Users/tim/GitHub/frankenstein/results/franken_results_groups.csv", "a")
        f.write("{},{}".format(SAMPLE_SIZE, identifier))
        for name in label_names:
            name_index = label_names.index(name)
            absolute_count = (predicted == name_index).sum()
            percentage = (absolute_count / len(predicted)) * 100
            f.write("," + str(percentage))
        f.write("\n")
        f.close()


def loopThroughSampleStrategy():
    """Main loop for the handling of Frankenstein samples."""
    if TESTING_METHOD == "shifts":
        for shift in range(NUMBER_OF_SHIFTS):
            test_samples = loadSamples(shift)
            print("Shift {}: {} samples".format(shift, len(test_samples)))
            classifier_output = classifyFrankenstein(test_samples)
            writeOutput(classifier_output, shift)
    elif TESTING_METHOD == "groups":
        for group in range(1, NUMBER_OF_GROUPS + 1):
            test_samples = loadSamples(group)
            print("Group {}: {} samples".format(group, len(test_samples)))
            classifier_output = classifyFrankenstein(test_samples)
            writeOutput(classifier_output, group)


raw_text = loadFrankenstein()
tokens = getTokens()
for SAMPLE_SIZE in SAMPLE_SIZES:
    training_samples, training_labels, label_names = loadTrainingSamples()
    loopThroughSampleStrategy()
