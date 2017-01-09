import sys
import pickle
import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


SAMPLE_SIZES = [200, 400, 800, 1600]
SHIFT_PROPORTION = 0.1
NUMBER_OF_SHIFTS = int(1 / SHIFT_PROPORTION)


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


def getSamples(shift_number):
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


def loadSamples(shift):
    """Checks whether samples have already been made and either loads them or calls getSamples."""
    franken_smpls_path = "/Users/tim/GitHub/frankenstein/sampled_texts/franken_samples/"
    try:
        f = open("{}samples_{}-s{}.pck".format(franken_smpls_path, SAMPLE_SIZE, shift), "rb")
        franken_samples = pickle.load(f)
    except:
        franken_samples = getSamples(shift)
        f = open("{}samples_{}-s{}.pck".format(franken_smpls_path, SAMPLE_SIZE, shift), "wb")
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


def writeOutput(predicted, shift_num):
    if SAMPLE_SIZE == SAMPLE_SIZES[0] and shift_num == 1:
        f = open("/Users/tim/GitHub/frankenstein/results/franken_results.csv", "w")
        f.write("sample_size,shift_percentage")
        for author in label_names:
            f.write(",%_" + author)
        f.write("\n")
        f.close()
    shift_percentage = SHIFT_PROPORTION * 100 * shift_num
    f = open("/Users/tim/GitHub/frankenstein/results/franken_results.csv", "a")
    f.write("{},{}".format(SAMPLE_SIZE, shift_percentage))
    for name in label_names:
        name_index = label_names.index(name)
        absolute_count = (predicted == name_index).sum()
        percentage = (absolute_count / len(predicted)) * 100
        f.write("," + str(percentage))
    f.write("\n")
    f.close()


def loopThroughSampleShifts():
    """Main loop for the handling of Frankenstein samples."""
    for shift in range(NUMBER_OF_SHIFTS):
        # test_samples = getSamples(shift)
        test_samples = loadSamples(shift)
        print("Shift {}: {} samples".format(shift, len(test_samples)))
        classifier_output = classifyFrankenstein(test_samples)
        writeOutput(classifier_output, shift)


for SAMPLE_SIZE in SAMPLE_SIZES:
    raw_text = loadFrankenstein()
    tokens = getTokens()
    training_samples, training_labels, label_names = loadTrainingSamples()
    loopThroughSampleShifts()
