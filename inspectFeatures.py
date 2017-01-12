import pickle
import re
import sys
from sklearn.feature_extraction.text import CountVectorizer

SAMPLE_SIZES = [25, 50, 100, 200]


def loadTrainSamples():
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


def inspectSamples(train_samples):
    """Counts the number of words in each raw_text and returns them."""
    tkn_regex = re.compile(r'[A-Za-z0-9]+')
    count_vect = CountVectorizer(analyzer='word', token_pattern=tkn_regex, max_features=200)
    train_counts = count_vect.fit_transform(train_samples)
    features = count_vect.get_feature_names()
    return train_counts, features


def getFewestSamples():
    """Get lowest amount of samples for a single author in each sample size."""
    counts = []
    for name in names:
        name_index = names.index(name)
        sample_count = labels.count(name_index)
        # print("{} samples for {} at sample size {}".format(sample_count, name, SAMPLE_SIZE))
        counts.append(sample_count)
    lowest_count = min(counts)
    new_sample_list = []
    new_labels = []
    starting_sample = 0
    for n in names:
        n_index = names.index(n)
        end_sample = starting_sample + lowest_count
        new_sample_list.extend(known_samples[starting_sample:end_sample])
        new_labels.extend(labels[starting_sample:end_sample])
        print("{} samples for {} at sample size {}. New amount: {}".format(counts[n_index], n, SAMPLE_SIZE, len(labels[starting_sample:end_sample])))
        starting_sample = starting_sample + counts[n_index]
    return new_sample_list, new_labels


def saveNewSamples():
    """Save sample lists with equal amount of samples per author."""
    new_samples_path = "/Users/tim/GitHub/frankenstein/sampled_texts/equalized_samples/"
    samples_structure = [new_smpls, new_lbls, names]
    g = open(new_samples_path + "samples_" + str(SAMPLE_SIZE) + ".pck", "wb")
    pickle.dump(samples_structure, g)
    g.close()


for SAMPLE_SIZE in SAMPLE_SIZES:
    known_samples, labels, names = loadTrainSamples()
    # train_cnts, ftrs = inspectSamples(known_samples)
    new_smpls, new_lbls = getFewestSamples()
    print("Total amount at sample size {} is {}".format(SAMPLE_SIZE, len(new_smpls)))
    saveNewSamples()
