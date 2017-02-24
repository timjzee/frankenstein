import pickle
import sys
import re
import numpy as np
from nltk.tag import pos_tag_sents
from nltk.tokenize import word_tokenize
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
# from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline

SAMPLE_SIZES = [25, 50, 100, 200, 400]
NUMBER_OF_CV = 10
FEATURE_TYPE = "TAGS"


def getFunctionWords():
    """Gets list of function words for feature vector."""
    try:
        f = open("/Users/tim/GitHub/frankenstein/function_words.pck", "rb")
    except:
        print("Can't find function word list.")
        sys.exit()
    f_words = pickle.load(f)
    f.close()
    return f_words


def loadSamples():
    """Loads samples and labels from pickle and returns a list of samples, a list of sample labels and a list of label names."""
    samples_path = "/Users/tim/GitHub/frankenstein/sampled_texts/check/"
    try:
        g = open("{}samples_{}.pck".format(samples_path, SAMPLE_SIZE), "rb")
    except:
        print("Can't find sample file.")
        sys.exit()
    sample_files = pickle.load(g)
    g.close()
    smpls, lbls, nms = sample_files
    return smpls, lbls, nms


def tagSamples(s_list):
    """Takes a list of samples and returns a list of tagged samples, where each tagged sample is a continues string of tags seperated by whitespaces. This format is easy to parse for the Scikit Learn vectorizer."""
    new_list = []
    for s in s_list:
        sentences = re.split(r'([;.!?][ "])', s)
        sentences2 = []
        for sentence_index in range(0, len(sentences) - 1, 2):
            sentences2.append(sentences[sentence_index] + sentences[sentence_index + 1])
        sentences2.append(sentences[-1])
        tokenized_sentences = [word_tokenize(sentence) for sentence in sentences2]
        tagged_sentences = pos_tag_sents(tokenized_sentences)
        tag_list = [tpl[1] for sentence in tagged_sentences for tpl in sentence]
        tag_string = " ".join(tag_list)
        new_list.append(tag_string)
    return new_list


def classifyTestSamples(train_smpls, train_lbls, test_smpls, test_lbls):
    if FEATURE_TYPE == "TOKENS":
        tkn_regex = re.compile(r'[A-Za-z0-9]+')
        classifier = Pipeline([('vect', CountVectorizer(analyzer='word', token_pattern=tkn_regex, vocabulary=function_words)), ('clf', SVC(kernel="linear"))])
        classifier = classifier.fit(train_smpls, train_lbls)
        predicted_classes = classifier.predict(test_smpls)
    elif FEATURE_TYPE == "TAGS":
        train_tags = tagSamples(train_smpls)
        test_tags = tagSamples(test_smpls)
        tag_regex = re.compile(r'[^ ]+')
        classifier = Pipeline([('vect', CountVectorizer(analyzer='word', token_pattern=tag_regex, lowercase=False, ngram_range=(2, 2))), ('clf', SVC(kernel="linear"))])
        classifier = classifier.fit(train_tags, train_lbls)
        predicted_classes = classifier.predict(test_tags)
    acc = np.mean(predicted_classes == test_lbls)
    print("Mean accuracy:", acc)
    return predicted_classes


def makeCrossValFolds():
    """Makes cross-validation folds and loops through them."""
    folder = StratifiedKFold(n_splits=NUMBER_OF_CV)
    num_samples = len(samples)
    fold_generator = folder.split(np.zeros(num_samples), sample_labels)
    return fold_generator


def writeEvalScores(predicted, actual, fold):
    """Calculates TP, FP, TN and FN values relative to total number of samples in cross-fold."""
    conf_matrix = confusion_matrix(actual, predicted)
    if SAMPLE_SIZE == SAMPLE_SIZES[0] and fold == 1:
        f = open("/Users/tim/GitHub/frankenstein/results/check4/equalized_results_{}-cv_{}.csv".format(NUMBER_OF_CV, FEATURE_TYPE), "w")
        f.write("sample_size,fold")
        for true_author in label_names:
            for predicted_author in label_names:
                f.write(",true_" + true_author + "_pred_" + predicted_author)
        f.write("\n")
        f.close()
    f = open("/Users/tim/GitHub/frankenstein/results/check4/equalized_results_{}-cv_{}.csv".format(NUMBER_OF_CV, FEATURE_TYPE), "a")
    f.write("{},{}".format(SAMPLE_SIZE, fold))
    for tr_author in label_names:
        for pr_author in label_names:
            true_index = label_names.index(tr_author)
            predicted_index = label_names.index(pr_author)
            absolute_num = conf_matrix[true_index, predicted_index]
            percentage = (absolute_num / len(predicted)) * 100
            f.write("," + str(percentage))
    f.write("\n")
    f.close()


def loopThroughCV():
    counter = 0
    for train_index, test_index in cv_generator:
        counter += 1
        train_index_list = train_index.tolist()
        train_samples = [samples[i] for i in train_index_list]
        train_labels = [sample_labels[j] for j in train_index_list]
        test_index_list = test_index.tolist()
        test_samples = [samples[k] for k in test_index_list]
        test_labels = [sample_labels[l] for l in test_index_list]
        classifier_output = classifyTestSamples(train_samples, np.array(train_labels), test_samples, np.array(test_labels))
        writeEvalScores(classifier_output, test_labels, counter)


function_words = getFunctionWords()
for SAMPLE_SIZE in SAMPLE_SIZES:
    samples, sample_labels, label_names = loadSamples()
    cv_generator = makeCrossValFolds()
    loopThroughCV()
