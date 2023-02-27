"""Practical 1

Greatly inspired by Stanford CS224 2019 class.
"""

import sys

import pprint

import matplotlib.pyplot as plt
from gensim.models import KeyedVectors
from gensim.test.utils import datapath
import random
import nltk

nltk.download('reuters')
nltk.download('pl196x')
import random

import numpy as np
import scipy as sp
from nltk.corpus import reuters
from nltk.corpus.reader import pl196x
from sklearn.decomposition import PCA, TruncatedSVD

START_TOKEN = '<START>'
END_TOKEN = '<END>'

np.random.seed(0)
random.seed(0)


#################################
# TODO: a)
def distinct_words(corpus):
    """ Determine a list of distinct words for the corpus.
        Params:
            corpus (list of list of strings): corpus of documents
        Return:
            corpus_words (list of strings): list of distinct words across the
            corpus, sorted (using python 'sorted' function)
            num_corpus_words (integer): number of distinct words across the
            corpus
    """
    corpus_words = []
    num_corpus_words = -1

    # ------------------
    # Write your implementation here.
    for text in corpus:
        corpus_words.extend([word for word in text])
    corpus_words = sorted(list(set(corpus_words)))
    num_corpus_words = len(corpus_words)
    # ------------------

    return corpus_words, num_corpus_words


# ---------------------
# Run this sanity check
# Note that this not an exhaustive check for correctness.
# ---------------------

# Define toy corpus
test_corpus = ["START Ala miec kot i pies END".split(" "),
               "START Ala lubic kot END".split(" ")]
test_corpus_words, num_corpus_words = distinct_words(test_corpus)

# Correct answers
ans_test_corpus_words = sorted(list(set(['Ala', 'END', 'START', 'i', 'kot', 'lubic', 'miec', 'pies'])))
ans_num_corpus_words = len(ans_test_corpus_words)

# Test correct number of words
assert (num_corpus_words == ans_num_corpus_words), "Incorrect number of distinct words. Correct: {}. Yours: {}".format(
    ans_num_corpus_words, num_corpus_words)

# Test correct words
assert (test_corpus_words == ans_test_corpus_words), "Incorrect corpus_words.\nCorrect: {}\nYours:   {}".format(
    str(ans_test_corpus_words), str(test_corpus_words))

# Print Success
print("-" * 80)
print("Passed All Tests!")
print("-" * 80)


#################################
# TODO: b)
def compute_co_occurrence_matrix(corpus, window_size=4):
    """ Compute co-occurrence matrix for the given corpus and window_size (default of 4).

        Note: Each word in a document should be at the center of a window.
            Words near edges will have a smaller number of co-occurring words.

              For example, if we take the document "START All that glitters is not gold END" with window size of 4,
              "All" will co-occur with "START", "that", "glitters", "is", and "not".

        Params:
            corpus (list of list of strings): corpus of documents
            window_size (int): size of context window
        Return:
            M (numpy matrix of shape (number of corpus words, number of corpus words)):
                Co-occurence matrix of word counts.
                The ordering of the words in the rows/columns should be the
                same as the ordering of the words given by the distinct_words
                function.
            word2Ind (dict): dictionary that maps word to index
                (i.e. row/column number) for matrix M.
    """
    words, num_words = distinct_words(corpus)
    M = None
    word2Ind = {}

    # ------------------
    # Write your implementation here.
    for i, word in enumerate(words):
        word2Ind[word] = i

    d = dict()
    M = np.zeros((num_words, num_words))

    for text in corpus:
        for cnt, word in enumerate(text):
            tokens = text[cnt + 1:cnt + 1 + window_size]
            for token in tokens:
                key = tuple(sorted([token, word]))
                if key not in d:
                    d[key] = 0
                d[key] += 1

    for key, value in d.items():
        x = word2Ind[key[0]]
        y = word2Ind[key[1]]
        M[x, y] = M[y, x] = value

    # ------------------

    return M, word2Ind


# ---------------------
# Run this sanity check
# Note that this is not an exhaustive check for correctness.
# ---------------------

# Define toy corpus and get student's co-occurrence matrix
test_corpus = ["START Ala miec kot i pies END".split(" "),
               "START Ala lubic kot END".split(" ")]
M_test, word2Ind_test = compute_co_occurrence_matrix(
    test_corpus, window_size=1)

# Correct M and word2Ind
M_test_ans = np.array([
    [0., 0., 2., 0., 0., 1., 1., 0.],
    [0., 0., 0., 0., 1., 0., 0., 1.],
    [2., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 1., 0., 0., 1.],
    [0., 1., 0., 1., 0., 1., 1., 0.],
    [1., 0., 0., 0., 1., 0., 0., 0.],
    [1., 0., 0., 0., 1., 0., 0., 0.],
    [0., 1., 0., 1., 0., 0., 0., 0.]
])

word2Ind_ans = {
    'Ala': 0, 'END': 1, 'START': 2, 'i': 3, 'kot': 4, 'lubic': 5, 'miec': 6,
    'pies': 7}

# Test correct word2Ind
assert (word2Ind_ans == word2Ind_test), "Your word2Ind is incorrect:\nCorrect: {}\nYours: {}".format(word2Ind_ans,
                                                                                                     word2Ind_test)

# Test correct M shape
assert (M_test.shape == M_test_ans.shape), "M matrix has incorrect shape.\nCorrect: {}\nYours: {}".format(M_test.shape,
                                                                                                          M_test_ans.shape)

# Test correct M values
for w1 in word2Ind_ans.keys():
    idx1 = word2Ind_ans[w1]
    for w2 in word2Ind_ans.keys():
        idx2 = word2Ind_ans[w2]
        student = M_test[idx1, idx2]
        correct = M_test_ans[idx1, idx2]
        if student != correct:
            print("Correct M:")
            print(M_test_ans)
            print("Your M: ")
            print(M_test)
            raise AssertionError(
                "Incorrect count at index ({}, {})=({}, {}) in matrix M. Yours has {} but should have {}.".format(idx1,
                                                                                                                  idx2,
                                                                                                                  w1,
                                                                                                                  w2,
                                                                                                                  student,
                                                                                                                  correct))

# Print Success
print("-" * 80)
print("Passed All Tests!")
print("-" * 80)


#################################
# TODO: c)
def reduce_to_k_dim(M, k=2):
    """ Reduce a co-occurence count matrix of dimensionality
        (num_corpus_words, num_corpus_words)
        to a matrix of dimensionality (num_corpus_words, k) using the following
         SVD function from Scikit-Learn:
            - http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html

        Params:
            M (numpy matrix of shape (number of corpus words, number
                of corpus words)): co-occurence matrix of word counts
            k (int): embedding size of each word after dimension reduction
        Return:
            M_reduced (numpy matrix of shape (number of corpus words, k)):
            matrix of k-dimensioal word embeddings.
            In terms of the SVD from math class, this actually returns U * S
    """
    n_iters = 10  # Use this parameter in your call to `TruncatedSVD`
    M_reduced = None
    print("Running Truncated SVD over %i words..." % (M.shape[0]))

    # ------------------
    # Write your implementation here.

    svd = TruncatedSVD(n_components=k, n_iter=n_iters, random_state=13)
    M_reduced = svd.fit_transform(M)

    # ------------------

    print("Done.")
    return M_reduced


# ---------------------
# Run this sanity check
# Note that this not an exhaustive check for correctness
# In fact we only check that your M_reduced has the right dimensions.
# ---------------------

# Define toy corpus and run student code
test_corpus = ["START Ala miec kot i pies END".split(" "),
               "START Ala lubic kot END".split(" ")]
M_test, word2Ind_test = compute_co_occurrence_matrix(test_corpus, window_size=1)
M_test_reduced = reduce_to_k_dim(M_test, k=2)

# Test proper dimensions
assert (M_test_reduced.shape[0] == 8), "M_reduced has {} rows; should have {}".format(M_test_reduced.shape[0], 8)
assert (M_test_reduced.shape[1] == 2), "M_reduced has {} columns; should have {}".format(M_test_reduced.shape[1], 2)

# Print Success
print("-" * 80)
print("Passed All Tests!")
print("-" * 80)


#################################
# TODO: d)
def plot_embeddings(M_reduced, word2Ind, words, filename=None):
    """ Plot in a scatterplot the embeddings of the words specified
        in the list "words".
        NOTE: do not plot all the words listed in M_reduced / word2Ind.
        Include a label next to each point.

        Params:
            M_reduced (numpy matrix of shape (number of unique words in the
            corpus , k)): matrix of k-dimensioal word embeddings
            word2Ind (dict): dictionary that maps word to indices for matrix M
            words (list of strings): words whose embeddings we want to
            visualize
    """

    # ------------------
    # Write your implementation here.
    x_val = []
    y_val = []

    for word in words:
        [x, y] = M_reduced[word2Ind[word]]
        x_val.append(x)
        y_val.append(y)

    fig, ax = plt.subplots()
    ax.scatter(x_val, y_val)

    for word in words:
        [x, y] = M_reduced[word2Ind[word]]
        ax.annotate(word, (x, y))

    if filename:
        plt.savefig(filename)
    # ------------------#


# ---------------------
# Run this sanity check
# Note that this not an exhaustive check for correctness.
# The plot produced should look like the "test solution plot" depicted below.
# ---------------------

print("-" * 80)
print("Outputted Plot:")

M_reduced_plot_test = np.array([[1, 1], [-1, -1], [1, -1], [-1, 1], [0, 0]])
word2Ind_plot_test = {
    'test1': 0, 'test2': 1, 'test3': 2, 'test4': 3, 'test5': 4}
words = ['test1', 'test2', 'test3', 'test4', 'test5']
plot_embeddings(M_reduced_plot_test, word2Ind_plot_test, words)

print("-" * 80)


#################################
# TODO: e)
# -----------------------------
# Run This Cell to Produce Your Plot
# ------------------------------

def read_corpus_pl():
    """ Read files from the specified Reuter's category.
        Params:
            category (string): category name
        Return:
            list of lists, with words from each of the processed files
    """
    pl196x_dir = nltk.data.find('corpora/pl196x')
    pl = pl196x.Pl196xCorpusReader(
        pl196x_dir, r'.*\.xml', textids='textids.txt', cat_file="cats.txt")
    tsents = pl.tagged_sents(fileids=pl.fileids(), categories='cats.txt')[:5000]

    return [[START_TOKEN] + [
        w[0].lower() for w in list(sent)] + [END_TOKEN] for sent in tsents]


def plot_unnormalized(corpus, words):
    M_co_occurrence, word2Ind_co_occurrence = compute_co_occurrence_matrix(
        corpus)
    M_reduced_co_occurrence = reduce_to_k_dim(M_co_occurrence, k=2)
    plot_embeddings(M_reduced_co_occurrence, word2Ind_co_occurrence, words, "unnormalized.png")


def plot_normalized(corpus, words):
    M_co_occurrence, word2Ind_co_occurrence = compute_co_occurrence_matrix(
        corpus)
    M_reduced_co_occurrence = reduce_to_k_dim(M_co_occurrence, k=2)
    # Rescale (normalize) the rows to make them each of unit-length
    M_lengths = np.linalg.norm(M_reduced_co_occurrence, axis=1)
    M_normalized = M_reduced_co_occurrence / M_lengths[:, np.newaxis]  # broadcasting
    plot_embeddings(M_normalized, word2Ind_co_occurrence, words, "normalized.png")


pl_corpus = read_corpus_pl()
words = [
    "sztuka", "śpiewaczka", "literatura", "poeta", "obywatel"]

plot_normalized(pl_corpus, words)
plot_unnormalized(pl_corpus, words)

#################################
# Section 2:
#################################
# Then run the following to load the word2vec vectors into memory.
# Note: This might take several minutes.
wv_from_bin_pl = KeyedVectors.load("../word2vec_100_3_polish.bin")


# -----------------------------------
# Run Cell to Load Word Vectors
# Note: This may take several minutes
# -----------------------------------


#################################
# TODO: a)
def get_matrix_of_vectors(wv_from_bin, required_words):
    """ Put the word2vec vectors into a matrix M.
        Param:
            wv_from_bin: KeyedVectors object; the 3 million word2vec vectors
                         loaded from file
        Return:
            M: numpy matrix shape (num words, 300) containing the vectors
            word2Ind: dictionary mapping each word to its row number in M
    """
    words = list(wv_from_bin.key_to_index.keys())
    print("Shuffling words ...")
    random.shuffle(words)
    words = words[:10000]
    print("Putting %i words into word2Ind and matrix M..." % len(words))
    word2Ind = {}
    M = []
    curInd = 0
    for w in words:
        try:
            M.append(wv_from_bin.get_vector(w))
            word2Ind[w] = curInd
            curInd += 1
        except KeyError:
            continue
    for w in required_words:
        try:
            M.append(wv_from_bin.get_vector(w))
            word2Ind[w] = curInd
            curInd += 1
        except KeyError:
            continue
    M = np.stack(M)
    print("Done.")
    return M, word2Ind


# -----------------------------------------------------------------
# Run Cell to Reduce 300-Dimensinal Word Embeddings to k Dimensions
# Note: This may take several minutes
# -----------------------------------------------------------------

#################################
# TODO: a)
M, word2Ind = get_matrix_of_vectors(wv_from_bin_pl, words)
M_reduced = reduce_to_k_dim(M, k=2)

words = ["sztuka", "śpiewaczka", "literatura", "poeta", "obywatel"]
plot_embeddings(M_reduced, word2Ind, words, "reduced.png")

#################################
# TODO: b)
# Polysemous Words
# ------------------
# Write your polysemous word exploration code here.
def polysemous_pl(word: str):
    polysemous = wv_from_bin_pl.most_similar(word)
    for key, similarity in polysemous:
        print(key, similarity)

polysemous_pl("stówa")
# słowa 0.6893048286437988
# cent 0.6367954015731812
# słowo 0.6246823072433472
# stówka 0.6103435158729553
# słówko 0.608944833278656
# pens 0.5825462937355042
# tów 0.5744858980178833
# wers 0.573552668094635
# centym 0.5726915597915649
# komunał 0.5709105730056763

polysemous_pl("baba")

polysemous_pl("staw")

# ------------------

#################################
# TODO: c)
# Synonyms & Antonyms
# ------------------
# Write your synonym & antonym exploration code here.

def synonyms_antonyms_pl(w1, w2, w3):
    w1_w2_dist = wv_from_bin_pl.distance(w1, w2)
    w1_w3_dist = wv_from_bin_pl.distance(w1, w3)

    print("Synonyms {}, {} have cosine distance: {}".format(w1, w2, w1_w2_dist))
    print("Antonyms {}, {} have cosine distance: {}".format(w1, w3, w1_w3_dist))

synonyms_antonyms_pl(w1 = "radosny", w2 = "pogodny", w3 = "smutny")
# Synonyms radosny, pogodny have cosine distance: 0.3306429386138916
# Antonyms radosny, smutny have cosine distance: 0.3478999137878418

synonyms_antonyms_pl(w1 = "cyfrowy", w2 = "elektroniczny", w3 = "analogowy")
# Synonyms cyfrowy, elektroniczny have cosine distance: 0.21015697717666626
# Antonyms cyfrowy, analogowy have cosine distance: 0.24363106489181519

synonyms_antonyms_pl(w1 = "lśniacy", w2 = "błyszczący", w3 = "matowy")


#################################
# TODO: d)
# Solving Analogies with Word Vectors
# ------------------

# ------------------
# Write your analogy exploration code here.
pprint.pprint(wv_from_bin_pl.most_similar(
    positive=["syn", "kobieta"], negative=["mezczyzna"]))
# [('córka', 0.6928777098655701),
#  ('dziecko', 0.6763085722923279),
#  ('matka', 0.6552439332008362),
#  ('żona', 0.6547046899795532),
#  ('siostra', 0.6358523368835449),
#  ('mąż', 0.6058387160301208),
#  ('dziewczę', 0.6008315086364746),
#  ('rodzic', 0.5781418681144714),
#  ('ojciec', 0.5779308676719666),
#  ('rodzeństwo', 0.5768202543258667)]

pprint.pprint(wv_from_bin_pl.most_similar(
    positive=["ksiezniczka", "dziecko"], negative=["krolowa"]))

pprint.pprint(wv_from_bin_pl.most_similar(
    positive=["ksiezniczka", "dziecko"], negative=["krolowa"]))


#################################
# TODO: e)
# Incorrect Analogy
# ------------------
# Write your incorrect analogy exploration code here.

pprint.pprint(wv_from_bin_pl.most_similar(
    positive=["chirurg", "kobieta"], negative=["mezczyzna"]))

# ------------------


#################################
# TODO: f)
# Guided Analysis of Bias in Word Vectors
# Here `positive` indicates the list of words to be similar to and
# `negative` indicates the list of words to be most dissimilar from.
# ------------------
pprint.pprint(wv_from_bin_pl.most_similar(
    positive=['kobieta', 'szef'], negative=['mezczyzna']))

# [('własika', 0.5678122639656067),
#  ('agent', 0.5483713150024414),
#  ('oficer', 0.5411549210548401),
#  ('esperów', 0.5383270978927612),
#  ('interpol', 0.5367037653923035),
#  ('antyterrorystyczny', 0.5327680110931396),
#  ('komisarz', 0.5326411128044128),
#  ('europolu', 0.5274547338485718),
#  ('bnd', 0.5271410346031189),
#  ('pracownik', 0.5215375423431396)]

pprint.pprint(wv_from_bin_pl.most_similar(
    positive=['mezczyzna', 'prezes'], negative=['kobieta']))

# [('wiceprezes', 0.6396454572677612),
#  ('czlonkiem', 0.5929950475692749),
#  ('przewodniczący', 0.5746127963066101),
#  ('czlonek', 0.5648552179336548),
#  ('przewodniczacym', 0.5586849451065063),
#  ('wiceprzewodniczący', 0.5560489892959595),
#  ('obowiazków', 0.5549101233482361),
#  ('obowiazani', 0.5544129610061646),
#  ('dyrektor', 0.5513691306114197),
#  ('obowiazany', 0.5471130609512329)]

#################################
# TODO: g)
# Independent Analysis of Bias in Word Vectors
# ------------------

pprint.pprint(wv_from_bin_pl.most_similar(
    positive=['kobieta', 'chirurg'], negative=['mezczyzna']))

#################################
# Section 3:
# English part
#################################
def load_word2vec():
    """ Load Word2Vec Vectors
        Return:
            wv_from_bin: All 3 million embeddings, each lengh 300
    """
    import gensim.downloader as api
    wv_from_bin = api.load("word2vec-google-news-300")
    vocab = list(wv_from_bin.key_to_index.keys())
    print("Loaded vocab size %i" % len(vocab))
    return wv_from_bin


wv_from_bin = load_word2vec()

#################################
# TODO:
# Find English equivalent examples for points b) to g).
#################################

#################################
# TODO: b)
# Polysemous Words
# ------------------
# Write your polysemous word exploration code here.

def polysemous_en(word: str):
    polysemous = wv_from_bin.most_similar(word)
    for key, similarity in polysemous:
        print(key, similarity)

polysemous_en("bark")
polysemous_en("squash")

# ------------------

#################################
# TODO: c)
# Synonyms & Antonyms
# ------------------
# Write your synonym & antonym exploration code here.

w1 = "happy"
w2 = "cheerful"
w3 = "sad"
w1_w2_dist = wv_from_bin.distance(w1, w2)
w1_w3_dist = wv_from_bin.distance(w1, w3)

print("Synonyms {}, {} have cosine distance: {}".format(w1, w2, w1_w2_dist))
print("Antonyms {}, {} have cosine distance: {}".format(w1, w3, w1_w3_dist))

#################################
# TODO: d)
# Solving Analogies with Word Vectors
# ------------------

# ------------------
# Write your analogy exploration code here.
pprint.pprint(wv_from_bin.most_similar(
    positive=["son", "woman"], negative=["man"]))

#################################
# TODO: e)
# Incorrect Analogy
# ------------------
# Write your incorrect analogy exploration code here.
pprint.pprint(wv_from_bin.most_similar(
    positive=["", ""], negative=[""]))
# ------------------


#################################
# TODO: f)
# Guided Analysis of Bias in Word Vectors
# Here `positive` indicates the list of words to be similar to and
# `negative` indicates the list of words to be most dissimilar from.
# ------------------
pprint.pprint(wv_from_bin.most_similar(
    positive=['woman', 'boss'], negative=['man']))
print()
pprint.pprint(wv_from_bin.most_similar(
    positive=['man', 'boss'], negative=['woman']))

#################################
# TODO: g)
# Independent Analysis of Bias in Word Vectors
# ------------------

