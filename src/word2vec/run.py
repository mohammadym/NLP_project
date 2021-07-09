#!/usr/bin/env python

import os
import csv

import numpy as np

from utils.treebank import TeleText
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
import time
import os
import arabic_reshaper
from bidi.algorithm import get_display

from word2vec import *
from sgd import *

# Check Python Version
import sys

assert sys.version_info[0] == 3
assert sys.version_info[1] >= 5

labels = ['news', 'sport']


# Reset the random seed to make sure that everyone gets the same results
random.seed(314)
# if not os.path.exists("../models/word2vec"):
#     os.mkdir("../models/word2vec")


def word2vec_model(label):
    dataset = TeleText(label=label)
    tokens = dataset.tokens()
    nWords = len(tokens)

    # We are going to train 10-dimensional vectors for this assignment
    dimVectors = 10

    # Context size
    C = 5

    # Reset the random seed to make sure that everyone gets the same results
    random.seed(31415)
    np.random.seed(9265)

    startTime = time.time()
    wordVectors = np.concatenate(
        ((np.random.rand(nWords, dimVectors) - 0.5) /
         dimVectors, np.zeros((nWords, dimVectors))),
        axis=0)
    wordVectors = sgd(
        lambda vec: word2vec_sgd_wrapper(skipgram, tokens, vec, dataset, C,
                                         negSamplingLossAndGradient),
        wordVectors, 0.3, 40000, None, True, PRINT_EVERY=10, label=label)
    # Note that normalization is not called here. This is not a bug,
    # normalizing during training loses the notion of length.

    print("sanity check: cost at convergence should be around or below 10")
    print("training took %d seconds" % (time.time() - startTime))

    # concatenate the input and output word vectors
    wordVectors = np.concatenate(
        (wordVectors[:nWords, :], wordVectors[nWords:, :]),
        axis=0)
    return wordVectors, tokens, wordVectors


for label in labels:
    print(label)
    word2vect_vectors, tokens, wordVectors = word2vec_model(label)

    with open('reports/word2vec/top_{}_words.txt'.format(label), 'r') as top_file:
        visualizeWords = [word.replace('\n', '') for word in top_file.readlines()]
    visualizeIdx = []
    mywords = []
    for word in visualizeWords:
        if word in tokens:
            visualizeIdx.append(tokens[word])
            mywords.append(word)
    visualizeWords = mywords
    visualizeIdx = visualizeIdx[:15]
    visualizeWords = visualizeWords[:15]
    visualizeVecs = wordVectors[visualizeIdx, :]
    temp = (visualizeVecs - np.mean(visualizeVecs, axis=0))
    covariance = 1.0 / len(visualizeIdx) * temp.T.dot(temp)
    U, S, V = np.linalg.svd(covariance)
    coord = temp.dot(U[:, 0:2])
    for i in range(len(visualizeIdx)):
        plt.text(coord[i, 0], coord[i, 1], get_display(arabic_reshaper.reshape('{}'.format(visualizeWords[i]))),
                 bbox=dict(facecolor='green', alpha=0.1))

    plt.xlim((np.min(coord[:, 0]), np.max(coord[:, 0])))
    plt.ylim((np.min(coord[:, 1]), np.max(coord[:, 1])))

    if not os.path.exists('reports/word2vec'):
        os.mkdir('reports/word2vec')
    print('saving ,', label)
    plt.savefig('reports/word2vec/word_{}_vectors.png'.format(label))
