#!/usr/bin/env python

import random
import numpy as np
from utils.treebank import TeleSentiment
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import time
import json

from word2vec import *
from sgd import *

# Check Python Version
import sys
assert sys.version_info[0] == 3
assert sys.version_info[1] >= 5

labels=["news","sport"]

# Reset the random seed to make sure that everyone gets the same results
random.seed(314)
for label in labels:
    print("start label :"+label)
    dataset = TeleSentiment(label=label)
    tokens = dataset.tokens()
    nWords = len(tokens)

    # We are going to train 10-dimensional vectors for this assignment
    dimVectors = 10

    # Context size
    C = 5

    # Reset the random seed to make sure that everyone gets the same results
    random.seed(31415)
    np.random.seed(9265)

    startTime=time.time()
    wordVectors = np.concatenate(
       ((np.random.rand(nWords, dimVectors) - 0.5) /
         dimVectors, np.zeros((nWords, dimVectors))),
        axis=0)
    wordVectors = sgd(
        lambda vec: word2vec_sgd_wrapper(skipgram, tokens, vec, dataset, C,
            negSamplingLossAndGradient),
        wordVectors, 0.3, 40000, None, True, PRINT_EVERY=10)
    # Note that normalization is not called here. This is not a bug,
    # normalizing during training loses the notion of length.

    print("sanity check: cost at convergence should be around or below 10")
    print("training took %d seconds" % (time.time() - startTime))

    # concatenate the input and output word vectors
    wordVectors = np.concatenate(
        (wordVectors[:nWords,:], wordVectors[nWords:,:]),
        axis=0)

    with open('../../models/word2vec/' + label + '.word2vec.npy', 'wb') as f:
        np.save(f, wordVectors)
    a_file = open('../../models/word2vec/' + label + '.word2vec.tokens.json', "w")
    json.dump(tokens, a_file)
    a_file.close()
