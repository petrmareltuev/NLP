'''import math
SaS = [115, 10, 2, 0]
Pap = [58, 7, 0, 0]
WH = [20, 11, 6, 38]

def log_vec(vector):
    result_vector = [math.log10(i) for i in vector if (i > 0)]
    return result_vector


def L2_norm(vector):
    sum = 0
    for el in vector:
        sum += el ** 2
    res = np.sqrt(sum)
    return res


print(log_vec(SaS))
print(log_vec(Pap))
print(log_vec(WH))

print (L2_norm(SaS))'''



import nltk, gensim
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import gutenberg

def train_model(fileid):
    """
        training a gensim model, see also: https://radimrehurek.com/gensim/models/word2vec.html
    """
    # min-count: only include words in the model with a min-count
    return gensim.models.Word2Vec(gutenberg.sents(fileid), min_count=5, size=300,
                                  workers=4, window=10, sg=1, negative=5, iter=10)


model = train_model('shakespeare-hamlet.txt')
model

print(model.most_similar(positive=['Hamlet']))
print()
print(model.most_similar(positive=['King']))
print()
print(model.most_similar(positive=['great']))

