import gensim, nltk
import numpy as np
import pandas as pd
import math
import codecs
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from nltk.tokenize import word_tokenize, sent_tokenize


def viz(pca=True):
    wv, vocabulary = load_embeddings("my_book.vec")

    if pca:
        pca = PCA(n_components=2, whiten=True)
        Y = pca.fit(wv[:300, :]).transform(wv[:300, :])
    else:
        tsne = TSNE(n_components=2, random_state=0)
        Y = tsne.fit_transform(wv[:200, :])

    np.set_printoptions(suppress=True)

    plt.scatter(Y[:, 0], Y[:, 1])
    for label, x, y in zip(vocabulary, Y[:, 0], Y[:, 1]):
        if label.lower() not in stopwords.words('english'):
            plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.show()


def load_embeddings(file_name):
    with codecs.open(file_name, 'r', 'utf-8') as f_in:
        vocabulary, wv = zip(*[line.strip().split(' ', 1) for line in f_in if len(line.strip().split()) != 2])
        wv = np.loadtxt(wv)
    return wv, vocabulary


def P_at_k(model_1, model_2):
    words_to_evaluate = ['water', 'way', 'earth', 'time', 'lock']
    k = len(words_to_evaluate)
    precision = 0
    for word in words_to_evaluate:
        precision += Is_similar(word, model_1, model_2)
    return precision / k


def Is_similar(word, model_1, model_2):
    for w1 in model_1.most_similar(word)[:5]:
        for w2 in model_2.most_similar(word)[:5]:
            if w1[0] == w2[0].lower():
                return 1
    return 0


model = gensim.models.KeyedVectors.load_word2vec_format("my_book.vec", binary=False)

print(model.most_similar('holmes'))
print(model.most_similar('adventures'))
print(model.most_similar('precisely'))
print()

viz(pca=False)

model_evaluate = gensim.models.KeyedVectors.load_word2vec_format("asoif_fastText.model", binary=False)

# window = 10, size = 300, min_count = 1
print(P_at_k(model, model_evaluate))

# window = 5
model2 = gensim.models.KeyedVectors.load_word2vec_format("model2.vec", binary=False)
print(P_at_k(model2, model_evaluate))

# size = 100
model3 = gensim.models.KeyedVectors.load_word2vec_format("model3.vec", binary=False)
print(P_at_k(model3, model_evaluate))

# window = 1
model4 = gensim.models.KeyedVectors.load_word2vec_format("model4.vec", binary=False)
print(P_at_k(model4, model_evaluate))

# size = 10, min_count = 1
model5 = gensim.models.KeyedVectors.load_word2vec_format("model5.vec", binary=False)
print(P_at_k(model5, model_evaluate))


file_book = open('my_book.txt')
my_book = file_book.read()
text_by_sent = sent_tokenize(my_book)
sents = [word.lower() for word in text_by_sent if not word in stopwords.words('english')]
sents = [word_tokenize(sent) for sent in sents]
tokens = []
for sent in sents:
    tokens = tokens + [word for word in sent]
unique_tokens = set(tokens)

idf ={}
N = len(sents)
for token in unique_tokens:
    idf[token] = 0
    for sent in sents:
        if token in sent:
            idf[token] += 1
    idf[token] = math.log10(N/idf[token])

sents_representation = []
for sent in sents:
    sent_vec = np.zeros(300)
    for word in sent:
        sent_vec = np.add(sent_vec, model.wv[word] * idf[word])
    sents_representation.append(np.divide(sent_vec, len(sent)))


def search(phrase):
    phrase_vec = np.zeros(300)
    for word in word_tokenize(phrase):
        phrase_vec = np.add(phrase_vec, model.wv[word] * idf[word])

    dot_product = np.abs([np.dot(phrase_vec, vec) / np.linalg.norm(phrase_vec) / np.linalg.norm(vec) for vec in sents_representation])
    df = pd.DataFrame(dot_product)
    for id in df.sort_values(0, ascending= False)[:5].index:
        print(text_by_sent[id])
    print()
    print()


search('stay at home')
search('sherlock holmes')
search('good day')
search('window shutters')
d