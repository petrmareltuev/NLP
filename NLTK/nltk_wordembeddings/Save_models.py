import gensim
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

file_book = open('my_book.txt')
my_book = file_book.read()
sent_tokens = sent_tokenize(my_book)
tokens = []
for sent in sent_tokens:
    tokens.append([word.lower() for word in word_tokenize(sent)])
tokens = [word for word in tokens if not word in stopwords.words('english')]


# min-count: only include words in the model with a min-count
model = gensim.models.Word2Vec(tokens, min_count=1, size=300, workers=4, window=10, sg=1, negative=5, iter=10)
model.save("my_book.model")
model.wv.save_word2vec_format("my_book.vec", binary=False)

model2 = gensim.models.Word2Vec(tokens, min_count=5, size=300, workers=4, window=5, sg=1, negative=5, iter=10)
model2.wv.save_word2vec_format("model2.vec", binary=False)

model3 = gensim.models.Word2Vec(tokens, min_count=5, size=100, workers=4, window=10, sg=1, negative=5, iter=10)
model3.wv.save_word2vec_format("model3.vec", binary=False)

model4 = gensim.models.Word2Vec(tokens, min_count=5, size=300, workers=4, window=1, sg=1, negative=5, iter=10)
model4.wv.save_word2vec_format("model4.vec", binary=False)

model5 = gensim.models.Word2Vec(tokens, min_count=1, size=10, workers=4, window=10, sg=1, negative=5, iter=10)
model5.wv.save_word2vec_format("model5.vec", binary=False)