import gensim, nltk
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize, sent_tokenize
import spacy
from sklearn.cluster import KMeans
file_book = open('my_book.txt')
my_book = file_book.read()
sents = sent_tokenize(my_book)

nlp = spacy.load("en_core_web_sm")
sents_spacy = list(nlp.pipe(sents))

v = sents_spacy[0].vector
k = 10

X = [sent.vector for sent in sents_spacy]

kmeans = KMeans(n_clusters=k, random_state=1).fit(X)
np.sqrt(kmeans.inertia_)


#
# with open("exercises/en/countries.json") as f:
#     COUNTRIES = json.loads(f.read())
# with open("exercises/en/country_text.txt") as f:
#     TEXT = f.read()
#
# nlp = spacy.load("en_core_web_lg")
# matcher = PhraseMatcher(nlp.vocab)
# patterns = list(nlp.pipe(COUNTRIES))
# matcher.add("COUNTRY", None, *patterns)
#
# # Create a doc and reset existing entities
# doc = nlp(TEXT)
# doc.ents = []
#
# # Iterate over the matches
# for match_id, start, end in matcher(doc):
#     # Create a Span with the label for "GPE"
#     span = ____(____, ____, ____, label=____)
#
#     # Overwrite the doc.ents and add the span
#     doc.ents = list(doc.ents) + [____]
#
#     # Get the span's root head token
#     span_root_head = ____.____.____
#     # Print the text of the span root's head token and the span text
#     print(span_root_head.____, "-->", span.text)
#
# # Print the entities in the document
# print([(ent.text, ent.label_) for ent in doc.ents if ent.label_ == "GPE"])
