#import spacy
#nlp = spacy.load("en_core_web_md")

#doc = nlp(u'This is a sentence, which has two parts.')
#print(doc.text)
#print(doc.lang_)
#print(list(doc.sents))
'''
token_list = []

# create token list (word, documents' id where word occurs) and sort it
for i, fileid in enumerate(gutenberg.fileids()):
    file_tokens = [(w, i) for w in gutenberg.words(fileid)]
    # for boolean index we don't need duplicates
    file_tokens = list(set(file_tokens))
    token_list.extend(file_tokens)
token_list = sorted(token_list, key=itemgetter(0, 1))

# create the index; id of doc which word occurs in
inv_index = {}
for (word, doc_no) in token_list:
    if not word in inv_index:
        inv_index[word] = {'postings': set(), 'df': 0}
    inv_index[word]['postings'].add(doc_no)

# set document frequency (df), number of documents in which word occurs
for word, vals in inv_index.items():
    inv_index[word]['df'] = len(inv_index[word]['postings'])


# query -- AND
def process_query(query_words):
    res = inv_index[query_words[0]]['postings']
    for qw in query_words:
        res = res & inv_index[qw]['postings']
    return res


print('Caesar', inv_index['Caesar'])
print('Brutus', inv_index['Brutus'])
print('')
print("search ['Caesar','Brutus','hand']", process_query(['Caesar', 'Brutus', 'hand']))
print("search ['Caesar']", process_query(['Caesar']))
print("search ['Caesar', 'Jane']", process_query(['Caesar', 'Jane']))
'''

'''
import nltk

from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from operator import itemgetter

size_paragraph = 50

# sent_tokenizing book
file_book = open('my_book.txt')
my_book = file_book.read().decode('utf-8')
tokens = sent_tokenize(my_book)

# split book into paragraphs
documents = {}
for token_id, token in enumerate(tokens):
    doc_id = token_id / size_paragraph
    if doc_id not in documents:
        documents[doc_id] = ''
    documents[doc_id] += token
print("number of paragraphs: " + str(len(documents)))

token_list = []

for fileid in range(len(documents)):
    file_tokens = [(w, fileid) for w in word_tokenize(documents[fileid])]
    # for boolean index we don't need duplicates
    file_tokens = list(set(file_tokens))
    token_list.extend(file_tokens)
token_list = sorted(token_list, key=itemgetter(0, 1))

# create the index; id of doc which word occurs in
inv_index = {}
for (word, doc_no) in token_list:
    if not word in inv_index:
        inv_index[word] = {'postings': set(), 'df': 0}
    inv_index[word]['postings'].add(doc_no)


# set document frequency (df), number of documents in which word occurs
for word, values in inv_index.items():
    inv_index[word]['df'] = len(inv_index[word]['postings'])


def search(param):
    words = word_tokenize(str(param))
    print("Search for " + str(words))
    try:
        res = inv_index[words[0]]['postings']
        for qw in words:
            res = res & inv_index[qw]['postings']
    except KeyError as ex:
        res = "This word did not found: " + str(ex.message)
    return res


print(search('gates'))
print(search('of'))
print(search('hell'))
print(search('gates of hell'))
print(search('Arnold Schwarzenegger'))
'''