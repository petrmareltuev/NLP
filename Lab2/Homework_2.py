import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from operator import itemgetter

size_paragraph = 50

# sent_tokenizing book
file_book = open('my_book.txt')
my_book = file_book.read()
tokens = sent_tokenize(my_book)

# split book into paragraphs
documents = {}
for token_id, token in enumerate(tokens):
    doc_id = int(token_id / size_paragraph)
    if doc_id not in documents:
        documents[doc_id] = ''
    documents[doc_id] += token
print("number of paragraphs: " + str(len(documents)))

# word_dictionary = {word: {id_doc: []}}
words_dictionary = {}
for fileid in documents:
    for word_id, word in enumerate(word_tokenize(documents[fileid])):
        if word not in words_dictionary:
            words_dictionary[word] = {}
        if fileid not in words_dictionary[word]:
            words_dictionary[word][fileid] = []
        words_dictionary[word][fileid].append(word_id)


def search(param):
    words = word_tokenize(str(param))
    print("Search for " + str(words))
    doc_ids = ()
    try:
        doc_ids = set(words_dictionary[words[0]])
        for word in words:
            #print("docs in '" + word + "': " + str(set(words_dictionary[word])))
            doc_ids = doc_ids & set(words_dictionary[word])
            if len(doc_ids) == 0:
                print("Nothing found")

        for doc_id in doc_ids:
            lists =[]
            for word in words:
                lists.append(words_dictionary[word][doc_id])
            #print(lists)
            window = 7
            doc_1 = lists[0]
            for doc_list in lists:
                tmp = []
                for number in doc_list:
                    for i in range(-window, window):
                        if number+i in doc_1:
                            tmp.append(number+i)
                doc_1 = tmp

            if len(doc_1) != 0:
                print("Found in " + str(doc_id) + " document. In " + str(doc_1) + " positions")

    except KeyError as ex:
        print(str(ex) + " not found")


search("Church hell")
search("patience hell")
search("Arnold Schwarzenegger")
search("gates of hell")
