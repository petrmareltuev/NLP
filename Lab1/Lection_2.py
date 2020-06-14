from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

my_book = "A wiki is run using wiki software, otherwise known as a wiki engine. A wiki engine is a type of content " \
          "management system, but it differs from most other such systems, including blog software, in that the " \
          "content is created without any defined owner or leader, and wikis have little inherent structure, " \
          "allowing structure to emerge according to the needs of the users.[2] There are dozens of different wiki " \
          "engines in use, both standalone and part of other software, such as bug tracking systems "

tokens = word_tokenize(my_book)
print(tokens)

stopwords_list = stopwords.words('english')
print(stopwords_list)

filtered_tokens = [t for t in tokens
                   if t not in stopwords_list]

print(filtered_tokens)
