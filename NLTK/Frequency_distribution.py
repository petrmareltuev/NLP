import nltk
import matplotlib
from nltk.book import *
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize

file = open('my_book.txt')
my_book = file.read()
tokens = word_tokenize(my_book)
my_text = nltk.Text(tokens)


f_dist = FreqDist(my_text)
f_dist2 = FreqDist(text1)
most_commons_my = [word for word, counts in f_dist.most_common(50)]
most_commons_moby = [word for word, counts in f_dist2.most_common(50)]
f_dist.plot(50)

differences_my = []
for word in most_commons_my:
    if not word in most_commons_moby:
        differences_my.append(word)
differences_moby = []
for word in most_commons_moby:
    if not word in most_commons_my:
        differences_moby.append(word)


print(' \n\n\nMost Common My\n' + ' | '.join(most_commons_my))
print(' \nMost Common Moby\n' + ' | '.join(most_commons_moby))
print(' \nDifferences\n' + ' | '.join(differences_my))
print(' | '.join(differences_moby))

print('\n')
for word, counts in f_dist.most_common(50):
    print("{0:15} {1:d}, {2:d}".format(word, counts, f_dist2[word]))
