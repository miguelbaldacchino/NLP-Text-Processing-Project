
from preprocessing import preprocess
from ngrams import buildNgrams, buildLibraryNGrams
from vrtparser import vrtParser
import time

# list of lists of strings
sentences = vrtParser('test.vrt')

preprocessed_sentences = {}
ngrams_sentence = {}
ngrams_sentence_lib = {}
i = 0

for sentence in sentences:
    preprocessed = preprocess(sentence)  # returns a list of token lists (one per sentence)
    preprocessed_sentences[i] = preprocessed
    i +=1

print('Preprocessed Sentences \n\n\n', preprocessed_sentences)

for sentence in preprocessed_sentences:
    print('N-Gram 1 \n\n\n', buildNgrams(sentence))