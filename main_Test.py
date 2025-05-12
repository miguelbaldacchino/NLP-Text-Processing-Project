from vrtparser import vrtParser
from preprocessing import flattenPreprocessCorpus
from ngrams import buildNgrams, buildLibraryNGrams, displayNGrams
import time
import os
from vanilla import VanillaLanguageModel

file_path = 'test.vrt'

parsed = vrtParser(file_path) 


# flattens and preprocesses corpus
flatten_corpus = flattenPreprocessCorpus(parsed)

ngram = buildNgrams(flatten_corpus)
print(ngram['Unigrams'])
model = VanillaLanguageModel(ngram)
print("Generated:", model.generateSentence())