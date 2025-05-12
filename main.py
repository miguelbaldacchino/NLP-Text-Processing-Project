from extraction import loadCorpus
from preprocessing import flattenPreprocessCorpus
from ngrams import buildNgrams, buildLibraryNGrams, displayNGrams
import time
from laplace import LaplaceLanguageModel
from unk import wordCounter, replaceRareWords

parsed = loadCorpus('Maltese-Corpus', 10)

for i, sentence in enumerate(parsed):
    if i <= 2:
        print(f'Sentence {i}: {sentence}\n')


# flattens and preprocesses corpus
flatten_corpus = flattenPreprocessCorpus(parsed)
words_count = wordCounter(flatten_corpus)
new_corpus = replaceRareWords(parsed, words_count)
#print(flatten_corpus)
flatten_corpus = flattenPreprocessCorpus(new_corpus)

start_manual = time.perf_counter()
ngram = buildNgrams(flatten_corpus)
end_manual = time.perf_counter()

#displayNGrams(ngram)
    
start_lib = time.perf_counter()
#ngramlib = buildLibraryNGrams(flatten_corpus)
end_lib = time.perf_counter()

#displayNGrams(ngramlib)
    
print(f"Corpus N-Gram Timing:")
print(f"Manual:  {(end_manual - start_manual)*1000:.4f} ms")
print(f"Library: {(end_lib - start_lib)*1000:.4f} ms\n")

model = LaplaceLanguageModel(ngram)
print("Generated:", model.generateSentence())
print("Generated:", model.generateSentence())
print("Generated:", model.generateSentence())
print("Generated:", model.generateSentence())
print("Generated:", model.generateSentence())
print("Generated:", model.generateSentence())
print("Generated:", model.generateSentence())
print("Generated:", model.generateSentence())
print("Generated:", model.generateSentence())
print("Generated:", model.generateSentence())
print("Generated:", model.generateSentence())
print("Generated:", model.generateSentence())