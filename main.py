from vrtparser import vrtParser
from preprocessing import flattenPreprocessCorpus
from ngrams import buildNgrams, buildLibraryNGrams, displayNGrams
import time

path  = 'test.vrt'
parsed = vrtParser(path)
for i, sentence in enumerate(parsed):
    #if i <= 2:
    print(f'Sentence {i}: {sentence}\n')

# flattens and preprocesses corpus
flatten_corpus = flattenPreprocessCorpus(parsed)

print(flatten_corpus)

start_manual = time.perf_counter()
ngram = buildNgrams(flatten_corpus)
end_manual = time.perf_counter()

displayNGrams(ngram)
    
start_lib = time.perf_counter()
ngramlib = buildLibraryNGrams(flatten_corpus)
end_lib = time.perf_counter()

displayNGrams(ngramlib)
    
print(f"Corpus N-Gram Timing:")
print(f"Manual:  {(end_manual - start_manual)*1000:.4f} ms")
print(f"Library: {(end_lib - start_lib)*1000:.4f} ms\n")