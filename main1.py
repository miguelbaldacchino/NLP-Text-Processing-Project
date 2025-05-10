from vrtparser import vrtParser
from preprocessing import preprocess
from ngrams import buildNgrams, buildLibraryNGrams
import time

path  = 'test.vrt'
parsed = vrtParser(path)
for i, sentence in enumerate(parsed):
    #if i <= 2:
    print(f'Sentence {i}: {sentence}\n')
    
flatten_corpus = []
for sentence in parsed:
    preprocessed = preprocess(sentence) 
    for word in preprocessed:
        flatten_corpus.extend(word)

print(flatten_corpus)

start_manual = time.perf_counter()
ngram = buildNgrams(flatten_corpus)
end_manual = time.perf_counter()

for title, value in ngram.items():
    print(title)
    for sub_value, count in value.items():
        print(f'{sub_value}: {count}')
    print('\n')
    
start_lib = time.perf_counter()
ngramlib = buildLibraryNGrams(flatten_corpus)
end_lib = time.perf_counter()

for title, value in ngramlib.items():
    print(title)
    for sub_value, count in value.items():
        print(f'{sub_value}: {count}')
    print('\n')
    
print(f"Corpus N-Gram Timing:")
print(f"Manual:  {(end_manual - start_manual)*1000:.4f} ms")
print(f"Library: {(end_lib - start_lib)*1000:.4f} ms\n")