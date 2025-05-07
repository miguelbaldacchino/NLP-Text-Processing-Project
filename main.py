from preprocessing import preprocess
from ngrams import buildNgrams, buildLibraryNGrams
import time

sentences = ['Hello my name is Miguel, ok??? ,.. This, this is my testing testing case ok']

preprocessed_sentences = {}
ngrams_sentence = {}
ngrams_sentence_lib = {}
i = 0

for sentence in sentences:
    preprocessed = preprocess(sentence)  # returns a list of token lists (one per sentence)
    preprocessed_sentences[i] = preprocessed

    for idx, token_list in enumerate(preprocessed):
        start_manual = time.perf_counter()
        ngrams_sentence[(i, idx)] = buildNgrams(token_list)  # key is (sentence_id, sub-sentence_id)
        end_manual = time.perf_counter()

        start_lib = time.perf_counter()
        ngrams_sentence_lib[(i, idx)] = buildLibraryNGrams(token_list)
        end_lib = time.perf_counter()

        print(f"\nSentence {i}-{idx} timing:")
        print(f"Manual:  {(end_manual - start_manual)*1000:.4f} ms")
        print(f"Library: {(end_lib - start_lib)*1000:.4f} ms")
    i += 1

print("Preprocessed Sentences:")
print(preprocessed_sentences)

print("\nN-Grams:")
# Assuming ngrams_sentence is a dictionary where keys are n-grams and values are counts
for ngram, count in ngrams_sentence.items():    
    print(f"{ngram}: {count}")
    print('\n')

print("\nN-Grams Library:")
# Similarly, for the library-based n-grams
for ngram, count in ngrams_sentence_lib.items():
    print(f"{ngram}: {count}")
    print('\n')
