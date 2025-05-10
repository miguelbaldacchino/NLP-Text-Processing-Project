from preprocessing import preprocess
from ngrams import buildNgrams, buildLibraryNGrams
from vrtparser import vrtParser
import time

# list of lists of strings
sentences = vrtParser('test.vrt')
print('sentences: \n\n', sentences)
flatten_corpus = []
for sentence in sentences:
    preprocessed = preprocess(sentence)  # returns a list of token lists (one per sentence)
    for sub_sentence in preprocessed:
        flatten_corpus.extend(sub_sentence)
        
print('FLAT\n\n\n', flatten_corpus)
start_manual = time.perf_counter()
ngrams_result = buildNgrams(flatten_corpus)
end_manual = time.perf_counter()

start_lib = time.perf_counter()
ngrams_result_lib = buildLibraryNGrams(flatten_corpus)
end_lib = time.perf_counter()

print(f"Corpus N-Gram Timing:")
print(f"Manual:  {(end_manual - start_manual)*1000:.4f} ms")
print(f"Library: {(end_lib - start_lib)*1000:.4f} ms\n")


# Print results
print("Manual N-Grams:")
for ngram_type, counts in ngrams_result.items():
    print(f"\n{ngram_type}:")
    for ngram, count in list(counts.items())[:10]:  # Print first 10 for brevity
        print(f"{ngram}: {count}")

print("\nLibrary N-Grams:")
for ngram_type, counts in ngrams_result_lib.items():
    print(f"\n{ngram_type}:")
    for ngram, count in list(counts.items())[:10]:  # Print first 10 for brevity
        print(f"{ngram}: {count}")


from collections import Counter

def buildNgrams(tokenized_sentence):
    unigram_count = {}
    bigram_count = {}
    trigram_count = {}


    for i in range(len(tokenized_sentence)):
        # Unigram
        unigram = tokenized_sentence[i]
        unigram_count[unigram] = unigram_count.get(unigram, 0) + 1

        # Bigram (no crossing sentence boundaries)
        if i >= 1:
            if tokenized_sentence[i-1] != '</s>':
                bigram = (tokenized_sentence[i-1], tokenized_sentence[i])
                bigram_count[bigram] = bigram_count.get(bigram, 0) + 1

        # Trigram (no crossing sentence boundaries)
        if i >= 2:
            if (tokenized_sentence[i-2] != '</s>' and
                tokenized_sentence[i-1] != '</s>'):
                trigram = (tokenized_sentence[i-2], tokenized_sentence[i-1], tokenized_sentence[i])
                trigram_count[trigram] = trigram_count.get(trigram, 0) + 1

    return {
        'Unigrams': unigram_count,
        'Bigrams': bigram_count,
        'Trigrams': trigram_count
    }

def buildLibraryNGrams(tokenized_sentence):
    unigrams = Counter(tokenized_sentence)


    # Bigram: skip if either token is </s>
    bigrams = Counter(
        (tokenized_sentence[i], tokenized_sentence[i+1])
        for i in range(len(tokenized_sentence)-1)
        if tokenized_sentence[i] != '</s>'
    )

    # Trigram: skip if any token is </s>
    trigrams = Counter(
        (tokenized_sentence[i], tokenized_sentence[i+1], tokenized_sentence[i+2])
        for i in range(len(tokenized_sentence)-2)
        if tokenized_sentence[i] != '</s>' and
        tokenized_sentence[i+1] != '</s>'
    )

    return {
        'Unigrams': dict(unigrams),
        'Bigrams': dict(bigrams),
        'Trigrams': dict(trigrams)
    }


its iterating the whole corpus, because <s> </s> counts are 5 (as should be), but not all the words appear in the unigram etc
