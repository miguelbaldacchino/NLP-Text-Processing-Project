from collections import Counter

def buildNgrams(tokenized_sentence):
    # storing - key: ngram, value: count
    unigram_count = {}
    bigram_count = {}
    trigram_count = {}
    
    # iterate sentences
    for i in range(len(tokenized_sentence)):        # iterate words
        # unigram
        #   access word
        unigram = tokenized_sentence[i]
        #   update unigram count
        unigram_count[unigram] = unigram_count.get(unigram, 0) + 1 # handles keys that havent been initiated
    
        # bigram
        if i >= 1:
            bigram = (tokenized_sentence[i-1], tokenized_sentence[i])
            bigram_count[bigram] = bigram_count.get(bigram, 0) + 1

        # trigram
        if i >= 2:
            trigram = (tokenized_sentence[i-2], tokenized_sentence[i-1], tokenized_sentence[i])
            trigram_count[trigram] = trigram_count.get(trigram, 0) + 1

            
    return {
        'Unigrams': unigram_count,
        'Bigrams': bigram_count,
        'Trigrams':trigram_count
    }
    
def buildLibraryNGrams(tokenized_sentence):
    unigrams = Counter(tokenized_sentence)
    bigrams = Counter((tokenized_sentence[i], tokenized_sentence[i+1]) for i in range(len(tokenized_sentence)-1))
    trigrams = Counter((tokenized_sentence[i], tokenized_sentence[i+1], tokenized_sentence[i+2]) for i in range(len(tokenized_sentence)-2))
    
    return {
        'Unigrams': dict(unigrams),
        'Bigrams': dict(bigrams),
        'Trigrams': dict(trigrams)
    }
    