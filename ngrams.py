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
