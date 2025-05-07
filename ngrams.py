def build_ngrams(tokenized_sentence):
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
        'Unigram': unigram_count,
        'Bigram': bigram_count,
        'Trigram':trigram_count
    }
    
    
            