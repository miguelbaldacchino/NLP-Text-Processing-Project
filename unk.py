def wordCounter(corpus):
    word_count = {}
    for word in corpus:
        if word in word_count:
            word_count[word] += 1
        else:
            word_count[word] = 1
    return word_count

def replaceRareWords(corpus, word_count):
    updated_corpus = []
    for sentence in corpus:
        new_sentence = []
        for word in sentence:
            if word_count.get(word, 0) <= 2:
                new_sentence.append('<UNK>')
            else:
                new_sentence.append(word)
        updated_corpus.append(new_sentence)
    return updated_corpus


