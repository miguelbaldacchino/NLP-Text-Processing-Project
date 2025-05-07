from preprocessing import preprocess
from ngrams import build_ngrams

sentences = ['Hello my name, shania is Miguel Miguel Miguel,   and I like  Shania... .,. I   really do!']

preprocessed_sentences = {}
ngrams_by_sentence = {}
i = 0

for sentence in sentences:
    preprocessed = preprocess(sentence)  # returns a list of token lists (one per sentence)
    preprocessed_sentences[i] = preprocessed

    for idx, token_list in enumerate(preprocessed):
        ngrams_by_sentence[(i, idx)] = build_ngrams(token_list)  # key is (sentence_id, sub-sentence_id)

    i += 1

print("Preprocessed Sentences:")
print(preprocessed_sentences)

print("\nN-Grams:")
print(ngrams_by_sentence)
