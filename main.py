from extraction import loadCorpus, splitCorpus
from preprocessing import flattenPreprocessCorpus
from ngrams import buildNgrams, buildLibraryNGrams, displayNGrams
import time
from laplace import LaplaceLanguageModel
from unk import wordCounter, replaceRareWords
from saving import savingData

saving_data = savingData()
parsed = loadCorpus('Maltese-Corpus', 5)
train_parsed, test_parsed = splitCorpus(parsed)

for i, sentence in enumerate(train_parsed):
    if i <= 2:
        print(f'Sentence {i}: {sentence}\n')


# flattens and preprocesses corpus
flatten_train = flattenPreprocessCorpus(train_parsed)
train_words_count = wordCounter(flatten_train)
unk_train_corpus = replaceRareWords(parsed, train_words_count)
flatten_train = flattenPreprocessCorpus(unk_train_corpus)

flatten_test = flattenPreprocessCorpus(test_parsed)
test_words_count = wordCounter(flatten_test)
unk_test_corpus = replaceRareWords(parsed, test_words_count)
flatten_test = flattenPreprocessCorpus(unk_test_corpus)

saving_data.save('flatten_train.pkl', flatten_train)
saving_data.save('flatten_test.pkl', flatten_test)

start_manual = time.perf_counter()
unk_ngrams = buildNgrams(flatten_train)
end_manual = time.perf_counter()

saving_data.save('unigram_unk.pkl', unk_ngrams['Unigrams'])
saving_data.save('bigram_unk.pkl', unk_ngrams['Bigrams'])
saving_data.save('trigram_unk.pkl', unk_ngrams['Trigrams'])

#displayNGrams(ngram)
    
start_lib = time.perf_counter()
#ngramlib = buildLibraryNGrams(flatten_corpus)
end_lib = time.perf_counter()

#displayNGrams(ngramlib)
    
print(f"Corpus N-Gram Timing:")
print(f"Manual:  {(end_manual - start_manual)*1000:.4f} ms")
print(f"Library: {(end_lib - start_lib)*1000:.4f} ms\n")

model = LaplaceLanguageModel(unk_ngrams)
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