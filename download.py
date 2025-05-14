from extraction import loadCorpus, splitCorpus
from preprocessing import flattenCorpus, preprocess
from ngrams import buildNgrams, buildLibraryNGrams, displayNGrams
import time
from laplace import LaplaceLanguageModel
from unk import wordCounter, replaceRareWords
from pickles import savingData
import psutil

def memory_usage():
    process = psutil.Process()
    mem = process.memory_info().rss / 1024 / 1024  # Memory usage in MB
    return mem

saving_data = savingData()

parsed = loadCorpus('Maltese-Corpus', 10)

corpus = []
for sentence in parsed:
    corpus.extend(preprocess(sentence))  # preserves list of token lists (sentences)
train_parsed, test_parsed = splitCorpus(corpus)

for i, sentence in enumerate(train_parsed):
    if i <= 2:
        print(f'Sentence {i}: {sentence}\n')

flatten_train = flattenCorpus(train_parsed)
print('Finished Preprocess')

train_words_count = wordCounter(flatten_train)
print('Finished Word Count')

unk_train_corpus = replaceRareWords(train_parsed, train_words_count)
print('Finished Replacing Rare Words')

flatten_unk_train = flattenCorpus(unk_train_corpus)

flatten_test = flattenCorpus(test_parsed)
test_words_count = wordCounter(flatten_test)
unk_test_corpus = replaceRareWords(test_parsed, test_words_count)
flatten_unk_test = flattenCorpus(unk_test_corpus)

ngrams = buildNgrams(flatten_train)
print('Finished building manual N-Gram')

unk_ngrams = buildNgrams(flatten_unk_train)
print('Finished building UNK N-Gram')

ngramlib = buildLibraryNGrams(flatten_train)
print('Finished building library N-Gram')

saving_data.save('ngram_unk.pkl', unk_ngrams)

saving_data.save('ngram.pkl', ngrams)

saving_data.save('train.pkl', train_parsed)
saving_data.save('test.pkl', test_parsed)

saving_data.save('unk_train.pkl', unk_train_corpus)
saving_data.save('unk_test.pkl', unk_test_corpus)

model = LaplaceLanguageModel(unk_ngrams)
for _ in range(5):  
    print("Generated:", model.generateSentence())
    


