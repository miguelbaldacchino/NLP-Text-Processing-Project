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

mem_before_read = memory_usage()
start_read = time.perf_counter()
parsed = loadCorpus('Maltese-Corpus', 10)
end_read = time.perf_counter()
mem_after_read = memory_usage()

corpus = []
for sentence in parsed:
    corpus += preprocess(sentence)
train_parsed, test_parsed = splitCorpus(corpus)

for i, sentence in enumerate(train_parsed):
    if i <= 2:
        print(f'Sentence {i}: {sentence}\n')


# flattens and preprocesses corpus
mem_before_preprocess = memory_usage()
start_preprocess = time.perf_counter()
flatten_train = flattenCorpus(train_parsed)
print('Finished Preprocess')
end_preprocess = time.perf_counter()
mem_after_preprocess = memory_usage()

mem_before_count = memory_usage()
train_words_count = wordCounter(flatten_train)
print('Finished Word Count')
mem_after_count = memory_usage()

mem_before_replace = memory_usage()
unk_train_corpus = replaceRareWords(train_parsed, train_words_count)
print('Finished Replacing Rare Words')
mem_after_replace = memory_usage()

flatten_unk_train = flattenCorpus(unk_train_corpus)

flatten_test = flattenCorpus(test_parsed)
test_words_count = wordCounter(flatten_test)
unk_test_corpus = replaceRareWords(test_parsed, test_words_count)
flatten_unk_test = flattenCorpus(unk_test_corpus)

mem_before_ngram = memory_usage()
start_manual = time.perf_counter()
ngrams = buildNgrams(flatten_train)
print('Finished building manual N-Gram')
end_manual = time.perf_counter()
mem_after_ngram = memory_usage()

unk_ngrams = buildNgrams(flatten_unk_train)
print('Finished building UNK N-Gram')

mem_before_lib = memory_usage()    
start_lib = time.perf_counter()
ngramlib = buildLibraryNGrams(flatten_train)
print('Finished building library N-Gram')
end_lib = time.perf_counter()
mem_after_lib = memory_usage()

saving_data.save('ngram_unk.pkl', unk_ngrams)

saving_data.save('ngram.pkl', ngrams)

saving_data.save('train.pkl', train_parsed)
saving_data.save('test.pkl', test_parsed)

saving_data.save('unk_train.pkl', unk_train_corpus)
saving_data.save('unk_test.pkl', unk_test_corpus)

model = LaplaceLanguageModel(unk_ngrams)
for _ in range(5):  
    print("Generated:", model.generateSentence())
    

# Print time and memory usage
print(f"Corpus N-Gram Timing:")
print(f"Manual:  {(end_manual - start_manual)*1000:.4f} ms")
print(f"Library: {(end_lib - start_lib)*1000:.4f} ms\n")

print(f"Reading Corpus:")
print(f"Time:  {(end_read - start_read)*1000:.4f} ms")
print(f"Memory: {mem_after_read - mem_before_read:.4f} MB\n")

print(f"Preprocessing Corpus:")
print(f"Time:  {(end_preprocess - start_preprocess)*1000:.4f} ms")
print(f"Memory Change: {mem_after_preprocess - mem_before_preprocess:.4f} MB\n")

print(f"Word Counting:")
print(f"Memory Change: {mem_after_count - mem_before_count:.4f} MB\n")

print(f"Rare Word Replacement:")
print(f"Memory Change: {mem_after_replace - mem_before_replace:.4f} MB\n")

print(f"Manual N-Grams Generation:")
print(f"Memory Change: {mem_after_ngram - mem_before_ngram:.4f} MB\n")

print(f"Library N-Grams Generation:")
print(f"Memory Change: {mem_after_lib - mem_before_lib:.4f} MB\n")

print(f"Corpus size: {len(parsed)} sentences")
print(f"Training Corpus size: {len(train_parsed)} sentences")
print(f"Test Corpus size: {len(test_parsed)} sentences\n")

