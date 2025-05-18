from extraction import loadCorpus, splitCorpus
from preprocessing import flattenCorpus, preprocess
from ngrams import buildNgrams, buildLibraryNGrams
from laplace import LaplaceLanguageModel
from unk import wordCounter, replaceRareWords
from pickles import savingData
import time
import psutil

def memory_usage():
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024  # in MB

def print_section(title):
    print(f"\n{'='*10} {title} {'='*10}")

saving_data = savingData()
stats = {}

print_section("Corpus Reading")
mem_before = memory_usage()
start = time.perf_counter()
parsed = loadCorpus('Maltese-Corpus', 3)
end = time.perf_counter()
mem_after = memory_usage()
stats['Corpus Read'] = (end - start, mem_after - mem_before)
print("Corpus successfully loaded.")

print_section("Preprocessing")
mem_before = memory_usage()
start = time.perf_counter()
corpus = [word for sentence in parsed for word in preprocess(sentence)]
end = time.perf_counter()
mem_after = memory_usage()
stats['Preprocessing'] = (end - start, mem_after - mem_before)
print("Preprocessing complete.")

print_section("Splitting Corpus")
train_parsed, test_parsed = splitCorpus(corpus)
print("Training and testing data split.")

print_section("Flattening Training Corpus")
mem_before = memory_usage()
start = time.perf_counter()
flatten_train = flattenCorpus(train_parsed)
end = time.perf_counter()
mem_after = memory_usage()
stats['Flatten Train'] = (end - start, mem_after - mem_before)
print("Training corpus flattened.")

print_section("Word Counting")
mem_before = memory_usage()
start = time.perf_counter()
train_words_count = wordCounter(flatten_train)
end = time.perf_counter()
mem_after = memory_usage()
stats['Word Count'] = (end - start, mem_after - mem_before)
print("Word counting complete.")

print_section("Replacing Rare Words with UNK")
mem_before = memory_usage()
start = time.perf_counter()
unk_train_corpus = replaceRareWords(train_parsed, train_words_count)
end = time.perf_counter()
mem_after = memory_usage()
stats['UNK Replacement'] = (end - start, mem_after - mem_before)
print("UNK replacement done.")

print_section("Flattening UNK Corpus")
flatten_unk_train = flattenCorpus(unk_train_corpus)
print("UNK corpus flattened.")

print_section("Building Manual N-Gram")
mem_before = memory_usage()
start = time.perf_counter()
ngrams = buildNgrams(flatten_train)
end = time.perf_counter()
mem_after = memory_usage()
stats['Manual N-Gram'] = (end - start, mem_after - mem_before)
print("Manual N-gram model built.")

print_section("Building UNK N-Gram")
mem_before = memory_usage()
start = time.perf_counter()
unk_ngrams = buildNgrams(flatten_unk_train)
end = time.perf_counter()
mem_after = memory_usage()
stats['UNK N-Gram'] = (end - start, mem_after - mem_before)
print("UNK N-gram model built.")

print_section("Building Library N-Gram")
mem_before = memory_usage()
start = time.perf_counter()
ngramlib = buildLibraryNGrams(flatten_train)
end = time.perf_counter()
mem_after = memory_usage()
stats['Library N-Gram'] = (end - start, mem_after - mem_before)
print("Library N-gram model built.")

print_section("Processing Test Data")
flatten_test = flattenCorpus(test_parsed)
test_words_count = wordCounter(flatten_test)
unk_test_corpus = replaceRareWords(test_parsed, test_words_count)
flatten_unk_test = flattenCorpus(unk_test_corpus)
print("Test data processed and UNK replaced.")

print_section("Saving Data")
saving_data.save('ngram_unk.pkl', unk_ngrams)
saving_data.save('ngram.pkl', ngrams)
saving_data.save('train.pkl', train_parsed)
saving_data.save('test.pkl', test_parsed)
saving_data.save('unk_train.pkl', unk_train_corpus)
saving_data.save('unk_test.pkl', unk_test_corpus)
print("All models and corpora saved.")

print_section("TIME & MEMORY USAGE REPORT")
for stage, (time_sec, mem_mb) in stats.items():
    print(f"{stage} | Time: {time_sec:.2f}s | Memory Change: {mem_mb:.2f}MB")

print_section("N-GRAM TIMING COMPARISON")
manual_time = stats['Manual N-Gram'][0]
library_time = stats['Library N-Gram'][0]
speedup = library_time / manual_time if manual_time else float('inf')
print(f"Manual N-Gram Time : {manual_time:.4f} seconds")
print(f"Library N-Gram Time: {library_time:.4f} seconds")
print(f"Speedup (Library / Manual): {speedup:.2f}x")

print_section("CORPUS STATISTICS")
total_sentences = len(corpus)
total_words = sum(len(sentence) for sentence in corpus)
vocab = set(word for sentence in corpus for word in sentence)
vocab_size = len(vocab)
print(f"Total sentences after preprocessing: {total_sentences}")
print(f"Total number of words: {total_words}")
print(f"Vocabulary size: {vocab_size}")
