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
    return process.memory_info().rss / 1024 / 1024  # MB

saving_data = savingData()
stats = {}
# --- Corpus Reading ---
mem_before = memory_usage()
start = time.perf_counter()
parsed = loadCorpus('Maltese-Corpus', 300)
end = time.perf_counter()
mem_after = memory_usage()
stats['Corpus Read'] = (end - start, mem_after - mem_before)
print('done')

# --- Preprocessing ---
mem_before = memory_usage()
start = time.perf_counter()
corpus = []
for sentence in parsed:
    corpus.extend(preprocess(sentence))
end = time.perf_counter()
mem_after = memory_usage()
stats['Preprocessing'] = (end - start, mem_after - mem_before)
print('done')

# --- Splitting ---
train_parsed, test_parsed = splitCorpus(corpus)
print('done')

# --- Flatten Training Corpus ---
mem_before = memory_usage()
start = time.perf_counter()
flatten_train = flattenCorpus(train_parsed)
end = time.perf_counter()
mem_after = memory_usage()
stats['Flatten Train'] = (end - start, mem_after - mem_before)
print('done')

# --- Word Count ---
mem_before = memory_usage()
start = time.perf_counter()
train_words_count = wordCounter(flatten_train)
end = time.perf_counter()
mem_after = memory_usage()
stats['Word Count'] = (end - start, mem_after - mem_before)
print('done')

# --- Replace Rare Words (UNK) ---
mem_before = memory_usage()
start = time.perf_counter()
unk_train_corpus = replaceRareWords(train_parsed, train_words_count)
end = time.perf_counter()
mem_after = memory_usage()
stats['UNK Replacement'] = (end - start, mem_after - mem_before)
print('done')

# --- Flatten UNK Corpus ---
flatten_unk_train = flattenCorpus(unk_train_corpus)
print('done')

# --- Build Manual N-Gram ---
mem_before = memory_usage()
start = time.perf_counter()
ngrams = buildNgrams(flatten_train)
end = time.perf_counter()
mem_after = memory_usage()
stats['Manual N-Gram'] = (end - start, mem_after - mem_before)
print('done')

# --- Build UNK N-Gram ---
mem_before = memory_usage()
start = time.perf_counter()
unk_ngrams = buildNgrams(flatten_unk_train)
end = time.perf_counter()
mem_after = memory_usage()
stats['UNK N-Gram'] = (end - start, mem_after - mem_before)
print('done')

# --- Build Library N-Gram ---
mem_before = memory_usage()
start = time.perf_counter()
ngramlib = buildLibraryNGrams(flatten_train)
end = time.perf_counter()
mem_after = memory_usage()
stats['Library N-Gram'] = (end - start, mem_after - mem_before)
print('done')

# --- Flatten Test + Replace UNK ---
flatten_test = flattenCorpus(test_parsed)
test_words_count = wordCounter(flatten_test)
unk_test_corpus = replaceRareWords(test_parsed, test_words_count)
flatten_unk_test = flattenCorpus(unk_test_corpus)
print('done')

# --- Save Pickled Files ---
saving_data.save('ngram_unk.pkl', unk_ngrams)
saving_data.save('ngram.pkl', ngrams)
saving_data.save('train.pkl', train_parsed)
saving_data.save('test.pkl', test_parsed)
saving_data.save('unk_train.pkl', unk_train_corpus)
saving_data.save('unk_test.pkl', unk_test_corpus)
print('done')

# --- Generate Sample Sentences from Laplace Model ---
print("\nLaplace Model Sample Generation:\n")
model = LaplaceLanguageModel(unk_ngrams)
for _ in range(5):  
    print("Generated:", model.generateSentence())

# --- Summary Report ---
print("\n--- TIME & MEMORY USAGE REPORT (in seconds and MB) ---")
for stage, (time_sec, mem_mb) in stats.items():
    print(f"{stage:25} | Time: {time_sec:.2f}s | Memory Change: {mem_mb:.2f}MB")

# --- N-Gram Comparison ---
manual_time = stats['Manual N-Gram'][0]
library_time = stats['Library N-Gram'][0]
speedup = library_time / manual_time if library_time else float('inf')

print(f"\nN-Gram Timing Comparison:")
print(f"Manual  N-Gram Time: {manual_time:.4f} seconds")
print(f"Library N-Gram Time: {library_time:.4f} seconds")
print(f"Speedup (Library / Manual): {speedup:.2f}x")

print(f"Total sentences after preprocessing: {len(corpus)}")
# --- Word and Vocabulary Statistics ---
word_count = sum(len(sentence) for sentence in corpus)
vocab = set(word for sentence in corpus for word in sentence)
vocab_size = len(vocab)

print(f"Total number of words: {word_count}")
print(f"Vocabulary size: {vocab_size}")