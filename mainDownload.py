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


