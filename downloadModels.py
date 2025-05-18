# load_models.py
from pickles import savingData, retrieve
from vanilla import VanillaLanguageModel
from laplace import LaplaceLanguageModel
from unkModel import UNKLanguageModel
import time
import psutil

def memory_usage():
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024  # Return MB

def load_and_save_models():
    saving_data = savingData()
    stats = {}
    
    print("Loading pre-saved n-gram files...")
    start_mem = memory_usage()
    ngrams = retrieve('ngram')
    unk_ngrams = retrieve('ngram_unk')
    stats['Data Loading'] = (memory_usage() - start_mem, 0)
    
    print("\nBuilding Vanilla Model...")
    start_time = time.time()
    start_mem = memory_usage()
    vanilla_model = VanillaLanguageModel(ngrams, compute_probs=True)
    saving_data.save('vanilla_model.pkl', vanilla_model)
    stats['Vanilla'] = (time.time()-start_time, memory_usage()-start_mem)
    print(f"Vanilla model saved. Time: {stats['Vanilla'][0]:.2f}s | Memory: {stats['Vanilla'][1]:.2f}MB")
    
    print("\nBuilding Laplace Model...")
    start_time = time.time()
    start_mem = memory_usage()
    laplace_model = LaplaceLanguageModel(ngrams)
    saving_data.save('laplace_model.pkl', laplace_model)
    stats['Laplace'] = (time.time()-start_time, memory_usage()-start_mem)
    print(f"Laplace model saved. Time: {stats['Laplace'][0]:.2f}s | Memory: {stats['Laplace'][1]:.2f}MB")
    
    print("\nBuilding UNK Model...")
    start_time = time.time()
    start_mem = memory_usage()
    unk_model = UNKLanguageModel(unk_ngrams)
    saving_data.save('unk_model.pkl', unk_model)
    stats['UNK'] = (time.time()-start_time, memory_usage()-start_mem)
    print(f"UNK model saved. Time: {stats['UNK'][0]:.2f}s | Memory: {stats['UNK'][1]:.2f}MB")
    
    print("\n=== Resource Usage Summary ===")
    print(f"{'Model'} | {'Time (s)':<8} | {'RAM (MB)':<8}")
    print("-"*30)
    for model, (t, mem) in stats.items():
        print(f"{model} | {t:.2f}     | {mem:.2f}")
    print("Note: Data loading consumed", stats['Data Loading'][0], "MB")

def test_models():
    saving_data = savingData()
    
    print("\nTesting loaded models:")
    vanilla = retrieve('vanilla_model')
    laplace = retrieve('laplace_model')
    unk = retrieve('unk_model')
    
    test_input = "il-ġurnata hija"
    
    print("\nVanilla Model Generation:")
    print(vanilla.generateSentence(input_string=test_input))
    
    print("\nLaplace Model Generation:")
    print(laplace.generateSentence(input_string=test_input))
    
    print("\nUNK Model Generation:")
    print(unk.generateSentence(input_string=test_input))

if __name__ == "__main__":
    load_and_save_models()
    test_models()
