# interpolation_benchmark.py
import time
import psutil
from preprocessing import tokenize, lowercase
from pickles import retrieve

def memory_usage():
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024  # MB

def benchmark_interpolation(model, sentence, model_name):
    if isinstance(sentence, str):
        tokens = tokenize(sentence)
        tokens = [lowercase(token) for token in tokens]
    else:
        tokens = sentence
    
    mem_before = memory_usage()
    
    start_time = time.time()
    
    prob, _, _, _, _ = model.linearInterpolation(tokens)
    
    elapsed_time = time.time() - start_time
    mem_used = memory_usage() - mem_before
    
    print(f"\n{model_name} Model Interpolation:")
    print(f"  Sentence: '{' '.join(tokens)}'")
    print(f"  Probability: {prob}")
    print(f"  Time: {elapsed_time:.6f} seconds")
    print(f"  Memory: {mem_used:.2f} MB")
    
    return {
        "model": model_name,
        "probability": prob,
        "time": elapsed_time,
        "memory": mem_used
    }

def main():
    print("Loading models...")
    vanilla_model = retrieve('vanilla_model')
    laplace_model = retrieve('laplace_model')
    unk_model = retrieve('unk_model')
    
    test_sentence = "il-ġurnata hija ferm sabiħa, minn tlugħ ix-xemx sa nżulha."
    print(f"Benchmarking interpolation for sentence: '{test_sentence}'")
    
    results = []
    results.append(benchmark_interpolation(vanilla_model, test_sentence, "Vanilla"))
    results.append(benchmark_interpolation(laplace_model, test_sentence, "Laplace"))
    results.append(benchmark_interpolation(unk_model, test_sentence, "UNK"))
    
if __name__ == "__main__":
    main()
