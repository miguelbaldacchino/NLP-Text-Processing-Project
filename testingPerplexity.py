# perplexity_calculator.py
import time
import psutil
from vanilla import VanillaLanguageModel
from laplace import LaplaceLanguageModel
from unkModel import UNKLanguageModel
from pickles import retrieve

def memory_usage():
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024  # MB

def calculate_perplexity(models, test_corpus):
    results = {}
    
    for model_name, model in models.items():
        start_time = time.time()
        mem_before = memory_usage()
        
        # Calculate perplexity using model's built-in method
        perplexity_scores = model.perplexity(test_corpus)
        
        # Store results
        results[model_name] = {
            'Unigram': perplexity_scores['Unigram'],
            'Bigram': perplexity_scores['Bigram'],
            'Trigram': perplexity_scores['Trigram'],
            'Linear Interpolation': perplexity_scores['Linear Interpolation'],
            'Time': time.time() - start_time,
            'Memory': memory_usage() - mem_before
        }
    
    return results

def main():
    # Load test data
    try:
        test_corpus = retrieve('test')  # Original test corpus
        unk_test_corpus = retrieve('unk_test')  # UNK-processed test corpus
    except Exception as e:
        print(f"Error loading test data: {e}")
        return

    # Load models using retrieve()
    models = {
        'Vanilla': retrieve('vanilla_model'),
        'Laplace': retrieve('laplace_model'),
        'UNK': retrieve('unk_model')
    }

    # Calculate perplexities
    vanilla_laplace_results = calculate_perplexity(
        {'Vanilla': models['Vanilla'], 'Laplace': models['Laplace']}, 
        test_corpus
    )
    
    unk_results = calculate_perplexity(
        {'UNK': models['UNK']},
        unk_test_corpus
    )

    # Combine results
    all_results = {**vanilla_laplace_results, **unk_results}

    # Print results in table format
    print("\nPerplexity Results:")
    print(f"{'Model'} | {'Unigram'} | {'Bigram'} | {'Trigram'} | {'Interpolation'} | {'Time (s)':<8} | {'RAM (MB)':<8}")
    print("-" * 95)
    
    for model, scores in all_results.items():
        print(f"{model} | "
              f"{scores['Unigram']} | "
              f"{scores['Bigram']} | "
              f"{scores['Trigram']} | "
              f"{scores['Linear Interpolation']} | "
              f"{scores['Time']:<8.2f} | "
              f"{scores['Memory']:<8.2f}")


if __name__ == "__main__":
    main()
