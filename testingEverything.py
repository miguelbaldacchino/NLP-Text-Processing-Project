import time
import psutil
from pickles import retrieve
from preprocessing import tokenize, lowercase, preprocess

def memory_usage():
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024  # MB

def print_models(models):
    print("Available models:")
    for idx, model in enumerate(models.keys(), 1):
        print(f"  {idx}. {model}")

def select_model(models):
    while True:
        print_models(models)
        choice = input("Select a model by number: ")
        try:
            idx = int(choice) - 1
            model_name = list(models.keys())[idx]
            return model_name, models[model_name]
        except (ValueError, IndexError):
            print("Invalid selection. Please try again.")

def print_operations():
    print("\nOperations:")
    print("  1. Generate Sentence")
    print("  2. Sentence Probability")
    print("  3. Linear Interpolation")
    print("  4. Perplexity (on test set)")
    print("  5. Exit")

def select_operation():
    while True:
        print_operations()
        op = input("Select an operation by number: ")
        if op in {'1','2','3','4','5'}:
            return int(op)
        print("Invalid option, try again.")

def input_sentence():
    return input("Enter your sentence: ").strip()

def preprocess_sentence(sentence):
    tokens = tokenize(lowercase(sentence))
    cleaned = preprocess(tokens)
    # cleaned is a list of lists, so take the first one
    if not cleaned:
        return []
    return cleaned[0]


def run_interactive():
    print("Loading models...")
    models = {
        'Vanilla': retrieve('vanilla_model'),
        'Laplace': retrieve('laplace_model'),
        'UNK': retrieve('unk_model')
    }
    print("Models loaded.\n")

    # for perplexity we need test corpora, already preprocessed
    test_corpus = retrieve('test')
    unk_test_corpus = retrieve('unk_test')

    while True:
        print()
        model_name, model = select_model(models)
        op = select_operation()

        if op == 5:
            print("Exiting.")
            break

        # perplexity doesnt require a sentence
        if op == 4:
            print("Calculating perplexity on the test set...")
            if model_name == 'UNK':
                test = unk_test_corpus
            else:
                test = test_corpus
            start = time.time()
            mem_before = memory_usage()
            scores = model.perplexity(test)
            elapsed = time.time() - start
            mem_used = memory_usage() - mem_before
            print(f"\nPerplexity for {model_name} model:")
            for n, v in scores.items():
                print(f"  {n}: {v}")
            print(f"  Time: {elapsed:.4f}s")
            print(f"  RAM used: {mem_used:.2f} MB")
            continue

        # sm ops require a sentence
        sentence = input_sentence()
        tokens = preprocess_sentence(sentence)

        if op == 1:
            # Sentence Generation
            print("\nGenerated sentence:")
            try:
                print(model.generateSentence(input_string=sentence))
            except Exception as e:
                print(f"Error: {e}")

        elif op == 2:
            # Sentence Probability
            try:
                prob = model.Sen_Probability(tokens)
                print(f"\nSentence Probability: {prob}")
            except Exception as e:
                print(f"Error: {e}")

        elif op == 3:
            # Linear Interpolation
            try:
                prob, _, _, _, _ = model.linearInterpolation(tokens, preprocessed=True)
                print(f"\nLinear Interpolation Probability: {prob}")
            except Exception as e:
                print(f"Error: {e}")

        print("\n---\n")

if __name__ == "__main__":
    run_interactive()
