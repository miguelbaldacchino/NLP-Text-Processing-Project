from preprocessing import lowercase, tokenize
import os
import random

# takes file by file
def vrtParser(filepath):
    # list storing all sentences
    sentences = []
    
    with open(filepath, 'r', encoding='utf-8') as file:
        # currently in sentence flag
        in_sentence = False
        # list storing current sentence
        current_sentence = []

        
        # iterating every line in file
        for line in file:            
            # start of sentence
            if line.startswith("<s"):
                # removes spaces
                line = tokenize(line)
                in_sentence = True
                current_sentence = []
            # end of sentence
            elif line.startswith("</s>"):
                if current_sentence:
                    sentences.append(current_sentence)
                in_sentence = False
            # sentence contents
            elif in_sentence and not line.startswith("<"):
                word = line.split('\t')[0]  # Extract first column
                if word:  # Avoid empty strings
                    current_sentence.append(lowercase(word)) 
    return sentences

def loadCorpus(folder_path, max_files=None):
    parsed = []
    for i, file_name in enumerate(os.listdir(folder_path)):
        if max_files and i >= max_files:
            break
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            parsed.extend(vrtParser(file_path))
            print(f'Parsed {i}: {file_name}')
    return parsed

def splitCorpus(corpus, train_size = 0.8):
    random.shuffle(corpus)
    
    train_len = int(len(corpus) * train_size)
    
    train_sentences = corpus[:train_len]
    test_sentences = corpus[train_len:]
    
    return train_sentences, test_sentences
