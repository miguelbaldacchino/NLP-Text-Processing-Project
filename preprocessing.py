def preprocess(sentence):
    cleaned_sentence = []
    sentence = [removePunctuation(word) for word in sentence if tokenize(word) != '' and removePunctuation(word) != '']
        # skips empty lines
    if sentence:
        sentence = ['<s>'] + sentence + ['</s>']
        cleaned_sentence.append(sentence)
        
    return cleaned_sentence

def lowercase(text):
    # lowercases every character
    return text.lower()
        
def removePunctuation(text):
    # all punctuation characters
    punctuation_chars = {'!', '"', '#', '$', '%', '&', '(', ')', '*', '+', ',', 
                     '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', 
                     '_', '`', '{', '|', '}', '~'}
    # current word being worked on
    word = ''
    
    for char in text:
        # filters out punct chars
        if char not in punctuation_chars:
            word += char
    return word  
        
def tokenize(text):
    return text.split()

# corpus must be parsed through vrt parser first
def flattenPreprocessCorpus(corpus):
    flatten_corpus = []
    for sentence in corpus:
        preprocessed = preprocess(sentence) 
        for word in preprocessed:
            flatten_corpus.extend(word)
    return flatten_corpus