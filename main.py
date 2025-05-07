sentences = ['Hello my name, is Miguel,   and I like  Shania... .,. I   really do!']

def preprocess(text):
    text = lowercase(text)
    text = splitSentence(text)
    
    cleaned_sentences = []
    for sentence in text:
        sentence = removePunctuation(sentence)
        tokens = tokenize(sentence)
        # skips empty lines
        if tokens:
            tokens = ['<s>'] + tokens + ['</s>']
            cleaned_sentences.append(tokens)
        
    return cleaned_sentences
    

def lowercase(text):
    # lowercases every character
    return text.lower()
    
def splitSentence(text):
    # sentence ending punctuation
    sentence_endings = {'.', '!', '?'}
    # stores sentences found
    sentences = []
    # current sentence being worked on
    current_sentence = ''
    
    # for every character in parametered text
    for char in text:
        # if no sentence ending char, proceed
        if char not in sentence_endings:
            current_sentence += char
        # else end of sentence found
        else:
            # handles empty instances of current_sentence
            if current_sentence.strip():
                sentences.append(current_sentence)
                current_sentence = ''
    if current_sentence.strip():
        sentences.append(current_sentence)
    return sentences
        
def removePunctuation(text):
    # all punctuation characters
    punctuation_chars = {'!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', 
                     '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', 
                     '_', '`', '{', '|', '}', '~'}
    # current sentence being worked on
    sentence = ''
    
    for char in text:
        # filters out punct chars
        if char not in punctuation_chars:
            sentence += char
        
    return sentence
   
        
def tokenize(text):
    return text.split()

for sentence in sentences:
    print(preprocess(sentence))

