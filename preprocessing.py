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
