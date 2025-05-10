from preprocessing import lowercase, tokenize

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
