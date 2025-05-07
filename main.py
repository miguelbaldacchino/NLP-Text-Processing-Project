from preprocessing import preprocess

sentences = ['Hello my name, is Miguel,   and I like  Shania... .,. I   really do!']

for sentence in sentences:
    print(preprocess(sentence))

