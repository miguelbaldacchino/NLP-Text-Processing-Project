from laplace import LaplaceLanguageModel
from vanilla import VanillaLanguageModel
from pickles import retrieve

# flattens and preprocesses corpus

#flatten_train = retrieve('flatten_train')
#flatten_test = retrieve('flatten_test')
#flatten_unk_train = retrieve('flatten_unk_train')
#flatten_unk_test = retrieve('flatten_unk_test')

#ngram = retrieve('ngram')
ngram_unk = retrieve('ngram_unk')
ngram = retrieve('ngram')

laplace = LaplaceLanguageModel(ngram_unk)
for _ in range(3):  
    sentence = laplace.generateSentence(input_string='fjura fid- dinja')
    print("Laplace Generated:", sentence)
    
vanilla = VanillaLanguageModel(ngram)
for _ in range(3):  
    sentence = vanilla.generateSentence()
    full_sentence = ''
    for word in sentence:
        full_sentence += word + ' '
    print("Vanilla Generated:", full_sentence)
    prob, _, _, _ = vanilla.linearInterpolation(full_sentence)
    print('probability: ', prob)
    