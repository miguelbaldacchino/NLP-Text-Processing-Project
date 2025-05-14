from laplace import LaplaceLanguageModel
from vanilla import VanillaLanguageModel
from pickles import retrieve

# flattens and preprocesses corpus

#flatten_train = retrieve('flatten_train')
#flatten_test = retrieve('flatten_test')
#flatten_unk_train = retrieve('flatten_unk_train')
#flatten_unk_test = retrieve('flatten_unk_test')

#ngram = retrieve('ngram')
test = retrieve('test')
ngram_unk = retrieve('ngram_unk')
ngram = retrieve('ngram')

laplace = LaplaceLanguageModel(ngram_unk)
    
vanilla = VanillaLanguageModel(ngram)
for _ in range(3):  
    sentence = vanilla.generateSentence()
    print("Vanilla Generated:", sentence)
    prob, _, _, _, _ = vanilla.linearInterpolation(sentence)
    print('probability: ', prob)
for _ in range(3):  
    sentence = vanilla.generateSentence(input_string='Shania Saliba nhobbha')
    print("Vanilla Generated:", sentence)
    prob, _, _, _, _ = vanilla.linearInterpolation(sentence)
    print('probability: ', prob)

print(vanilla.perplexity(test))    