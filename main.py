from laplace import LaplaceLanguageModel
from vanilla import VanillaLanguageModel
from unkModel import UNKLanguageModel
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
    
vanilla = VanillaLanguageModel(ngram)
for _ in range(3):  
    sentence = vanilla.generateSentence()
    print("Vanilla Generated:", sentence)
    prob, _, _, _, _ = vanilla.linearInterpolation(sentence)
    print('probability: ', prob)
for _ in range(3):  
    sentence = vanilla.generateSentence(input_string='Shania Saliba nħobbha')
    print("Vanilla Generated:", sentence)
    prob, _, _, _, _ = vanilla.linearInterpolation(sentence)
    print('interpolation: ', prob)
    print('sentence probability:', vanilla.Sen_Probability(sentence) )

print(vanilla.perplexity(test))    

laplace = LaplaceLanguageModel(ngram)
for _ in range(3):  
    sentence = laplace.generateSentence()
    print("Laplace Generated:", sentence)
    prob, _, _, _, _ = laplace.linearInterpolation(sentence)
    print('probability: ', prob)
for _ in range(3):  
    sentence = laplace.generateSentence(input_string='Shania Saliba nħobbha')
    print("Laplace Generated:", sentence)
    prob, _, _, _, _ = laplace.linearInterpolation(sentence)
    print('probability: ', prob)
    print('sentence probability:', laplace.Sen_Probability(sentence) )
print(laplace.perplexity(test))    

unkmodel = UNKLanguageModel(ngram_unk)
for _ in range(3):  
    sentence = unkmodel.generateSentence()
    print("UNK Generated:", sentence)
    prob, _, _, _, _ = unkmodel.linearInterpolation(sentence)
    print('probability: ', prob)
for _ in range(3):  
    sentence = unkmodel.generateSentence(input_string='Shania Saliba nħobbha')
    print("UNK Generated:", sentence)
    prob, _, _, _, _ = unkmodel.linearInterpolation(sentence)
    print('probability: ', prob)
    print('sentence probability:', unkmodel.Sen_Probability(sentence) )

print(unkmodel.perplexity(test))   
