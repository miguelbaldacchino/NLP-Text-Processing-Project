import random
from collections import defaultdict
from preprocessing import tokenize, lowercase

class VanillaLanguageModel:
    def __init__(self, ngrams, compute_probs=True):
        self.unigram_freqs = ngrams['Unigrams']
        self.bigram_freqs = ngrams['Bigrams']
        self.trigram_freqs = ngrams['Trigrams']
        
        if compute_probs:
            self.total_count = sum(self.unigram_freqs.values())
            self.unigram_probs = self.computeUnigramProbs()
            self.bigram_probs = self.computeBigramProbs()
            self.trigram_probs = self.computeTrigramProbs()
        
    def computeUnigramProbs(self):
        return {word: count / self.total_count for word, count in self.unigram_freqs.items()}
    
    def computeBigramProbs(self):
        probs = {}
        for (w1, w2), count in self.bigram_freqs.items():
            if w1 not in probs:
                probs[w1] = {}
            w1_count = self.unigram_freqs.get(w1, 0)  
            probs[w1][w2] = count / w1_count
        return probs
    
    def computeTrigramProbs(self):
        probs = {}
        for (w1, w2, w3), count in self.trigram_freqs.items():
            context = (w1, w2)
            if context not in probs:
                probs[context] = {}
            bigram_count = self.bigram_freqs.get((w1, w2), 0)  
            probs[context][w3] = count / bigram_count
        return probs
    
    def wordChosen(self, probabilities_dict):
        words = list(probabilities_dict.keys())
        probabilities = list(probabilities_dict.values())
        
        if not words:
            return '</s>'
        return random.choices(words, weights=probabilities, k=1)[0]
    
    def generateSentence(self, max_length=15, min_length=10, input_string=''):
        if isinstance(input_string, str):
            input_tokens = tokenize(input_string)
            sentence = []
            for token in input_tokens:
                token = lowercase(token)
                sentence.append(token)
            input_tokens = sentence
        else:
            input_tokens = input_string
        
        sentence = ['<s>'] + input_tokens
        attempts = 0
        while len(sentence) < max_length and attempts < 100:  # loop limit
            attempts += 1

            if len(sentence) < 2:
                context = sentence[-1]
                dict = self.bigram_probs.get(context, {})
            else:
                trigram_context = (sentence[-2], sentence[-1])
                dict = self.trigram_probs.get(trigram_context, {})
                if not dict:
                    # Fallback to bigram using the last word
                    dict = self.bigram_probs.get(sentence[-1], {})
            
            next_word = self.wordChosen(dict)
            
            if next_word.endswith('-'):
                continue
            if not next_word:
                break
            if next_word == '</s>':
                break
            sentence.append(next_word)

        return sentence[1:]

    
    def linearInterpolation(self, sentence, l1=0.1, l2=0.3, l3=0.6):
        
        total_prob = 1.0
        uni_p = []
        bi_p = []
        tri_p = []
        
        for i in range(len(sentence)):
            w1 = sentence[i-2] if i >= 2 else None
            w2 = sentence[i-1] if i >= 1 else None
            w3 = sentence[i]
            
            uni = self.unigram_probs.get(w3, 0.00000001)
            bi = self.bigram_probs.get(w2, {}).get(w3, 0.00000001) if w2 else 0
            tri = self.trigram_probs.get((w1, w2), {}).get(w3, 0.00000001) if w1 and w2 else 0

            uni_p.append(uni)
            bi_p.append(bi)
            tri_p.append(tri)
            
            total_prob *= (l1 * uni) + (l2 * bi) + (l3 * tri)
            
        return total_prob, uni_p, bi_p, tri_p
    
    def probabilityTable(self, corpus): 
        prob_unigram = 0.0
        prob_bigram = 0.0
        prob_trigram = 0.0
        prob_interpolation = 0.0
        for sentence in corpus:
            prob, uni, bi, tri = self.linearInterpolation(sentence)
            for probab in uni:
                prob_unigram += probab
            for probab in bi:
                prob_bigram += probab
            for probab in tri:
                prob_trigram += probab
            prob_interpolation += prob
            
            

                    
        