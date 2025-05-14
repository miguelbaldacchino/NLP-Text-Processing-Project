from vanilla import VanillaLanguageModel
from preprocessing import tokenize, lowercase

# As i understand it, 
# add 1 to every numerator, add vocab_szize to every denominator, when calculating probs
# and in unigram, n + v as denominator
# n - no. of tokens, v - no. of distinct words

class LaplaceLanguageModel(VanillaLanguageModel):
    def __init__(self, ngrams):
        # inherited raw counts, will override probs of vanilla
        super().__init__(ngrams, compute_probs=False) 
        
        self.N = sum(self.unigram_freqs.values())
        self.V = len(self.unigram_freqs)
        
        self.unigram_probs = self.computeUnigramProbsLaplace()
        self.bigram_probs = self.computeBigramProbsLaplace()
        self.trigram_probs = self.computeTrigramProbsLaplace()
        
    def computeUnigramProbsLaplace(self):
        denom = self.N + self.V
        probabilities = {}
        for w, count in self.unigram_freqs.items():
            probabilities[w] = (count + 1) / denom
        return probabilities
    
    def computeBigramProbsLaplace(self):
        probs = {}
        for (w1, w2), count in self.bigram_freqs.items():
            if w1 not in probs:
                probs[w1] = {}
            denom = self.unigram_freqs.get(w1, 0) + self.V
            probs[w1][w2] = (count + 1) / denom
        return probs

    def computeTrigramProbsLaplace(self):
        probs = {}
        for (w1, w2, w3), count in self.trigram_freqs.items():
            context = (w1, w2)
            if context not in probs:
                probs[context] = {}
            denom = self.bigram_freqs.get(context, 0) + self.V 
            probs[context][w3] = (count + 1) / denom
        return probs
    