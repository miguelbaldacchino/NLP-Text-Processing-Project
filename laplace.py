from vanilla import VanillaLanguageModel

# As i understand it, 
# add 1 to every numerator, add vocab_szize to every denominator, when calculating probs
# and in unigram, n + v as denominator
# n - no. of tokens, v - no. of distinct words

class LaplaceLanguageModel(VanillaLanguageModel):
    def __init__(self, ngrams):
        # inherited raw counts, will override probs of vanilla
        super().__init__(ngrams) 
        
        self.N = sum(self.unigram_freqs.values())
        self.V = len(self.unigram_freqs)
        
        self.laplace_unigram_probs = self.computeUnigramProbsLaplace()
        self.laplace_bigram_probs = self.computeBigramProbsLaplace()
        self.laplace_trigram_probs = self.computeTrigramProbsLaplace()
        
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
    
    def wordChosen(self, probabilities_dict):
        # same as in VanillaLM
        return super().wordChosen(probabilities_dict)
    
    def generateSentence(self, max_length=15, min_length=10):
        sentence = ['<s>']
        attempts = 0
        while len(sentence) < max_length and attempts < 100:
            attempts += 1

            if len(sentence) == 1:
                dist = self.laplace_bigram_probs.get('<s>', {})
            else:
                context = (sentence[-2], sentence[-1])
                dist = self.laplace_trigram_probs.get(context, {})

            next_word = self.wordChosen(dist)

            if not next_word:
                break
            if next_word == '</s>' and len(sentence) < min_length:
                continue
            if next_word.endswith('-'):
                continue
            if next_word == '</s>':
                break
            sentence.append(next_word)

        return sentence[1:]

