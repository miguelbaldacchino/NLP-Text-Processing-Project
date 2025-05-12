import random
from collections import defaultdict

class VanillaLanguageModel:
    def __init__(self, ngrams):
        self.unigram_freqs = ngrams['Unigrams']
        self.bigram_freqs = ngrams['Bigrams']
        self.trigram_freqs = ngrams['Trigrams']
        
        self.total_unigrams = sum(self.unigram_freqs.values())
        self.bigram_probs = self.computeBigramProbs()
        self.trigram_probs = self.computeTrigramProbs()
        
    def computeBigramProbs(self):
        probs = {}
        for (w1, w2), count in self.bigram_freqs.items():
            if w1 not in probs:
                probs[w1] = {}
            w1_count = self.unigram_freqs.get(w1, 1)  # Avoid division by 0
            probs[w1][w2] = count / w1_count
        return probs
    
    def computeTrigramProbs(self):
        probs = {}
        for (w1, w2, w3), count in self.trigram_freqs.items():
            context = (w1, w2)
            if context not in probs:
                probs[context] = {}
            bigram_count = self.bigram_freqs.get((w1, w2), 1)  # Avoid division by 0
            probs[context][w3] = count / bigram_count
        return probs
    
    def wordChosen(self, probabilities_dict):
        words = list(probabilities_dict.keys())
        probabilities = list(probabilities_dict.values())
        
        if not words:
            return '</s>'
        return random.choices(words, weights=probabilities, k=1)[0]
    
    def generateSentence(self, max_length=15, min_length=10):
        sentence = ['<s>']
        attempts = 0
        while len(sentence) < max_length and attempts < 100:  # loop limit
            attempts += 1

            if len(sentence) == 1:
                next_word = self.wordChosen(self.bigram_probs.get('<s>', {}))
            else:
                context = (sentence[-2], sentence[-1])
                next_word = self.wordChosen(self.trigram_probs.get(context, {}))
            
            if not next_word:
                break
            if next_word == '</s>' and len(sentence) < min_length or next_word.endswith('-'):
                continue
            if next_word == '</s>':
                break
            sentence.append(next_word)

        return sentence[1:]

    
    def sentenceProbability(self, sentence):
        probability = 1.0
        for i in range(2, len(sentence)):
            context = (sentence[i-2], sentence[i-1])
            word = sentence[i]
            probability *= self.trigram_probs.get(context, {}).get(word, 0.0000001)
        return probability
