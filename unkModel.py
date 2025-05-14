from laplace import LaplaceLanguageModel
from preprocessing import tokenize, lowercase
from preprocessing import flattenCorpus
from unk import replaceRareWords, wordCounter

class UNKLanguageModel(LaplaceLanguageModel):
    def __init__(self, ngrams):
        super().__init__(ngrams)
        self.vocab = set(self.unigram_freqs.keys())  # define known vocab
    
    def generateSentence(self, max_length=15, input_string=''):
        if isinstance(input_string, str):
            input_tokens = tokenize(input_string)
            input_tokens = [lowercase(token) for token in input_tokens]
        else:
            input_tokens = input_string

        # Replace unseen tokens with <UNK>
        input_tokens = [w if w in self.vocab else '<UNK>' for w in input_tokens]

        sentence = ['<s>'] + input_tokens
        attempts = 0
        while len(sentence) < max_length and attempts < 100:
            attempts += 1

            if len(sentence) < 2:
                context = sentence[-1]
                dict = self.bigram_probs.get(context, {})
            else:
                trigram_context = (sentence[-2], sentence[-1])
                dict = self.trigram_probs.get(trigram_context, {})
                if not dict:
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
    
    def Sen_Probability(self, sentence):
        if isinstance(sentence, str):
            sentence = tokenize(sentence)
            sentence = [lowercase(token) for token in sentence]

        sentence = [w if w in self.vocab else '<UNK>' for w in sentence]

        # Add start/end tokens if needed
        if sentence[0] != '<s>':
            sentence = ['<s>'] + sentence
        if sentence[-1] != '</s>':
            sentence = sentence + ['</s>']

        prob = 1.0

        for i in range(len(sentence)):
            w1 = sentence[i - 2] if i >= 2 else None
            w2 = sentence[i - 1] if i >= 1 else None
            w3 = sentence[i]

            # Try trigram first
            if w1 and w2:
                context = (w1, w2)
                prob_dict = self.trigram_probs.get(context, {})
                p = prob_dict.get(w3, None)
                if p is not None:
                    prob *= p
                    continue

            # Fallback to bigram
            if w2:
                prob_dict = self.bigram_probs.get(w2, {})
                p = prob_dict.get(w3, None)
                if p is not None:
                    prob *= p
                    continue

            # If both missing, fallback to very small value
            prob *= 1e-20

        return prob
