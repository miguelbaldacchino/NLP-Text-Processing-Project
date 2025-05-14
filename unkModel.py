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
