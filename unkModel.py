from laplace import LaplaceLanguageModel
from preprocessing import tokenize, lowercase, preprocess, removePunctuation


# Note: At this stage, removing unseen words satisfies the count <= 2 rule, since the corpus has already had his <UNK> replacing, meaning whatever words arent in vocab also satisfy this rule.
class UNKLanguageModel(LaplaceLanguageModel):
    def __init__(self, ngrams):
        super().__init__(ngrams)
        self.vocab = set(self.unigram_freqs.keys())  # define known vocab
    
    def generateSentence(self, max_length=15, input_string=''):
        if isinstance(input_string, str):
            input_tokens = tokenize(input_string)
            sentence = []
            for token in input_tokens:
                token = lowercase(token)
                token = removePunctuation(token)
                if token:  
                    sentence.append(token)
            input_tokens = sentence
        else:
            input_tokens = [removePunctuation(lowercase(token)) for token in input_string if removePunctuation(lowercase(token)) != '']


        # replace unseen tokens with <UNK>
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

        # add start/end tokens if needed
        if sentence[0] != '<s>':
            sentence = ['<s>'] + sentence
        if sentence[-1] != '</s>':
            sentence = sentence + ['</s>']

        prob = 1.0

        for i in range(len(sentence)):
            w1 = sentence[i - 2] if i >= 2 else None
            w2 = sentence[i - 1] if i >= 1 else None
            w3 = sentence[i]

            # try trigram first
            if w1 and w2:
                context = (w1, w2)
                prob_dict = self.trigram_probs.get(context, {})
                p = prob_dict.get(w3, None)
                if p is not None:
                    prob *= p
                    continue

            # fallback to bigram
            if w2:
                prob_dict = self.bigram_probs.get(w2, {})
                p = prob_dict.get(w3, None)
                if p is not None:
                    prob *= p
                    continue

            # if both missing, fallback to very small value
            prob *= 1e-20

        return prob
    
    def linearInterpolation(self, sentence, l1=0.1, l2=0.3, l3=0.6, preprocessed=False):
        if not sentence:
            return 1e-20, [], [], [], 0
        if not preprocessed:
            # lowercase and tokenize first
            if isinstance(sentence, str):
                sentence = tokenize(lowercase(sentence))
            else:
                # assume already list of tokens; lowercase each
                sentence = [lowercase(w) for w in sentence]
            cleaned = preprocess(sentence)
            if not cleaned:
                return 1e-20, [], [], [], 0
            # take the first (and only) cleaned sentence for scoring
            sentence = cleaned[0]
        sentence = [w if w in self.vocab else '<UNK>' for w in sentence]
        total_prob = 1.0
        uni_p = []
        bi_p = []
        tri_p = []
        addedS = 0
        if sentence[0] != '<s>':
            sentence = ['<s>'] + sentence
            addedS += 1
        if sentence[-1] != '</s>':
            sentence = sentence + ['</s>']
            addedS += 1
        for i in range(len(sentence)):
            w1 = sentence[i-2] if i >= 2 else None
            w2 = sentence[i-1] if i >= 1 else None
            w3 = sentence[i]
            
            uni = self.unigram_probs.get(w3, 1e-20)
            bi = self.bigram_probs.get(w2, {}).get(w3, 1e-20) if w2 else 0
            tri = self.trigram_probs.get((w1, w2), {}).get(w3, 1e-20) if w1 and w2 else 0

            uni_p.append(uni)
            bi_p.append(bi)
            tri_p.append(tri)
            
            total_prob *= (l1 * uni) + (l2 * bi) + (l3 * tri)
        
        return total_prob, uni_p, bi_p, tri_p, addedS