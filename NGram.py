import nltk
from nltk.corpus import brown
from nltk import word_tokenize
from nltk.util import ngrams
from collections import Counter
import math

nltk.download('brown')
nltk.download('punkt_tab')

class BigramLanguageModel:
    def __init__(self, corpus_name='brown', category='government', num_sentences=10000):
        self.corpus_name = corpus_name
        self.category = category
        self.num_sentences = num_sentences
        self.unigrams = Counter()
        self.bigrams = Counter()
        self.V = 0  # Vocabulary size
        self.epsilon = 1e-10  # avoid log(0)
        
        # Load and preprocess data
        self._load_and_preprocess()
        
    def _load_and_preprocess(self):
       
        print(f"Loading {self.num_sentences} sentences from {self.corpus_name} corpus ({self.category} category)...")
        sentences = brown.sents(categories=self.category)[:self.num_sentences]
        
        # Add tokens <s> và </s> for marking the begin/end of a sentence
        processed_sentences = []
        for sent in sentences:
            # Add start and end tokens, and lowercase
            processed_sent = ['<s>'] + [w.lower() for w in sent] + ['</s>']
            processed_sentences.append(processed_sent)
        
        # Calculate unigram and bigram counts
        for sent in processed_sentences:
            self.unigrams.update(sent)
            self.bigrams.update(ngrams(sent, 2))
        
        self.V = len(self.unigrams)
        print(f"Vocabulary size: {self.V}")
        print(f"Total unigrams: {sum(self.unigrams.values())}")
        print(f"Total bigrams: {sum(self.bigrams.values())}")
    
    def mle_probability(self, w1, w2):
        """Maximum Likelihood Estimation probability"""
        return self.bigrams[(w1, w2)] / self.unigrams[w1] if self.unigrams[w1] > 0 else 0.0
    
    def laplace_probability(self, w1, w2):
        """Laplace (add-one) smoothing probability"""
        return (self.bigrams[(w1, w2)] + 1) / (self.unigrams[w1] + self.V)
    
    def perplexity(self, test_sent, smoothing='laplace'):
       #Choose smoothing function
        if smoothing == 'mle':
            prob_func = self.mle_probability
        elif smoothing == 'laplace':
            prob_func = self.laplace_probability
        else:
            raise ValueError(f"Unknown smoothing method: {smoothing}")
        
        # Add start/end tokens and tokenize
        test_tokens = ['<s>'] + word_tokenize(test_sent.lower()) + ['</s>']
        test_bigrams = list(ngrams(test_tokens, 2))
        
        if not test_bigrams:
            return 0.0
            
        log_prob_sum = 0.0
        zero_probs = 0
        
        for w1, w2 in test_bigrams:
            p = prob_func(w1, w2)
            if p > 0:
                log_prob_sum += math.log2(p)
            else:
                zero_probs += 1
                log_prob_sum += math.log2(self.epsilon)  # Avoid -inf
        
        # if there are many zero probabilities, perplexity will be large
        if zero_probs > 0:
            print(f"Warning: {zero_probs}/{len(test_bigrams)} bigrams had zero probability")
        
        return pow(2, -log_prob_sum / len(test_bigrams))

if __name__ == "__main__":
    model = BigramLanguageModel(category='news', num_sentences=10000)
    
    # Test sentences
    test_sentences = [
        "the government is",
        "the president said",
        "Does our society have a runaway?",
        "unknown words should be handled"
    ]
    
    # Evaluate Perplexity
    for sent in test_sentences:
        print(f"\nTest sentence: '{sent}'")
        print(f"Perplexity (MLE): {model.perplexity(sent, 'mle'):.2f}")
        print(f"Perplexity (Laplace): {model.perplexity(sent, 'laplace'):.2f}")
