import numpy as np

from typing import List


class NgramModel:
    """
    A class to represent an n-gram model.

    Attributes
    ----------
    n : int
        The size of the n-grams.
    ngrams : dict
        A dictionary of n-grams and their counts.
    
    Methods
    -------
    score(sentence: List[str]) -> float
        Scores a sentence based on the n-gram model.
    fit(source_texts: List[List[str]]) -> None
        Fits the n-gram model to the source texts.
    """
    def __init__(self, n: int) -> None:
        self.n = n
        self.ngrams = dict()
        self.ngram_probs = dict()

    def score(self, sentence: List[str]) -> float:
        """
        Scores a sentence based on the n-gram model.

        Parameters
        ----------
        sentence : List[str]
            The sentence to score.
        """
        ngram_total = 0
        for i in range(len(sentence) - self.n + 1):
            ngram = (sentence[i], sentence[i+1])
            if ngram in self.ngrams:
                ngram_total += np.log2(self.ngrams[ngram])
        return ngram_total

    def log_prob(self, sentence: List[str]) -> float:
        """
        Returns the log probability of a sentence.

        Parameters
        ----------
        sentence : List[str]
            The sentence to calculate the log probability of.
        """
        log_prob = 0
        for i in range(len(sentence) - self.n + 1):
            ngram = tuple(sentence[i:i+self.n])
            if ngram[:-1] in self.ngram_probs and ngram[-1] in self.ngram_probs[ngram[:-1]]:
                log_prob += np.log(self.ngram_probs[ngram[:-1]][ngram[-1]])
            else:
                log_prob += np.log(1e-10)
        return log_prob

    def fit(self, source_texts: List[List[str]]) -> 'NgramModel':
        """
        Fits the n-gram model to the source texts.

        Parameters
        ----------
        source_texts : List[List[str]]
            The source texts to fit the n-gram model to.
        """
        for line in source_texts:
            for i in range(len(line) - self.n + 1):
                ngram = tuple(line[i:i+self.n])
                self.ngrams[ngram] = self.ngrams.get(ngram, 0) + 1
                if ngram[:-1] not in self.ngram_probs:
                    self.ngram_probs[ngram[:-1]] = {}
                self.ngram_probs[ngram[:-1]][ngram[-1]] = self.ngram_probs[ngram[:-1]].get(ngram[-1], 0) + 1
        for ngram, probs in self.ngram_probs.items():
            total = sum(probs.values())
            for key in probs:
                probs[key] /= total
        return self
