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
            ngram = ' '.join(sentence[i:i+self.n])
            if ngram in self.ngrams:
                ngram_total += np.log2(self.ngrams[ngram])
        return ngram_total

    def fit(self, source_texts: List[List[str]]) -> None:
        """
        Fits the n-gram model to the source texts.

        Parameters
        ----------
        source_texts : List[List[str]]
            The source texts to fit the n-gram model to.
        """
        for line in source_texts:
            for i in range(len(line) - self.n + 1):
                ngram = ' '.join(line[i:i+self.n])
                self.ngrams[ngram] = self.ngrams.get(ngram, 0) + 1

