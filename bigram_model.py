import numpy as np

from processing import split_syllables


class BigramModel:
    def __init__(self):
        self.bigrams = {}
        # syllabic or phonemic should be a parameter
        # in init

    # sentence must be already split
    def score(self, sentence):
        bigram_total = 0
        for i in range(len(sentence) - 2):
            bigram = ' '.join(sentence[i:i+3])
            if bigram in self.bigrams:
                bigram_total += np.log2(self.bigrams[bigram])
        return bigram_total

    # let's pass the texts as a 2D array already
    # i.e. syllables must be already split
    def fit(self, source_texts):
        for line in source_texts:
            #syllables = split_syllables(line)
            for i in range(len(line) - 2):
                bigram = ' '.join(line[i:i+3])
                self.bigrams[bigram] = self.bigrams.get(bigram, 0) + 1

