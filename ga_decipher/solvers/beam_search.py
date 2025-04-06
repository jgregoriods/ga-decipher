import matplotlib.pyplot as plt
import numpy as np
import time

from typing import List
from ga_decipher.ngram_model import NgramModel
from ga_decipher.scoring import calculate_final_score
from ga_decipher.solvers.solver import Solver


class BeamSearch(Solver):
    """
    A class to represent a beam search algorithm for solving a substitution
    cipher.

    Attributes
    ----------
    source_symbols : List[str]
        The source symbols to be mapped from.
    target_symbols : List[str]
        The target symbols to be mapped to.
    source_text : List[List[str]]
        The source text to decode.
    model : NgramModel
        The n-gram model to use for scoring.
    nodes : int
        The number of nodes in the search tree.
    beam_width : int
        The width of the beam.
    starting_key : Dict[str, str]
        The starting decipherment key.
    best_key : Dict[str, str]
        The best decipherment key found.
    best_scores : List[float]
        The best scores found in each iteration.
    avg_scores : List[float]
        The average scores found in each iteration.
    
    Methods
    -------
    run(iterations: int) -> None
        Runs the beam search algorithm.
    plot(filename: str = '') -> None
        Plots the evolution of the best and average scores of the population.
    write_result(filename: str) -> None
        Writes the best key to a file.
    """

    def __init__(self, source_symbols: List[str], target_symbols: List[str],
                 source_text: List[List[str]], model: NgramModel, nodes: int,
                 beam_width: int) -> None:
        super().__init__(source_symbols, target_symbols, source_text, model)
        self.nodes = nodes
        self.beam_width = beam_width
        self.starting_key = {source: target for source, target in zip(source_symbols, target_symbols)}
        self.best_key = self.starting_key
        self.best_scores = []
        self.avg_scores = []

    def run(self, iterations: int, callback: callable = None) -> None:
        """
        Runs the beam search algorithm.

        Parameters
        ----------
        iterations : int
            The number of iterations to run the algorithm.
        """
        start = time.time()

        init_score = calculate_final_score(self.source_text, self.starting_key, self.model)
        beam = [(self.starting_key, init_score)]

        print('\nSearching for the best key ...\n')

        for i in range(iterations):
            candidates = [beam[0]]
            for cipher_key, _ in beam:
                for _ in range(self.nodes):
                    k, l = np.random.choice(list(cipher_key.keys()), 2)
                    new_key = cipher_key.copy()
                    new_key[k], new_key[l] = new_key[l], new_key[k]
                    new_score = calculate_final_score(self.source_text, new_key, self.model)
                    candidates.append((new_key, new_score))

            beam = sorted(candidates, key=lambda x: x[1], reverse=True)[:self.beam_width]
            self.best_key = beam[0][0]
            self.best_scores.append(beam[0][1])
            self.avg_scores.append(np.mean([score for _, score in beam]))

            if callback is not None:
                callback(i, self.best_key, self.best_scores)
            else:
                print(f'Generation {i + 1} done')
                print(f'Best: {self.best_scores[-1]}')
                top_symbols = {symbol: self.best_key[symbol] for symbol in self.source_symbols[:5]}
                print(f'Best key: {top_symbols} ...\n')

        end = time.time()
        print('Time elapsed: {:.2f}s'.format(end - start))
