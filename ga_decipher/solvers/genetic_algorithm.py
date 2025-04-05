import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import time

from typing import List, Dict
from ga_decipher.ngram_model import NgramModel
from ga_decipher.scoring import calculate_final_score


class Genome:
    """
    A class to represent a genome in the genetic algorithm.

    Attributes
    ----------
    target_symbols : List[str]
        The target symbols to be mapped to.
    genes : List[str]
        The genes of the genome (same elements as in target symbols).
        If not provided, the genes are target symbols randomly shuffled.
    score : float
        The score of the genome. We use -1 to indicate that the genome has not
        been evaluated yet.
    freeze : Dict[str, str]
        A dictionary of symbols that are fixed in the mapping.

    Methods
    -------
    mutate()
        Mutates the genome by randomly swapping two genes.
    """
    def __init__(self, target_symbols: List[str], genes: List[str] = list(),
                 score: float = -1, freeze: Dict[str, str] = dict()) -> None:
        random_symbols = target_symbols[:]
        if freeze:
            random_symbols = [i for i in random_symbols if i not in freeze.values()]
        np.random.shuffle(random_symbols)
        self.genes = genes or random_symbols
        self.score = score

    def mutate(self) -> None:
        """
        Mutates the genome by randomly swapping two genes.
        """
        j, k = np.random.choice(range(len(self.genes) - 1), size=2, replace=False)
        self.genes[j], self.genes[k] = self.genes[k], self.genes[j]
        self.score = -1


class GeneticAlgorithm:
    """
    A class to represent a genetic algorithm for solving the substitution
    cipher problem.

    Attributes
    ----------
    source_symbols : List[str]
        The source symbols to be mapped from.
    target_symbols : List[str]
        The target symbols to be mapped to.
    source_text : List[List[str]]
        The source text to be decoded.
    model : NgramModel
        The n-gram model to be used for scoring.
    population_size : int
        The size of the population.
    num_parents : int
        The number of parents to be selected for reproduction.
    prob_mut : float
        The probability of mutation.
    prob_cross : float
        The probability of crossover.
    freeze : Dict[str, str]
        A dictionary of symbols that are fixed in the mapping.
    n_cores : int
        The number of cores to be used in parallel processing.
        if 0, the number of cores is set to the number of available cores.

    Methods
    -------
    fitness_function(cipher_key: Dict[str, str])
        Calculates the fitness of a genome.
    ox1(parent1: List[str], parent2: List[str]) -> List[str]
        Order crossover.
    run(generations: int)
        Evolves the population for a number of generations.
    plot(filename: str = "")
        Plots the best and average scores of the population.
    write_result(filename: str)
        Writes the best key to a file.
    """
    def __init__(self, source_symbols: List[str], target_symbols: List[str],
                 source_text: List[List[str]], model: NgramModel,
                 population_size: int, num_parents: int, prob_mut: float,
                 prob_cross: float, freeze: Dict[str, str] = {},
                 n_cores: int = 0) -> None:
        self.source_symbols = source_symbols
        self.target_symbols = target_symbols
        self.source_text = source_text
        self.model = model
        self.population_size = population_size
        self.num_parents = num_parents
        self.num_children = int(np.ceil(population_size / num_parents * 2))
        self.prob_cross = prob_cross
        self.prob_mut = prob_mut
        self.freeze = freeze
        self.n_cores = n_cores if n_cores > 0 else mp.cpu_count()

        if freeze:
            self.source_symbols = [symbol for symbol in self.source_symbols if symbol not in freeze]

        print('Initializing population...')
        start = time.time()
        self.genomes = [Genome(target_symbols, freeze=self.freeze) for _ in range(self.population_size)]
        pool = mp.Pool(self.n_cores)
        population_genes = [{self.source_symbols[i]: genome.genes[i]
                             for i in range(len(self.source_symbols))}
                             for genome in self.genomes]

        for gene in population_genes:
            gene.update(self.freeze)

        fitness = np.array(pool.map(self.fitness_function, population_genes))

        pool.close()
        pool.join()

        for i, genome in enumerate(self.genomes):
            genome.score = fitness[i]

        end = time.time()
        print('Time elapsed: {:.2f}s'.format(end - start))
        print('Done')

        self.genomes.sort(key=lambda x: x.score, reverse=True)
        self.best_scores = [self.genomes[0].score]
        self.avg_scores = [np.mean([genome.score for genome in self.genomes])]
        self.best_key = {self.source_symbols[i]: self.genomes[0].genes[i] for i in range(len(self.source_symbols))}
        self.best_key.update(self.freeze)

    def fitness_function(self, cipher_key: Dict[str, str]) -> float:
        """
        Calculates the fitness of a genome.

        Parameters
        ----------
        cipher_key : Dict[str, str]
            The cipher key to be evaluated.

        Returns
        -------
        float
            The fitness of the genome.
        """
        return calculate_final_score(self.source_text, cipher_key, self.model)

    def ox1(self, parent1: List[str], parent2: List[str]) -> List[str]:
        """
        Order crossover. Randomly selects a segment from parent1 and fills
        the missing genes from parent2.
        
        Parameters
        ----------
        parent1 : List[str]
            The genes of the first parent genome.
        parent2 : List[str]
            The genes of the second parent genome.
        
        Returns
        -------
        List[str]
            The genes of the child genome after crossover.
        """
        i = np.random.randint(0, len(parent1) - 1)
        j = np.random.randint(i + 1, len(parent1))
        segment = parent1[i:j]
        missing = [k for k in parent2 if k not in segment]
        pref, suff = missing[:i], missing[i:]
        return pref + segment + suff

    def run(self, generations: int) -> None:
        """
        Evolves the population for a number of generations.

        Parameters
        ----------
        generations : int
            The number of generations.
        """
        start = time.time()
        print('\nEvolving...')
        for i in range(generations):
            parents = self.genomes[:self.num_parents]
            np.random.shuffle(parents)
            children = []
            for j in range(0, len(parents), 2):
                parent1 = parents[j]
                parent2 = parents[j + 1]
                for k in range(self.num_children):
                    if np.random.random() < self.prob_cross:
                        new_genes = self.ox1(parent1.genes, parent2.genes)
                        child = Genome(self.target_symbols, new_genes, freeze=self.freeze)
                    else:
                        child = Genome(self.target_symbols, parent1.genes, score=parent1.score, freeze=self.freeze)
                    if np.random.random() < self.prob_mut:
                        child.mutate()
                    children.append(child)
            population_subset = [child for child in children if child.score == -1]
            population_genes = [{self.source_symbols[x]: child.genes[x]
                                 for x in range(len(self.source_symbols))}
                                 for child in population_subset]
            for gene in population_genes:
                gene.update(self.freeze)
            pool = mp.Pool(self.n_cores)
            fitness = np.array(pool.map(self.fitness_function, population_genes))
            pool.close()
            pool.join()
            for j, child in enumerate(population_subset):
                child.score = fitness[j]
            self.genomes = children
            self.genomes.sort(key=lambda x: x.score, reverse=True)
            self.genomes = self.genomes[:self.population_size]
            self.best_scores.append(self.genomes[0].score)
            self.avg_scores.append(np.mean([genome.score for genome in self.genomes]))
            self.best_key = {self.source_symbols[i]: self.genomes[0].genes[i]
                             for i in range(len(self.source_symbols))}
            self.best_key.update(self.freeze)

            if (i + 1) % 10 == 0:
                print(f'Generation {i + 1} done')
                print(f'Best: {self.best_scores[-1]}')
                print(f'Avg: {self.avg_scores[-1]}')
                print(f'Best key: {dict(zip(self.source_symbols[:5], self.genomes[0].genes[:5]))} ...\n')

        end = time.time()
        print('Time elapsed: {:.2f}s'.format(end - start))

    def plot(self, filename: str = '') -> None:
        """
        Plots the evolution of the best and average scores of the population.

        Parameters
        ----------
        filename : str
            The filename to save the plot to. If not provided, the plot is
            shown.
        """
        plt.axes(xlabel='generation', ylabel='score')
        plt.plot(self.best_scores)
        plt.plot(self.avg_scores)
        plt.legend(['best score', 'avg score'])
        if filename:
            try:
                plt.savefig(filename)
            except IOError as e:
                print(f'Error: Unable to save plot to file: {e}')
        else:
            plt.show()

    def write_result(self, filename: str) -> None:
        """
        Writes the best key to a file.

        Parameters
        ----------
        filename : str
            The filename to write the key to.
        """
        try:
            with open(filename, 'w') as file:
                for k, v in self.best_key.items():
                    file.write(f'{k} {v}\n')
        except IOError as e:
            print(f'Error: Unable to write to file: {e}')

    def evaluate(self, eval_text: List[str]) -> None:
        """
        Evaluates the best key against a set of known values.

        Parameters
        ----------
        eval_text : List[str]
            The symbols and respective values to be used for evaluation.
        """
        eval_symbols = [x.split() for x in eval_text]
        eval_map = {x[0]: x[1] for x in eval_symbols}

        correct_count = sum(1 for k, v in self.best_key.items() if k in eval_map and eval_map[k] == v)
        total_count = len(self.best_key)

        print(f'Correct symbols: {correct_count} / {total_count} ({correct_count / total_count * 100:.2f}%)')

