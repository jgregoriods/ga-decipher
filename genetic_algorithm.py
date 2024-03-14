import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import time

from scoring import calculate_final_score


class Genome:
    def __init__(self, target_symbols, genes=None, score=None, freeze=None):
        random_symbols = target_symbols[:]
        if freeze:
            random_symbols = [i for i in random_symbols
                              if i not in list(freeze.values())]
        np.random.shuffle(random_symbols)
        self.genes = genes or random_symbols
        self.score = score

    def mutate(self):
        j, k = np.random.choice([i for i in range(len(self.genes) - 1)],
                                size=2, replace=False)
        self.genes[j], self.genes[k] = self.genes[k], self.genes[j]
        self.score = None


class GeneticAlgorithm:
    def __init__(self, source_symbols, target_symbols, source_text, model,
                 population_size, num_parents, num_elite, prob_cross, prob_mut,
                 freeze=None, n_cores=10):
        self.source_symbols = source_symbols
        self.target_symbols = target_symbols
        self.source_text = source_text
        self.model = model
        self.population_size = population_size
        self.num_parents = num_parents
        self.num_elite = num_elite
        self.num_children = int(np.ceil(population_size / num_parents * 2))
        self.prob_cross = prob_cross
        self.prob_mut = prob_mut
        self.freeze = freeze or dict()
        self.n_cores = n_cores

        if freeze:
            self.source_symbols = [symbol for symbol in self.source_symbols
                                   if symbol not in freeze]

        print('Initializing population...')
        start = time.time()
        self.genomes = [Genome(target_symbols, freeze=self.freeze)
                        for i in range(self.population_size)]
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
        self.best_key = {self.source_symbols[i]: self.genomes[0].genes[i]
                         for i in range(len(self.source_symbols))}
        self.best_key.update(self.freeze)

    def fitness_function(self, cipher_key):
        return calculate_final_score(self.source_text, cipher_key, self.model)

    def ox1(self, parent1, parent2):
        """ Order crossover """
        i = np.random.randint(0, len(parent1) - 1)
        j = np.random.randint(i + 1, len(parent1))
        segment = parent1[i:j]
        missing = [k for k in parent2 if k not in segment]
        pref, suff = missing[:i], missing[i:]
        return pref + segment + suff

    def evolve(self, generations):
        print('Evolving...')
        for i in range(generations):
            elite = self.genomes[:self.num_elite]
            parents = self.genomes[:self.num_parents]
            np.random.shuffle(parents)
            children = []
            for j in range(0, len(parents), 2):
                parent1 = parents[j]
                parent2 = parents[j+1]
                for k in range(self.num_children):
                    if np.random.random() < self.prob_cross:
                        new_genes = self.ox1(parent1.genes, parent2.genes)
                        # target symbols can be None by default in Genome
                        child = Genome(self.target_symbols, new_genes, freeze=self.freeze)
                    else:
                        child = Genome(self.target_symbols, parent1.genes, score=parent1.score, freeze=self.freeze)
                    if np.random.random() < self.prob_mut:
                        child.mutate()
                    children.append(child)
            population_subset = [child for child in children if child.score is None]
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
            self.genomes = elite + children
            self.genomes.sort(key=lambda x: x.score, reverse=True)
            self.genomes = self.genomes[:self.population_size]
            self.best_scores.append(self.genomes[0].score)
            self.avg_scores.append(np.mean([genome.score for genome in self.genomes]))
            self.best_key = {self.source_symbols[i]: self.genomes[0].genes[i]
                             for i in range(len(self.source_symbols))}
            self.best_key.update(self.freeze)
            print('\rGen {}\tBest: {:.2f}\tAvg: {:.2f}\t{}'.format(i+1, self.best_scores[-1],
                                                                   self.avg_scores[-1],
                                                                   self.genomes[0].genes[:5]), end='')
        print('\n')

    def plot(self):
        plt.style.use('seaborn')
        plt.axes(xlabel='generation', ylabel='score')
        plt.plot(self.best_scores, color='#F8766D')
        plt.plot(self.avg_scores, color='black')
        plt.legend(['best score', 'avg score'])
        plt.show()
