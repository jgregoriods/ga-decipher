import argparse
import os

from ngram_model import NgramModel
from genetic_algorithm import GeneticAlgorithm
from processing import split_syllables, get_top_symbols, split_symbols


def main(args: argparse.Namespace) -> None:
    source_file = args.source
    target_file = args.target

    n_gram = args.n_gram

    num_symbols = args.num_symbols
    ignore = ['?'] + args.ignore.split(',') if args.ignore else ['?']
    population_size = args.pop_size
    num_parents = args.num_parents
    mutation_rate = args.mutation_rate
    crossover_rate = args.crossover_rate
    generations = args.generations
    n_cores = args.n_cores

    try:
        with open(source_file, 'r') as file:
            source_text = file.read().splitlines()
    except IOError:
        print(f'Error: Could not read file {source_file}')
        return

    try:
        with open(target_file, 'r') as file:
            target_text = file.read().splitlines()
    except IOError:
        print(f'Error: Could not read file {target_file}')
        return

    source_text_split = [split_symbols(line) for line in source_text]
    target_text_split = [split_syllables(line) for line in target_text]
    
    bigram_model = NgramModel(n_gram)
    bigram_model.fit(target_text_split)

    source_symbols = get_top_symbols(source_text_split, num_symbols, ignore)
    target_symbols = get_top_symbols(target_text_split, num_symbols)

    ga = GeneticAlgorithm(
        source_symbols,
        target_symbols,
        source_text_split,
        bigram_model,
        population_size,
        num_parents,
        mutation_rate,
        crossover_rate,
        n_cores=n_cores
    )
    ga.evolve(generations)

    if args.output:
        if not os.path.exists(args.output):
            try:
                os.makedirs(args.output)
            except OSError:
                print(f'Error: Could not create directory {args.output}')
                return
        ga.write_result(os.path.join(args.output, 'best_key.txt'))
        ga.plot(os.path.join(args.output, 'plot.png'))

    if args.eval:
        try:
            with open(args.eval, 'r') as file:
                eval_text = file.read().splitlines()
                eval_symbols = [x.split() for x in eval_text]
                eval_map = {x[0]: x[1] for x in eval_symbols}
        except IOError:
            print(f'Error: Could not read file {args.eval}')
            return

        tot = 0
        for k, v in ga.best_key.items():
            if k in eval_map and eval_map[k] == v:
                tot += 1
        print(f'Correct symbols: {tot} / {len(ga.best_key)} ({tot / len(ga.best_key) * 100:.2f}%)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('source', type=str, help='Source file')
    parser.add_argument('target', type=str, help='Target file')

    parser.add_argument('-ng', '--n-gram', type=int, default=3, help='N-gram')
    parser.add_argument('-s', '--num-symbols', type=int, default=50, help='Number of symbols')
    parser.add_argument('-i', '--ignore', type=str, help='Ignore symbols')
    parser.add_argument('-p', '--pop-size', type=int, default=1000, help='Population size')
    parser.add_argument('-n', '--num-parents', type=int, default=500, help='Number of parents')
    parser.add_argument('-m', '--mutation-rate', type=float, default=0.5, help='Mutation rate')
    parser.add_argument('-c', '--crossover-rate', type=float, default=0.8, help='Crossover rate')
    parser.add_argument('-g', '--generations', type=int, default=500, help='Number of generations')
    parser.add_argument('-nc', '--n-cores', type=int, default=0, help='Number of cores')
    parser.add_argument('-e', '--eval', type=str, help='Evaluation file')
    parser.add_argument('-o', '--output', type=str, help='Output folder')

    args = parser.parse_args()

    main(args)

