import argparse
import os

from ngram_model import NgramModel
from genetic_algorithm import GeneticAlgorithm
from processing import split_syllables, get_top_symbols, split_symbols


DEFAULT_N_GRAM = 3
DEFAULT_NUM_SYMBOLS = 50
DEFAULT_POP_SIZE = 2000
DEFAULT_NUM_PARENTS = 1000
DEFAULT_MUTATION_RATE = 0.5
DEFAULT_CROSSOVER_RATE = 0.8
DEFAULT_GENERATIONS = 1000
DEFAULT_N_CORES = 0


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=str, help='Source file')
    parser.add_argument('target', type=str, help='Target file')
    parser.add_argument('-ng', '--n-gram', type=int, default=DEFAULT_N_GRAM, help='N-gram')
    parser.add_argument('-s', '--num-symbols', type=int, default=DEFAULT_NUM_SYMBOLS, help='Number of symbols')
    parser.add_argument('-i', '--ignore', type=str, help='Ignore symbols')
    parser.add_argument('-n', '--pop-size', type=int, default=DEFAULT_POP_SIZE, help='Population size')
    parser.add_argument('-p', '--num-parents', type=int, default=DEFAULT_NUM_PARENTS, help='Number of parents')
    parser.add_argument('-m', '--mutation-rate', type=float, default=DEFAULT_MUTATION_RATE, help='Mutation rate')
    parser.add_argument('-c', '--crossover-rate', type=float, default=DEFAULT_CROSSOVER_RATE, help='Crossover rate')
    parser.add_argument('-g', '--generations', type=int, default=DEFAULT_GENERATIONS, help='Number of generations')
    parser.add_argument('-nc', '--n-cores', type=int, default=DEFAULT_N_CORES, help='Number of cores')
    parser.add_argument('-e', '--eval', type=str, help='Evaluation file')
    parser.add_argument('-o', '--output', type=str, help='Output folder')
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """
    Main function.
    """
    source_file = args.source
    target_file = args.target
    n_gram = args.n_gram
    num_symbols = args.num_symbols
    ignore = ['?'] + args.ignore.split(',') if args.ignore else ['?']

    try:
        with open(source_file, 'r') as file:
            source_text = file.read().splitlines()
    except IOError as e:
        print(f'Error: Could not read source file {source_file}: {e}')
        return

    try:
        with open(target_file, 'r') as file:
            target_text = file.read().splitlines()
    except IOError as e:
        print(f'Error: Could not read target file {target_file}: {e}')
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
        args.pop_size,
        args.num_parents,
        args.mutation_rate,
        args.crossover_rate,
        n_cores=args.n_cores
    )
    ga.evolve(args.generations)

    if args.output:
        output_folder = args.output
        if not os.path.exists(output_folder):
            try:
                os.makedirs(output_folder)
            except OSError as e:
                print(f'Error: Could not create output directory {args.output}: {e}')
                return
        ga.write_result(os.path.join(output_folder, 'best_key.txt'))
        ga.plot(os.path.join(output_folder, 'plot.png'))

    if args.eval:
        try:
            with open(args.eval, 'r') as file:
                eval_text = file.read().splitlines()
                eval_symbols = [x.split() for x in eval_text]
                eval_map = {x[0]: x[1] for x in eval_symbols}
        except IOError as e:
            print(f'Error: Could not read file {args.eval}: {e}')
            return

        correct_count = sum(1 for k, v in ga.best_key.items() if k in eval_map and eval_map[k] == v)
        total_count = len(ga.best_key)
        print(f'Correct symbols: {correct_count} / {total_count} ({correct_count / total_count * 100:.2f}%)')


if __name__ == '__main__':
    args = parse_arguments()
    main(args)

