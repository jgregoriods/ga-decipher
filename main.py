from bigram_model import BigramModel
from genetic_algorithm import GeneticAlgorithm
from processing import split_japanese_syllables, get_top_symbols


def main():
    source_file = "data/hiragana.txt"
    target_file = "data/japanese.txt"

    num_symbols = 48
    ignore = ["?"]

    population_size = 2000
    num_parents = 1000
    mutation_rate = 0.5
    crossover_rate = 0.8
    generations = 100

    with open(source_file, "r") as file:
        source_text = file.read().splitlines()

    with open(target_file, "r") as file:
        target_text = file.read().splitlines()

    source_text_split = [list(x) for x in source_text]
    target_text_split = [split_japanese_syllables(line) for line in target_text]
    
    bigram_model = BigramModel()
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
        0,
        crossover_rate,
        mutation_rate
    )
    ga.evolve(generations)


if __name__ == "__main__":
    main()
