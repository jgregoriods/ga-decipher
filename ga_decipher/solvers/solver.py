from typing import List
from ga_decipher.ngram_model import NgramModel


class Solver:
    def __init__(self, source_symbols: List[str], target_symbols: List[str],
                 source_text: List[List[str]], model: NgramModel):
        self.source_symbols = source_symbols
        self.target_symbols = target_symbols
        self.source_text = source_text
        self.model = model

    def run(self, iterations: int, callback: callable = None) -> None:
        raise NotImplementedError('Solver.run() is not implemented')

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
