import re

from typing import List


class TextProcessor:
    def __init__(self, text: List[str]):
        self.text = text
        self.splitted_text = []

    def split(self, by: str = 'symbol') -> 'TextProcessor':
        if by == 'symbol':
            self.splitted_text = [line.split() if ' ' in line else list(line) for line in self.text]
        elif by == 'syllable':
            for line in self.text:
                res = []
                words = line.split()
                for word in words:
                    word = re.sub(r'([aeiou])', r'\1.', word)
                    word = re.sub(r'n(?![aeiou])', r'n.', word)
                    word = re.sub(r'([ptk])([ptk])', r'\1.\2', word)
                    word = word.split('.')
                    syllables = [syllable for syllable in word if syllable]
                    res += syllables
                self.splitted_text.append(res)
        else:
            raise ValueError("Invalid split type. Use 'symbol' or 'syllable'.")
        return self

    def get_top_symbols(self, n: int = 50, ignore: List[str] = list()) -> List[str]:
        symbols = {}
        for line in self.splitted_text:
            for symbol in line:
                symbols[symbol] = symbols.get(symbol, 0) + 1
        for symbol in ignore:
            symbols.pop(symbol, None)
        return sorted(symbols, key=symbols.get, reverse=True)[:n]

