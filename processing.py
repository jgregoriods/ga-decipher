import re

from typing import List


def split_syllables(sentence: str) -> List[str]:
    """
    Splits a target sentence into syllables.
    For now, this is a simple implementation that works
    for Rapanui and Japanese.

    Parameters
    ----------
    sentence : str
        The sentence to split.
    
    Returns
    -------
    List[str]
        A list of syllables.
    """
    res = []
    words = sentence.split()
    for word in words:
        word = re.sub(r'([aeiou])', r'\1.', word)
        word = re.sub(r'n(?![aeiou])', r'n.', word)
        word = re.sub(r'([ptk])([ptk])', r'\1.\2', word)
        word = word.split('.')
        syllables = [syl for syl in word if syl]
        res += syllables
    return res


def get_top_symbols(corpus: List[List[str]], n: int, ignore: List[str] = list()) -> List[str]:
    """
    Gets the top n symbols from a corpus.

    Parameters
    ----------
    corpus : List[List[str]]
        The corpus to get the symbols from.
    n : int
        The number of symbols to get.
    ignore : List[str]
        A list of symbols to ignore.

    Returns
    -------
    List[str]
        A list of the top n symbols.
    """
    symbols = {}
    for line in corpus:
        for symbol in line:
            symbols[symbol] = symbols.get(symbol, 0) + 1
    for symbol in ignore:
        symbols.pop(symbol, None)
    return sorted(symbols, key=symbols.get, reverse=True)[:n]


def split_symbols(sentence: str) -> List[str]:
    """
    Splits a source sentence into symbols.

    Parameters
    ----------
    sentence : str
        The sentence to split.

    Returns
    -------
    List[str]
        A list of symbols.
    """
    return sentence.split() if ' ' in sentence else list(sentence)

