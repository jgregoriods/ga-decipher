import re


def split_syllables(word):
    word = word.lower().replace(" ", "")
    return re.sub(r'(a|e|i|o|u)', r'\1.', word).split('.')[:-1]


# merge this with the previous function
def split_japanese_syllables(sentence):
    res = []
    words = sentence.split()
    for i in range(len(words)):
        word = words[i]

        x = re.sub(r"([aeiouāēīōū])", r"\1|", word)
        x = re.sub(r"n(?![aeiouāēīōū])", r"n|", x)
        
        x = x.replace("tt", "t|t").replace("kk", "k|k").replace("pp", "p|p")
        
        x = x.split("|")

        syllables = [j for j in x if j]

        res += syllables
    return res


def get_top_symbols(corpus, n, ignore=[]):
    symbols = {}
    for line in corpus:
        for symbol in line:
            symbols[symbol] = symbols.get(symbol, 0) + 1
    for symbol in ignore:
        symbols.pop(symbol, None)
    return sorted(symbols, key=symbols.get, reverse=True)[:n]

