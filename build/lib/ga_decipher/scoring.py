from typing import List, Dict
from ga_decipher.ngram_model import NgramModel


def decode_line(line: List[str], cipher_key: Dict[str, str]) -> str:
    """
    Decodes a line using a cipher key.

    Parameters
    ----------
    line : List[str]
        The line to decode.
    cipher_key : Dict[str, str]
        The cipher key to use.

    Returns
    -------
    str
        The decoded line.
    """
    decoded = []
    for symbol in line:
        decoded.append(cipher_key.get(symbol, '_'))
    return ' '.join(decoded)


def decode_texts(texts: List[List[str]], cipher_key: Dict[str, str]) -> List[str]:
    """
    Decodes a text using a cipher key.

    Parameters
    ----------
    texts : List[List[str]]
        The texts to decode.
    cipher_key : Dict[str, str]
        The cipher key to use.

    Returns
    -------
    List[str]
        The decoded texts.
    """
    decoded_texts = []
    for line in texts:
        decoded_line = decode_line(line, cipher_key)
        decoded_texts.append(decoded_line)
    return decoded_texts


def calculate_final_score(source_texts: List[List[str]], cipher_key: Dict[str, str],
                          ngram_model: NgramModel, method: str = 'log_prob') -> float:
    """
    Calculates the final score of a cipher key.

    Parameters
    ----------
    source_texts : List[List[str]]
        The source texts to decode.
    cipher_key : Dict[str, str]
        The cipher key to use.
    ngram_model : NgramModel
        The n-gram model to use.
    method : str
        The scoring method to use ('log_prob' or 'count').

    Returns
    -------
    float
        The final score of the cipher key.
    """
    decoded_texts = decode_texts(source_texts, cipher_key)
    all_sequences = []
    for line in decoded_texts:
        decoded_sequences = line.split('_')
        valid_sequences = [seq.strip() for seq in decoded_sequences if seq.strip()]
        all_sequences.extend(valid_sequences)
    score = 0
    for sequence in all_sequences:
        if method == 'log_prob':
            score += ngram_model.log_prob(sequence.split())
        elif method == 'count':
            score += ngram_model.score(sequence.split())
    return score
