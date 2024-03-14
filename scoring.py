def decode_line(line, cipher_key):
    decoded = []
    for symbol in line:
        if symbol in cipher_key:
            decoded.append(cipher_key[symbol])
        else:
            decoded.append("_")
    return " ".join(decoded)


# source texts must be passed as a 2d array
# e.g. [["1", "2", "1", "2"], ["3", "5", "3", "5"]]
def decode_texts(texts, cipher_key):
    decoded_texts = []
    for line in texts:
        decoded_line = decode_line(line, cipher_key)
        decoded_texts.append(decoded_line)
    return decoded_texts


# given a decipherment key, apply it to all independent
# texts and score the result
def calculate_final_score(source_texts, cipher_key, bigram_model):
    decoded_texts = decode_texts(source_texts, cipher_key)
    all_sequences = []
    for line in decoded_texts:
        decoded_sequences = line.split('_')
        valid_sequences = [seq.strip() for seq in decoded_sequences if seq.strip()]
        all_sequences.extend(valid_sequences)
    score = 0
    for sequence in all_sequences:
        score += bigram_model.score(sequence.split())
    return score

