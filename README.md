# ga-decipher

A simple implementation of beam search and a genetic algorithm to solve substitution ciphers, aimed at the decipherment of unknown scripts. The model uses a simple n-gram model to score candidate mappings and beam search or a genetic algorithm to search for the best solution. The program was devised to solve the Rongorongo script of Easter Island, but in theory can be used for any other unknown script.

## Installation

```
pip install git+https://github.com/jgregoriods/ga-decipher.git
```

## Usage

The source and target texts are expected to be simple text files with newline separated sentences. If an evaluation file is provided, it is expected to be a text file with pairs of source and target symbols separated by a space. Refer to the sample files in the `data` folder as examples.

## Example

For running the Japanese experiment with the hiragana text as source, evaluating the results on the correct mapping file and writing the results to a `japanese_results` folder:

```python
import os

from ga_decipher.text_processor import TextProcessor
from ga_decipher.ngram_model import NgramModel
from ga_decipher.solvers.beam_search import BeamSearch
from ga_decipher.utils import read_file


source_lines = read_file('data/hiragana.txt')
target_lines = read_file('data/japanese.txt')

source_text = TextProcessor(source_lines).split('symbol')
target_text = TextProcessor(target_lines).split('syllable')

ngram_model = NgramModel(3)
ngram_model.fit(target_text.splitted_text)

source_symbols = source_text.get_top_symbols(50)
target_symbols = target_text.get_top_symbols(50)

solver = BeamSearch(
    source_symbols=source_symbols,
    target_symbols=target_symbols,
    source_text=source_text.splitted_text,
    model=ngram_model,
    nodes=10,
    beam_width=100
)

solver.run(200)

if not os.path.exists('japanese_results'):
    try:
        os.makedirs('japanese_results')
    except OSError as e:
        print(f'Error: Could not create output directory: {e}')
        return

solver.write_result(os.path.join('japanese_results', 'best_key.txt'))
solver.plot(os.path.join('japanese_results', 'plot.png'))

eval_text = read_file('data/japanese_mapping.txt')
solver.evaluate(eval_text)
```

## Experiment

The model was tested on a corpus of Japanese sentences obtained from <a href="https://tatoeba.org">tatoeba</a>. The choice of Japanese is due to its logosyllabic writing system and relatively simple syllable structure, which make it an appropriate modern parallel for Rongorongo and Rapanui.

Two versions of the Japanese corpus were used - an artifically generated one where only hiragana was used, and the original one with mixed kanji, hiragana and katakana. The former allows for assessing performance for a purely syllabic writing system, while the latter more realistically represents a noisy scenario where logograms are mixed with syllables. A sample of 1000 sentences were used for fitting the language model, and another 1000 for decipherment.

The model correctly identifies 86% of the syllabic symbols in the hiragana corpus. Of the top 10 symbols, all are correctly identified (if one counts the reading of は as *wa*).

## Results

The Rapanui language model was fitted on a sample of short recitations and songs assumed to represent the genres and language present in Rongorongo (Barthel 1960; Blixen 1979; Campbell 1971). For the written corpus, the set of texts removing parallel passages and repetitive structured sequences usually called the "independent text" (Horley 2007) was used. The original Barthel encoding was converted to the simplified proposal by Horley (2021). For details, the reader is referred to the discussion and code in <a href="https://github.com/jgregoriods/rongopy">my other rongorongo repository</a>.

A second corpus was prepared by removing the anthropomorphs 200, 381 and 256, as well as glyph 3, under the assumption the former are allographs and are frequently omitted, and that the latter appears to be a decorator (see the discussion <a href="https://github.com/jgregoriods/rongopy/tree/master/ga_lstm">here</a>). The results are presented in the table below.


| Glyph | Code | Syllable<sub>1</sub> | Syllable<sub>2</sub> |
| --- | --- | --- | --- |
| <img src="img/200.png" height="32"> | 200 | a | |
| <img src="img/6.png" height="32"> | 6 | i | a |
| <img src="img/10.png" height="32"> | 10 | u | e |
| <img src="img/1.png" height="32"> | 1 | e | u |
| <img src="img/600.png" height="32"> | 600 | te | i |
| <img src="img/4.png" height="32"> | 4 | ma | te |
| <img src="img/2.png" height="32"> | 2 | ta | ta |
| <img src="img/3.png" height="32"> | 3 | o | |
| <img src="img/381.png" height="32"> | 381 | ka | |
| <img src="img/256.png" height="32"> | 256 | ru | |
| <img src="img/8.png" height="32"> | 8 | tu | ma |
| <img src="img/430.png" height="32"> | 430 | re | ra |
| <img src="img/9.png" height="32"> | 9 | ha | ha |
| <img src="img/5.png" height="32"> | 5 | ro | ka |

It is curious that the reading of 381 as *ka* has previously been suggested based on independent evidence (Davletshin 2022).

This is by no means an endorsement that the above solution is correct, as it heavily depends on the correct identification of the glyph inventory, the nature of rongorongo as mainly syllabic, the assumption that the chosen rapanui corpus is representative of the language in the glyphs, among other factors.
