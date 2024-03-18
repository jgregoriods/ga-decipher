# ga-decipher

## Installation

1. Install dependencies
```
pip install -r requirements.txt
```
2. Run the program
```
python main.py <source_file> <target_file> [options]
```

## Usage

To run the program, execute the main.py script with the following command-line arguments:

* `source_file`: Path to the file containing the encoded text.
* `target_file`: Path to the file containing the decoded text.

Additional options:

* `-ng, --n-gram`: N-gram size (default: 3)
* `-s, --num-symbols`: Number of symbols (default: 50)
* `-i, --ignore`: Symbols to ignore during decoding (comma-separated)
* `-n, --pop-size`: Population size (default: 1000)
* `-p, --num-parents`: Number of parents (default: 500)
* `-m, --mutation-rate`: Mutation rate (default: 0.5)
* `-c, --crossover-rate`: Crossover rate (default: 0.8)
* `-g, --generations`: Number of generations (default: 500)
* `-nc, --n-cores`: Number of CPU cores to use (default: 0, uses all available cores)
* `-e, --eval`: Evaluation file to assess accuracy
* `-o, --output`: Output folder to save results