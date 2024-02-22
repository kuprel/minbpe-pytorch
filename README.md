# minbpe-pytorch

Minimal, clean code for the (byte-level) Byte Pair Encoding (BPE) algorithm commonly used in LLM tokenization. The BPE algorithm is "byte-level" because it runs on UTF-8 encoded strings.

This uses PyTorch to add GPU training support to Andrej Karpathy's [minbpe](https://github.com/karpathy/minbpe).  It takes 148 seconds on an RTX4090 to train the `BasicTokenizer` with a vocab_size of 512 on 307MB of Enron emails.  The original code takes TBD on an M2 MacBook Air to do this.

## Usage

This script is contained in `train_pytorch.py`

```python
import os
import time
from minbpe import BasicTokenizer

# open some text and train a vocab of 512 tokens
text = open("tests/taylorswift.txt", "r", encoding="utf-8").read()

# create a directory for models, so we don't pollute the current directory
os.makedirs("models", exist_ok=True)

t0 = time.time()

# construct the Tokenizer object and kick off verbose training
tokenizer = BasicTokenizer()
tokenizer.train_pytorch(text, 512, verbose=True)
# writes two files in the models directory: name.model, and name.vocab
prefix = os.path.join("models", "basic")
tokenizer.save(prefix)

t1 = time.time()

print(f"Training took {t1 - t0:.2f} seconds")
```

## TODO

- Address what happens with repeated characters, e.g. "aaabdaaabac"
- Add GPU support for `encode` method
- Train on Project Gutenberg

## License

MIT