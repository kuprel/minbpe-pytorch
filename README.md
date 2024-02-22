# minbpe-cuda

Minimal, clean code for the (byte-level) Byte Pair Encoding (BPE) algorithm commonly used in LLM tokenization. The BPE algorithm is "byte-level" because it runs on UTF-8 encoded strings.

This is a fork of Andrej Karpathy's [minbpe](https://github.com/karpathy/minbpe) that adds CUDA support to the `train` method of the `BasicTokenizer` class.  When the `BasicTokenizer` is trained with a vocab_size of 512 on 307MB of Enron emails, it only took 148 seconds on an RTX4090.  The original code takes TBD to do this on my M2 macbook.

## quick start

This script is in `train_cuda.py`

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
tokenizer.train_cuda(text, 512, verbose=True)
# writes two files in the models directory: name.model, and name.vocab
prefix = os.path.join("models", "basic")
tokenizer.save(prefix)

t1 = time.time()

print(f"Training took {t1 - t0:.2f} seconds")
```

## TODO

Address repeated characters, e.g. "aaabdaaabac"
Encode on cuda

## License

MIT