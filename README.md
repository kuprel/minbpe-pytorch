# minbpe-pytorch

Minimal, clean code for the (byte-level) Byte Pair Encoding (BPE) algorithm commonly used in LLM tokenization. The BPE algorithm is "byte-level" because it runs on UTF-8 encoded strings.

This adds PyTorch/CUDA training and encoding support to Andrej Karpathy's [minbpe](https://github.com/karpathy/minbpe).  It takes 67.4 seconds on an H100 80GB SXM5 to train the `BasicTokenizer` with a vocab_size of 512 on 308MB of Enron emails.  The original code takes 2hrs 15min on an M2 Air with Python 3.11 to do this.  That is a 120x speedup.

## Usage

This script is contained in `train.py`

```python
import os
import time
import torch
from minbpe import BasicTokenizer

# open some text and train a vocab of 512 tokens
text = open("taylorswift.txt", "r", encoding="utf-8").read()

# create a directory for models, so we don't pollute the current directory
os.makedirs("models", exist_ok=True)

t0 = time.time()

# construct the Tokenizer object and kick off verbose training
tokenizer = BasicTokenizer()
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training with {device}")
tokenizer.train(text, 512, verbose=True, device=device)
# writes two files in the models directory: name.model, and name.vocab
prefix = os.path.join("models", "basic")
tokenizer.save(prefix)

t1 = time.time()

print(f"Training took {t1 - t0:.2f} seconds")

print("Testing the model")
assert(tokenizer.decode(tokenizer.encode(text)) == text)
print("Success")
```

## TODO

- Support MPS device for MacBooks, currently breaks for `torch.unique`
- Implement `RegexTokenizer` and `GPT4Tokenizer`

## License

MIT