# minbpe-pytorch

Minimal, clean code for the (byte-level) Byte Pair Encoding (BPE) algorithm commonly used in LLM tokenization. The BPE algorithm is "byte-level" because it runs on UTF-8 encoded strings.

This adds PyTorch/CUDA training support to Andrej Karpathy's [minbpe](https://github.com/karpathy/minbpe).  It takes 74.8 seconds on an H100 to train the `BasicTokenizer` with a vocab_size of 512 on 308MB of Enron emails.  The original code takes 2hrs 15min on an M2 Air with Python 3.11 to do this.  That is over 100x speedup.

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
tokenizer.train_pytorch(text, 512, verbose=True, device="cuda")
# writes two files in the models directory: name.model, and name.vocab
prefix = os.path.join("models", "basic")
tokenizer.save(prefix)

t1 = time.time()

print(f"Training took {t1 - t0:.2f} seconds")

print("Testing the model")
tok = BasicTokenizer()
tok.load(prefix + ".model")
assert(tok.decode(tok.encode(text)) == text)
print("Success")
```

## Repeated Characters Bug

The `merge` method is vectorized in PyTorch as follows:

```python
# create a mask for the pair
mask = torch.all(pairs == pair, dim=1)
# append a False to the mask to make it the same length as ids
mask = torch.cat((mask, torch.tensor([False]).cuda()))
# change the first element of every occurrence of the pair to the new id
ids[mask] = i + 256
# remove the second element of every occurrence of the pair
ids = ids[~torch.roll(mask, 1, 0)]
```

This results in undesired behavior when a character is repeated more than 2 times.  For example, 'aaa' is not handled properly since there are 2 pairs of 'aa' in the triple, (aa)a and a(aa).  What happens is that all repeated characters are replaced with one token, i.e if X = aa then aaa -> X not Xa.  This bug doesn't seem to have much effect on training the vocab though.

## TODO

- Train on Project Gutenberg
- Add PyTorch support for `encode` method
- Add MPS device support for MacBooks, currently breaks for `torch.unique`
- Fix repeated characters bug?

## License

MIT