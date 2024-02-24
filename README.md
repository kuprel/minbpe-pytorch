# minbpe-pytorch

Minimal, clean code for the (byte-level) Byte Pair Encoding (BPE) algorithm commonly used in LLM tokenization. The BPE algorithm is "byte-level" because it runs on UTF-8 encoded strings.

This adds PyTorch/CUDA training support to Andrej Karpathy's [minbpe](https://github.com/karpathy/minbpe).  It takes 74.8 seconds on an H100 to train the `BasicTokenizer` with a vocab_size of 512 on 308MB of Enron emails.  The original code takes 2hrs 15min on an M2 Air with Python 3.11 to do this.  That is over 100x speedup.

## Usage

This script is contained in `train_pytorch.py`

```python
import os
import time
import torch
from minbpe import BasicTokenizer

# open some text and train a vocab of 512 tokens
text = open("tests/taylorswift.txt", "r", encoding="utf-8").read()

# create a directory for models, so we don't pollute the current directory
os.makedirs("models", exist_ok=True)

t0 = time.time()

# construct the Tokenizer object and kick off verbose training
tokenizer = BasicTokenizer()
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training with {device}")
tokenizer.train_pytorch(text, 512, verbose=True, device=device)
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

## Implementation

The `train_pytorch` method is implemented in `minbpe.py` as follows:

```python
def train_pytorch(self, text: str, vocab_size: int, verbose=False, device='cuda'):
    assert vocab_size >= 256
    num_merges = vocab_size - 256

    # input text preprocessing
    text_bytes = text.encode("utf-8") # raw bytes
    ids = list(text_bytes) # list of integers in range 0..255

    int_type = torch.int16 if vocab_size <= 2**15 else torch.int32
    ids = torch.tensor(ids, dtype=int_type, device=device)
    merges = torch.zeros((num_merges, 2), dtype=int_type, device=device)
    false_tensor = torch.tensor([False], dtype=torch.bool, device=device)

    for i in range(num_merges):
        # determine the most common pair to merge next
        pairs = torch.stack((ids[:-1], ids[1:]), dim=1)
        unique, counts = torch.unique(pairs, return_counts=True, dim=0)
        pair_index = torch.argmax(counts)
        pair, count = unique[pair_index], counts[pair_index]
        merges[i] = pair

        # merge the pair
        # create a mask for the first element of every matching pair
        is_first_in_pair: torch.Tensor = torch.all(pairs == pair, axis=1)
        is_first_in_pair = torch.cat((is_first_in_pair, false_tensor))
        # create a mask for the second element of every matching pair
        is_second_in_pair = torch.roll(is_first_in_pair, 1, 0)
        # each token can only belong to one pair
        is_first_in_pair &= ~is_second_in_pair
        # change the first element of every occurrence of the pair to the new id
        ids[is_first_in_pair] = i + 256
        # remove the second element of every occurrence of the pair
        ids = ids[~is_second_in_pair]

        if verbose:
            print(f"merge {i+1}/{num_merges}: {tuple(pair.tolist())} -> {i + 256} had {count} occurrences")

    merges = merges.cpu().numpy()
    merges = [tuple(pair) for pair in merges]

    self.merges = {pair: j + 256 for j, pair in enumerate(merges)}

    vocab = {idx: bytes([idx]) for idx in range(256)} # int -> bytes
    for i in range(num_merges):
        idx = 256 + i
        pair = merges[i]
        vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
        if verbose:
            print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]})")
    self.vocab = vocab
```

## TODO

- Add `encode_pytorch` method
- Add MPS device support for MacBooks, currently breaks for `torch.unique`

## License

MIT