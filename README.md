# minbpe-pytorch

Minimal, clean code for the (byte-level) Byte Pair Encoding (BPE) algorithm commonly used in LLM tokenization. The BPE algorithm is "byte-level" because it runs on UTF-8 encoded strings.

This adds PyTorch/CUDA training and encoding support to Andrej Karpathy's [minbpe](https://github.com/karpathy/minbpe).  It takes 67.4 seconds on an H100 80GB SXM5 to train the `BasicTokenizer` with a vocab_size of 512 on 308MB of Enron emails.  The original code takes 2hrs 15min on an M2 Air with Python 3.11 to do this.  That is a 120x speedup.

## quick start

Install requirements:

```bash
$ pip install -r requirements.txt
```

Download Enron emails and save a 308MB text file to `tests/enron.txt`:

```bash
$ python get_enron_emails.py
```

Train a `BasicTokenizer` on the large text file:

```bash
$ python train.py
```

The model will be saved in the `models` directory.

## tests

The pytest library is used for tests. All of them are located in the `tests/` directory. First `pip install pytest`, then:

```bash
$ pytest -v .
```

## todo

- Speed up `encode` method for `RegexTokenizer`
- Implement `train` method for `RegexTokenizer`
- Support MPS device for MacBooks, currently breaks for `torch.unique`

## License

MIT