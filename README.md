# minbpe-pytorch

Minimal, clean code for the (byte-level) Byte Pair Encoding (BPE) algorithm commonly used in LLM tokenization. The BPE algorithm is "byte-level" because it runs on UTF-8 encoded strings.

This adds PyTorch/CUDA training and encoding support to Andrej Karpathy's [minbpe](https://github.com/karpathy/minbpe).  It takes 67.4 seconds on an H100 80GB SXM5 to train the `BasicTokenizer` with a vocab_size of 512 on 308MB of Enron emails.  The original code takes 2hrs 15min on an M2 Air with Python 3.11 to do this.  That is a 120x speedup.

## Quick Start

Run `python get_enron_emails.py` to download the enron emails dataset.  A 308 MB `enron.txt` file will be saved in the `tests` directory.  Then run `python train.py` to train a `BasicTokenizer` on this text.  The model will be saved in the `models` directory.

## TODO

- Support MPS device for MacBooks, currently breaks for `torch.unique`
- Implement `RegexTokenizer` and `GPT4Tokenizer`

## License

MIT