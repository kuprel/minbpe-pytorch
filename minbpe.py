import unicodedata
import torch
from torch import Tensor

def merge(ids: Tensor, pair: Tensor, idx: int):
    """
    In the list of integers (ids), replace all consecutive occurrences
    of pair with the new integer token idx
    Example: ids=[1, 2, 3, 1, 2], pair=(1, 2), idx=4 -> [4, 3, 4]
    """
    # create a mask for the first element of every matching pair
    pairs = torch.stack((ids[:-1], ids[1:]), dim=1)
    is_first_in_pair = (pairs == pair).all(axis=1)
    false_tensor = torch.tensor([False], dtype=torch.bool, device=ids.device)
    is_first_in_pair = torch.cat((is_first_in_pair, false_tensor))
    # create a mask for the second element of every matching pair
    is_second_in_pair = is_first_in_pair.roll(1)
    # each token can only belong to one pair
    is_first_in_pair &= ~is_second_in_pair
    is_second_in_pair = is_first_in_pair.roll(1)
    # change the first element of every matching pair to the new token
    ids[is_first_in_pair] = idx
    # remove the second element of every matching pair
    ids = ids[~is_second_in_pair]
    return ids

# first two helper functions...
def replace_control_characters(s: str) -> str:
    # we don't want to print control characters
    # which distort the output (e.g. \n or much worse)
    # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python/19016117#19016117
    # http://www.unicode.org/reports/tr44/#GC_Values_Table
    chars = []
    for ch in s:
        if unicodedata.category(ch)[0] != "C":
            chars.append(ch) # this character is ok
        else:
            chars.append(f"\\u{ord(ch):04x}") # escape
    return "".join(chars)

def render_token(t: bytes) -> str:
    # pretty print a token, escaping control characters
    s = t.decode('utf-8', errors='replace')
    s = replace_control_characters(s)
    return s

class BasicTokenizer:

    def __init__(self):
        # default: vocab size of 256 (all bytes), no merges, no patterns
        self.merges = {} # (int, int) -> int
        self.pattern = "" # str
        self.special_tokens = {} # str -> int, e.g. {'<|endoftext|>': 100257}
        self.vocab = self._build_vocab() # int -> bytes

    def train_pytorch(self, text: str, vocab_size: int, verbose=False, device='cuda'):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        # input text preprocessing
        text_bytes = text.encode("utf-8") # raw bytes
        ids = list(text_bytes) # list of integers in range 0..255

        int_type = torch.int16 if vocab_size <= 2**15 else torch.int32
        ids = torch.tensor(ids, dtype=int_type, device=device)
        merges = torch.zeros((num_merges, 2), dtype=int_type, device=device)

        for i in range(num_merges):
            # determine the most common pair to merge next
            pairs = torch.stack((ids[:-1], ids[1:]), dim=1)
            unique, counts = torch.unique(pairs, return_counts=True, dim=0)
            pair_index = torch.argmax(counts)
            pair, count = unique[pair_index], counts[pair_index]

            ids = merge(ids, pair, i + 256)
            merges[i] = pair

            if verbose:
                print(f"merge {i+1}/{num_merges}: {tuple(pair.tolist())} -> {i + 256} had {count} occurrences")

        self.merges_tensor = merges
        merges = merges.cpu().numpy()
        merges = [tuple(pair) for pair in merges]

        self.merges = {pair: i + 256 for i, pair in enumerate(merges)}

        vocab = {idx: bytes([idx]) for idx in range(256)} # int -> bytes
        for i in range(num_merges):
            idx = i + 256
            pair = merges[i]
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]})")
        self.vocab = vocab

    def decode(self, ids):
        # given ids (list of integers), return Python string
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    def encode(self, text: str, device: str = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        # given a string text, return the token ids
        text_bytes = text.encode("utf-8") # raw bytes
        ids = list(text_bytes) # list of integers in range 0..255

        int_type = torch.int16 if len(self.merges) <= 2**15 else torch.int32
        ids = torch.tensor(ids, dtype=int_type, device=device)

        while len(ids) >= 2:
            # find the pair with the lowest merge index
            pairs = torch.stack((ids[:-1], ids[1:]), dim=1)
            unique: Tensor = torch.unique(pairs, dim=0)

            merges = self.merges_tensor
            is_present = (merges[:, None] == unique[None]).all(-1).any(-1)
            if not is_present.any():
                break
            pair_index = is_present.nonzero()[0]
            pair = merges[pair_index]

            idx = pair_index.to(ids.dtype) + 256
            ids = merge(ids, pair, idx)
        return ids.cpu().tolist()

    def _build_vocab(self):
        # vocab is simply and deterministically derived from merges
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab

    def save(self, file_prefix):
        """
        Saves two files: file_prefix.vocab and file_prefix.model
        This is inspired (but not equivalent to!) sentencepiece's model saving:
        - model file is the critical one, intended for load()
        - vocab file is just a pretty printed version for human inspection only
        """
        # write the model: to be used in load() later
        model_file = file_prefix + ".model"
        with open(model_file, 'w') as f:
            # write the version, pattern and merges, that's all that's needed
            f.write("minbpe v1\n")
            f.write(f"{self.pattern}\n")
            # write the special tokens, first the number of them, then each one
            f.write(f"{len(self.special_tokens)}\n")
            for special, idx in self.special_tokens.items():
                f.write(f"{special} {idx}\n")
            # the merges dict
            for idx1, idx2 in self.merges:
                f.write(f"{idx1} {idx2}\n")
        # write the vocab: for the human to look at
        vocab_file = file_prefix + ".vocab"
        inverted_merges = {idx: pair for pair, idx in self.merges.items()}
        with open(vocab_file, "w", encoding="utf-8") as f:
            for idx, token in self.vocab.items():
                # note: many tokens may be partial utf-8 sequences
                # and cannot be decoded into valid strings. Here we're using
                # errors='replace' to replace them with the replacement char �.
                # this also means that we couldn't possibly use .vocab in load()
                # because decoding in this way is a lossy operation!
                s = render_token(token)
                # find the children of this token, if any
                if idx in inverted_merges:
                    # if this token has children, render it nicely as a merge
                    idx0, idx1 = inverted_merges[idx]
                    s0 = render_token(self.vocab[idx0])
                    s1 = render_token(self.vocab[idx1])
                    f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
                else:
                    # otherwise this is leaf token, just print it
                    # (this should just be the first 256 tokens, the bytes)
                    f.write(f"[{s}] {idx}\n")

    def load(self, model_file):
        """Inverse of save() but only for the model file"""
        assert model_file.endswith(".model")
        # read the model file
        merges = {}
        special_tokens = {}
        idx = 256
        with open(model_file, 'r', encoding="utf-8") as f:
            # read the version
            version = f.readline().strip()
            assert version == "minbpe v1"
            # read the pattern
            self.pattern = f.readline().strip()
            # read the special tokens
            num_special = int(f.readline().strip())
            for _ in range(num_special):
                special, special_idx = f.readline().strip().split()
                special_tokens[special] = int(special_idx)
            # read the merges
            for line in f:
                idx1, idx2 = map(int, line.split())
                merges[(idx1, idx2)] = idx
                idx += 1
        self.merges = merges
        self.special_tokens = special_tokens
        self.vocab = self._build_vocab()
