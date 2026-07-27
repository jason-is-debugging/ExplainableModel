import os
from re import A
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple

class Tokenizer:
    """Character-level tokenizer that maps characters to integer IDs and back.

    Includes four reserved special tokens by default:
        - <pad>: padding token (id 0)
        - <bos>: beginning of sequence (id 1)
        - <eos>: end of sequence (id 2)
        - <unk>: unknown token for out-of-vocabulary characters (id 3)

    Attributes:
        vocab: Ordered list of all known tokens.
        stoi: Mapping from token string to integer id.
        itos: Mapping from integer id to token string.
        pad_token_id, bos_token_id, eos_token_id, unk_token_id: Cached ids for the special tokens.
    """

    def __init__(self, vocab: List[str] = None) -> None:
        """Initialize the tokenizer, optionally from a pre-built vocabulary.

        Args:
            vocab: Optional list of tokens to seed the vocabulary with.
                When ``None``, the four default special tokens are used.
        """
        if vocab is None:
            vocab = ["<pad>", "<bos>", "<eos>", "<unk>"]

        self.vocab = vocab
        self.stoi = {ch: i for i, ch in enumerate(vocab)}
        self.itos = {i: ch for i, ch in enumerate(vocab)}

        self.pad_token = "<pad>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        self.unk_token = "<unk>"

        self.pad_token_id = self.stoi[self.pad_token]
        self.bos_token_id = self.stoi[self.bos_token]
        self.eos_token_id = self.stoi[self.eos_token]
        self.unk_token_id = self.stoi[self.unk_token]


    @property
    def vocab_size(self) -> int:
        """Return the total number of tokens currently in the vocabulary."""
        return len(self.vocab)

    def build_vocab(self, texts: List[str], min_freq: int = 1) -> 'Tokenizer':
        """Build (extend) the vocabulary by counting characters in ``texts``.

        Characters occurring at least ``min_freq`` times are appended to the
        vocabulary, the string-to-index and index-to-string maps are updated
        accordingly, and the encoder/decoder maps are kept in sync.

        Args:
            texts: Corpus of strings to mine characters from.
            min_freq: Minimum character frequency required to be added.

        Returns:
            The tokenizer instance (``self``) to allow method chaining.
        """
        freq  = {}
        for text in texts:
            for char in text:
                freq[char] = freq.get(char, 0) + 1

        for char, count in freq.items():
            if count >= min_freq and char not in self.vocab:
                self.vocab.append(char)
                self.stoi[char] = len(self.vocab) - 1
                self.itos[len(self.vocab) - 1] = char


        return self

    def encode(self, text: str) -> List[int]:
        """Encode a string into a list of token IDs.

        Unknown characters are mapped to ``unk_token_id``.

        Args:
            text: Input string to encode.

        Returns:
            List of integer token IDs, one per character.
        """
        return [self.stoi.get(char, self.unk_token_id) for char in text]

    def decode(self, ids: List[int]) -> str:
        """Decode a list of token IDs back into a string.

        Missing IDs are mapped to ``unk_token`` so decoding never raises.

        Args:
            ids: List of integer token IDs.

        Returns:
            The reconstructed string.
        """
        return ''.join([self.itos.get(id, self.unk_token) for id in ids])

    def __len__(self) -> int:
        """Return the size of the vocabulary (same as ``vocab_size``)."""
        return self.vocab_size

    def __getitem__(self, index: int) -> str:
        """Return the token at ``index`` (enables ``tokenizer[i]`` syntax)."""
        return self.vocab[index]


    def add_special_tokens(self, tokens: List[str]) -> None:
        """Append new tokens to the vocabulary and register their IDs.

        Only tokens not already present are added; existing IDs are preserved.

        Args:
            tokens: Token strings to register.
        """
        for token in tokens:
            if token not in self.vocab:
                self.vocab.append(token)
                self.stoi[token] = len(self.vocab) - 1
                self.itos[len(self.vocab) - 1] = token

    def save(self, path: str) -> None:
        """Save vocabulary to a file with tab-separated index and token per line."""
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, 'vocab.json'), 'w', encoding='utf-8') as f:
            for i, token in enumerate(self.vocab):
                f.write(f'{i}\t{token}\n')

    @classmethod
    def load(cls, path: str) -> 'Tokenizer':
        """Load a tokenizer whose vocabulary was written via :meth:`save`.

        Args:
            path: Directory containing the ``vocab.json`` file.

        Returns:
            A new ``Tokenizer`` instance with the loaded vocabulary.
        """
        tokenizer = cls()
        with open(os.path.join(path, 'vocab.json'), 'r', encoding='utf-8') as f:
            for line in f:
                i, token = line.strip().split('\t')
                tokenizer.vocab.append(token)
                tokenizer.stoi[token] = int(i)
                tokenizer.itos[int(i)] = token
        return tokenizer
