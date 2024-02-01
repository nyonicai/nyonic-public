"""Tokenizer module.

Implements a wrapper around the sentencepiece tokenizer.
The wrapper ensures storing and loading via fsspec and that
the training happens in a streaming fashion.

In this way we can use a squirrel-based iter stream to train
the tokenizer and store it.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import os
import typing as t
from copy import deepcopy
from itertools import product

import fsspec
import sentencepiece as spm

logger = logging.Logger("tokenizer_logger")


@dataclasses.dataclass
class SpmTokenizerConfig:
    """Dataclass to hold the SPM config.

    Values can be looked up in the underlying SentencePieceTrainer class.

    Args:
        whitespace_tokens_range: upper limit for the number of whitespace tokens.
            If None disablewhitespace tokenization.

        remove_extra_whitespaces:  Removes leading, trailing, and
          duplicate internal whitespace see here for more info
          https://github.com/google/sentencepiece/blob/master/doc/options.md
          We used to have this as True, but it is not ideal for code tokens.
          Value is nor False by default. And this will be deprecated in
          the future.

        split_digits_magnitude: put all numbers with up to this many digits into
            the vocab. E.g. 3 will put all numbers with up to
            3 digits into the vocab (0-999)

        input_sentence_size:maximum size of sentences the trainer loads,
            default 0 = unlimited
    """

    name: str
    model_type: str
    vocab_size: int
    byte_fallback: bool = True
    split_digits: bool = True
    split_digits_magnitude: int = 3
    whitespace_tokens_range: t.Optional[int] = 24
    remove_extra_whitespaces: bool = False
    input_sentence_size: int = 0

    def to_json(self) -> str:
        """Dumps the config to a json string."""
        return json.dumps(dataclasses.asdict(self))

    def from_json(text: str) -> "SpmTokenizerConfig":
        """Loads a `SpmTokenizerConfig` from a json string."""
        return SpmTokenizerConfig(**json.loads(text))


def _get_digit_vocab(n: int = 3) -> list[str]:
    """Generate list of all 0:10**n -1 -digit combinations.

    Args:
      n: Number of digits in each combination.


    """
    digit_vocab = []
    for i in range(1, n + 1):
        digit_vocab += list(map("".join, product("0123456789", repeat=i)))
    return digit_vocab


def _get_whitespace_tokens_vocab(whitespace_tokens_range: int) -> list[str]:
    """Generate list of "▁" combination in range(2,whitespace_tokens_range) .

    Args:
      whitespace_tokens_range: upper limit for the number of whitespace tokens.

    """
    return ["▁" * i for i in range(2, whitespace_tokens_range)]


T = t.TypeVar("T", bound="NyonicTokenizer")


class NyonicTokenizer:
    """Tokenizer utility class around sentencepiece."""

    pad_id: int = 0
    unk_id: int = 1
    bos_id: int = 2
    eos_id: int = 3

    add_dummy_prefix = False
    normalization_rule_name = "identity"
    default_user_defined_symbols = ["\n", "\t", "\r"]

    def __init__(self, location: str, fsspec_kwargs: t.Optional[t.Dict] = None) -> None:
        """Initializes a tokenizer object.

        The loading is lazy. It can be triggered using `load()` or upon first
        invocation of `encode()` or `decode()`.

        Args:
            location (str): path to tokenizer folder containing a config and binary.
            fsspec_kwargs: optional dict for fsspec configuration.
        """
        super().__init__()
        self.location = location
        self.fsspec_kwargs = fsspec_kwargs if fsspec_kwargs else {}
        self.config = None
        self.name = None
        self.tokenizer = None

    def load(self) -> "NyonicTokenizer":
        """Loads the tokenizer config and binary."""
        with fsspec.open(
            os.path.join(self.location, "config.json"), mode="rb", **self.fsspec_kwargs
        ) as src:
            json_string = src.read().decode("utf-8")
            self.config = SpmTokenizerConfig.from_json(json_string)
            self.name: str = self.config.name

        with fsspec.open(
            os.path.join(self.location, "tokenizer.model"),
            mode="rb",
            **self.fsspec_kwargs,
        ) as src:
            self.tokenizer = spm.SentencePieceProcessor(model_proto=src.read())

        return self

    def encode(
        self,
        input: t.List[t.Union[int, t.List[str]]],
        out_type: t.Optional[t.Any] = int,
    ) -> t.List:
        """Encodes the input into the corresponding vocab ids."""
        if self.tokenizer is None:
            self.load()
        assert self.tokenizer is not None, "Tokenizer is not loaded."
        return self.tokenizer.encode(input, out_type)

    def decode(self, input: t.List[t.Union[int, t.List[int]]]) -> t.List[str]:
        """Decodes input ids into the corresponding word pieces or sentences."""
        if self.tokenizer is None:
            self.load()
        assert self.tokenizer is not None, "Tokenizer is not loaded."
        return self.tokenizer.decode(input)

    def decode_single_token_bytes(self, token: int) -> bytes:
        """Decodes a single token id into the corresponding word piece or sentence."""
        return self.tokenizer.id_to_piece(token).encode("utf-8")

    @property
    def vocab(self) -> t.List[str]:
        """Tokenizer vocabulary."""
        return [
            self.tokenizer.id_to_piece(id)
            for id in range(self.tokenizer.get_piece_size())
        ]

    @property
    def n_vocab(self) -> int:
        """Return vocab size."""
        return self.tokenizer.get_piece_size()

    # The type var is needed for inheritance to work properly type hint wise.
    @classmethod
    def get_user_defined_symbols(cls: T, config: SpmTokenizerConfig) -> list[str]:
        """Get the default user defined symbols.

        Args:
            config: tokenizer config dataclass
        Returns:
            list of default user defined symbols as string
        """
        user_defined_symbols = deepcopy(cls.default_user_defined_symbols)
        # if we don't do a deepcopy the cls attribute get change permanently

        if config.whitespace_tokens_range:
            user_defined_symbols += _get_whitespace_tokens_vocab(
                config.whitespace_tokens_range
            )

        user_defined_symbols += _get_digit_vocab(config.split_digits_magnitude)
        return user_defined_symbols
