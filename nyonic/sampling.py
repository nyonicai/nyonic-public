"""The Sampling Strategies."""
from __future__ import annotations

import torch
from torch import Tensor
from typing import Callable

from model import GPTModel


def sample(
    logits: Tensor,
    strategy: str = "vanilla",
    temperature: float = 1.0,
    top_p: float = None,
    top_k: int = None,
) -> Tensor:
    """Sample the next token from logits.

    Args:
        logits: Tensor (batch_size, vocab_size). The logits tensor over vocabulary.
        strategy: str, defaults to "vanilla". The value is in choices ("vanilla",
        "greedy", "top_k", "top_p").
        temperature: float, defaults to 1.0. The value used to modulate the next token
        probabilities.
        top_p: float. The cumulative probability cutoff for top-p sampling.
        top_k: int. The top_k value for top-k sampling.

    Returns:
        Tensor (batch_size, ): The sampled ids based on the probability distribution .

    Examples:
        >>> # probs [0.0321, 0.0871, 0.2369, 0.6439], [0.1840, 0.2033, 0.2033, 0.4094]
        >>> logits = torch.tensor([[1., 2., 3., 4.], [0.0, 0.1, 0.1, 0.8]])
        >>> sample(logits, strategy="greedy")
        tensor([3, 3])
        >>> sample(logits, strategy="top_p", temperature=1., top_p=0.8)
        tensor([3, 3])
        >>> sample(logits, strategy="top_k", temperature=1., top_k=2)
        tensor([3, 3])
    """
    assert (
        len(logits.shape) == 2
    ), f"Input logits shape {logits.shape} must be  (batch_size, vocab_size)"

    if strategy == "greedy":
        next_token = torch.argmax(logits, dim=-1)
    elif strategy == "top_p":
        assert temperature > 0.0, "temperature should be a positive float value."
        assert top_p is not None, "top_p should be set in gen_conf.sampling."
        probs = torch.softmax(logits / temperature, dim=-1)
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        mask = probs_sum - probs_sort > top_p
        probs_sort[mask] = 0.0
        # Multinomial will normalize the input to sum to 1 along the last dimension.
        next_token = torch.multinomial(probs_sort, num_samples=1)
        next_token = probs_idx[range(probs_idx.shape[0]), next_token[:, 0]]
    elif strategy == "top_k":
        assert temperature > 0.0, "temperature should be a positive float value."
        assert top_k is not None, "top_k should be set in gen_conf.sampling."
        probs = torch.softmax(logits / temperature, dim=-1)
        top_probs, top_indices = torch.topk(probs, k=top_k)
        # Multinomial will normalize the input to sum to 1 along the last dimension.
        next_token = torch.multinomial(top_probs, num_samples=1)
        next_token = top_indices[range(top_indices.shape[0]), next_token[:, 0]]
    elif strategy == "vanilla":
        assert temperature > 0.0, "temperature should be a positive float value."
        probs = torch.softmax(logits / temperature, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
    else:
        raise NotImplementedError(f"{strategy} not yet implemented!")
    return next_token


@torch.no_grad()
def generate_tokens(
    model: GPTModel,
    tokens: Tensor,
    min_prompt_len: int,
    pad_id: int,
    sampler: Callable[[Tensor], Tensor],
) -> (Tensor, Tensor | None):
    """
    Generate text completion from prompt tokens using sampling.

    Args:
        model: LightningModel. It owns forward function to generate logits over vocab.
        tokens: Tensor. Input tokens tensor.
        min_prompt_len: int. Minimum prompt length in a batch.
        pad_id: int. Pad token id for padding.
        sampler: Callable[[Tensor], Tensor],

    Returns:
        A tuple:
        - tokens: Tensor (batch_size, total_len).
    """
    model.eval()
    tokens = torch.clone(tokens)
    prev_pos = 0
    input_text_mask = tokens != pad_id
    for cur_pos in range(min_prompt_len, tokens.size(1)):
        # it needs to start from 0 because there is no cache here
        # logits shape: (batch_size, cur_pos-prev_pos, vocab_size)
        logits = model.forward(tokens[:, 0:cur_pos])[:, prev_pos:cur_pos, :]
        next_token = sampler(logits[:, -1, :])

        # Replace tokens if generated, otherwise keep prompt tokens.
        next_token = next_token.reshape(-1)
        next_token = torch.where(
            input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
        )
        tokens[:, cur_pos] = next_token
        prev_pos = cur_pos
    return tokens
