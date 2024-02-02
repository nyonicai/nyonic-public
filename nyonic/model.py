"""The model definition for the Lightning model to train GPT models."""
from __future__ import annotations

from typing import Optional
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor


class LearnablePositionalEmbeddings(nn.Module):
    """Learnable positional encoder.

    Instead of using the Sine-Cosine fixed embeddings, we attribute a random embedding
    vector with each position token.

    This module also handles  stuffing, where the position embeddings are not
    purely increasing by on for each toeken position.
    """

    def __init__(self, model_args: DictConfig) -> None:
        """Initialize the positional encoder.

        Args:
            model_args: DictConfig containing the keys `vocab_size`, `d_embed`,
                        'context_len` and `dropout`.
        """
        super().__init__()

        self.dropout = nn.Dropout(model_args.dropout)
        self.register_buffer(
            "_token_pos", torch.arange(model_args.context_len).long().reshape(1, -1)
        )
        self.token_value_embedding = nn.Embedding(
            model_args.vocab_size, model_args.d_embed
        )
        self.token_position_embedding = nn.Embedding(
            model_args.context_len, model_args.d_embed
        )
        self._init_weights()

    def _get_position_emb(
        self, x: torch.Tensor, pos_idx: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Op to get a custom position embedding if their indices are provided.

        Args:
            x: tensor of the input token IDs.
            pos_idx: (optional) tensor of the position ids belonging to the tokens in x.
        """
        if pos_idx is None:
            pos_emb = self.token_position_embedding(
                self._token_pos.repeat(x.size(0), 1)[:, : x.size(1)]
            )
        else:
            assert pos_idx.shape == x.shape, (
                "Position indices shape and input shape do not line up."
                f" PosIdx shape: {pos_idx.shape}; Input shape: {x.shape}."
            )
            pos_emb = self.token_position_embedding(pos_idx)

        return pos_emb

    def forward(
        self, x: torch.Tensor, pos_idx: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward the input x through the embedding.

        Args:
            x: tensor of the input token IDs.
            pos_idx: (optional) tensor of the position ids belonging to the tokens in x.
        """
        N, T = x.shape
        out = self._get_position_emb(x, pos_idx)
        out = out + self.token_value_embedding(x)
        out = out.reshape(N, T, -1)
        return self.dropout(out)

    def _init_weights(self) -> None:
        """Initialize the embedding weights."""
        nn.init.normal_(self.token_value_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.token_position_embedding.weight, mean=0.0, std=0.02)


def get_mask(seq_len: int) -> Tensor:
    """Get mask matrix for Transformer.

    Returns:
        mask: torch.Tensor, (seq_len, seq_len). It is either a boolean matrix
            or a float matrix. Here we use a float matrix.
    """
    mask = torch.full((seq_len, seq_len), float("-inf"))
    mask = torch.triu(mask, diagonal=1).float()
    return mask


class GPTModel(nn.Module):
    """The GPT model definition that is parameterized by config."""

    def __init__(self, model_args: DictConfig) -> None:
        """Initialize the model."""
        super().__init__()

        self.model_args = model_args
        self.model_type = model_args.model_type
        self.pos_embed_type = model_args.pos_embed_type
        self.enable_final_norm = model_args.enable_final_norm
        self.enable_is_causal = model_args.enable_is_causal
        self.bias = model_args.bias

        if self.pos_embed_type == "learnable_pe":
            self.embedding = LearnablePositionalEmbeddings(model_args)
        else:
            raise ValueError(
                f"Unsupported positional embedding type: {self.pos_embed_type}"
            )

        final_norm = None
        if self.enable_final_norm:
            final_norm = nn.LayerNorm(model_args.d_embed)

        self.transformer_encoder: nn.Module
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_args.d_embed,
            nhead=model_args.n_heads,
            dim_feedforward=model_args.d_ff,
            dropout=model_args.dropout,
            activation=model_args.activation,
            layer_norm_eps=1e-5,
            norm_first=True,
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, model_args.n_layers, norm=final_norm
        )

        self.linear = nn.Linear(
            model_args.d_embed, model_args.vocab_size, bias=self.bias
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize the model weights."""
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)
        nn.init.normal_(self.linear.weight, mean=0.0, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[Tensor] = None,
        stuffing_mask: Optional[Tensor] = None,
        pos_idx: Optional[Tensor] = None,
    ) -> Tensor:
        """Perform a forward pass for both training and inference.

        Args:
            x: torch.Tensor, (batch_size, context_len). The input tensor.
            attn_mask: optional bool-valued tensor of the attention mask
            stuffing_mask: (optional) tensor indicating the sample indices of the
                stuffed context vector.
            pos_idx: (optional) int-tensor containing the position indices of the
                stuffed context vector.

        Returns:
            logits: torch.Tensor, (batch_size, context_len, hidden_dim).
        """
        if pos_idx is not None:
            pos_idx.to(x.device, dtype=torch.long)
        if stuffing_mask is not None:
            stuffing_mask.to(device=x.device, dtype=torch.long)
        x = self.embedding(x, pos_idx)

        if attn_mask is None:
            attn_mask = get_mask(x.size(1)).to(device=x.device, dtype=x.dtype)

        x = self.transformer_encoder(
            x,
            attn_mask=attn_mask,
            stuffing_mask=stuffing_mask,
            is_causal=self.enable_is_causal,
        )

        x = self.linear(x)
        return x
