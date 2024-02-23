# Copyright 2024 nyonic ai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The model definition for the Lightning model to train GPT models."""
from __future__ import annotations

import copy

import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from omegaconf import DictConfig
from torch import Tensor

from xformers.components.positional_embedding import RotaryEmbedding
import xformers.ops as xops

ACTIVATIONS = {"relu": F.relu, "gelu": F.gelu}

from einops import rearrange


def rearrange_view(
    tensor: Tensor | list[Tensor], pattern: str, **axes_lengths
) -> Tensor:
    """Rearrange the axes of a tensor with einops and return a view of the tensor.

    This is a thin wrapper around einops.rearrange but with a view like behaviour.
    hack until it get implemented in einops.
    https://github.com/arogozhnikov/einops/issues/296
    """
    tensor_output = rearrange(tensor, pattern, **axes_lengths)

    if not tensor_output.data_ptr() == tensor.data_ptr():
        del tensor_output
        # here we need to del otherwise if the error is catch
        # we might have memory problem. lets help the GC
        raise RuntimeError(
            "rearrange is not possible in-place use einops.rearange directly"
        )

    return tensor_output


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


class SimpleTokenEmbedding(nn.Embedding):
    """Simple Token Embedding.

    This is a wrapper around nn.Embedding with custom weight initialization.
    """

    def __init__(self, vocab_size: int, d_embed: int) -> None:
        """Initialize the positional encoder."""
        super().__init__(vocab_size, d_embed)

    def forward(self, x: Tensor, _pos_idx: Optional[torch.Tensor] = None) -> Tensor:
        """Perform a forward pass."""
        return super().forward(x)


class NyonicMLP(nn.Module):
    def __init__(
        self, d_embed: int, d_ff: int, dropout: float, bias: bool, activation: str
    ):
        super().__init__()
        self.linear1 = nn.Linear(d_embed, d_ff, bias=bias)
        self.linear2 = nn.Linear(d_ff, d_embed, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.activation = ACTIVATIONS[activation]

    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class NyonicAttention(nn.Module):
    def __init__(
        self,
        d_embed: int,
        num_heads: int,
        dropout: float = 0.0,
        rotary: bool = False,
        bias: bool = False,
        qk_layer_norm: bool = False,
    ) -> None:
        """Initializes the MHA module.

        Args:
            d_embed: integer of the model embedding dimension
            num_heads: number of attention heads.
            dropout: float for the dropout probability.
            rotary: bool indicating if rotary positional embeddings should be used.
            bias: bool indicating if bias should be used.
            qk_layer_norm: bool indicating if layer norm should be applied to the
                query and key projections.
        """
        super().__init__()
        self.embed_dim = d_embed

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = d_embed // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        self.in_proj_weight = nn.Parameter(torch.empty((3 * d_embed, d_embed)))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * d_embed)) if bias else None

        self.out_proj_weight = nn.Parameter(torch.empty((d_embed, d_embed)))
        self.out_proj_bias = nn.Parameter(torch.empty(d_embed)) if bias else None

        self.rotary = rotary
        self.qk_layer_norm = qk_layer_norm

        if self.rotary:
            self.rotary_embedding = RotaryEmbedding(self.head_dim)

        if self.qk_layer_norm:
            self.q_layer_norm = nn.LayerNorm(self.head_dim)
            self.k_layer_norm = nn.LayerNorm(self.head_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Module forward.

        Args:
            x: torch.Tensor of the input.
            attn_mask: (optional) Either bool-valued tensor of the attention mask or an
                AttentionMask enum to indicate the xFormers AttentionBias constructor.
            stuffing_mask: (optional) tensor indicating the sample indices of the
                stuffed context vector.
            is_causal: bool indicating if the attention op is causal. The exact
                effect depends on this module's attn_type.
        """
        B, T, D = x.shape
        attn_bias: xops.AttentionBias = xops.LowerTriangularMask()

        q, k, v = F.linear(x, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)

        q = q.view(q, "B T (N D) -> B T N D", N=self.num_heads)
        k = rearrange_view(k, "B T (N D) -> B T N D", N=self.num_heads)
        v = rearrange_view(v, "B T (N D) -> B T N D", N=self.num_heads)

        if self.qk_layer_norm:
            q = self.q_layer_norm(q)
            k = self.k_layer_norm(k)

        if self.rotary is not None:
            q = rearrange_view(q, "B T N D -> B N T D")
            k = rearrange_view(k, "B T N D -> B N T D")
            # Note: rotary_embedding use -2 as the sequence dimension.
            q, k = self.rotary_embedding(q=q, k=k)
            q = rearrange_view(q, "B N T D -> B T N D")
            k = rearrange_view(k, "B N T D -> B T N D")

        # we cast to the same type as v needed for mix precision training
        q = q.to(v.dtype)
        k = k.to(v.dtype)

        for tensor_to_check in [q, k, v]:
            assert tensor_to_check.dtype in [
                torch.bfloat16,
                torch.float16,
            ], f"Our use of flashattn@v2.3.0 does not support {tensor_to_check.dtype}."

        attn_output = (
            xops.memory_efficient_attention(
                query=q,
                key=k,
                value=v,
                p=self.dropout if self.training else 0.0,
                attn_bias=attn_bias,
                op=xops.MemoryEfficientAttentionFlashAttentionOp,
            )
            .continuew()
            .view(B, T, -1)
        )
        attn_output = F.linear(attn_output, self.out_proj_weight, self.out_proj_bias)
        return attn_output


class NyonicDecoderLayer(nn.Module):
    """Transformer encoder layer."""

    def __init__(
        self,
        d_embed: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "relu",
        norm_first: bool = False,
        rotary: bool = False,
        bias: bool = False,
        qk_layer_norm: bool = False,
    ) -> None:
        """Initializes the encoder layer.

        Args:
            d_embed: int of the model embedding dimension.
            num_heads: int for the number of attention heads.
            d_ff: int of the feed forward dimension.
            dropout: float for the dropout probability. Default: 0.1
            activation: str of the activation function. Default: "relu".
            norm_first: True if pre-normalization should be used in the transformer.
            rotary: bool indicating if rotary positional embeddings should be used.
            bias: bool indicating if bias should be used in every layerss
            qk_layer_norm: bool indicating if layer norm should be applied to the
                query and key vectors before the attention op.
        """
        super().__init__()

        self.bias = bias
        self.norm_first = norm_first

        self.self_attn = NyonicAttention(
            d_embed,
            num_heads,
            dropout=dropout,
            rotary=rotary,
            bias=bias,
            qk_layer_norm=qk_layer_norm,
        )
        self.ffn = NyonicMLP(d_embed, d_ff, dropout, bias, activation)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_embed)
        self.norm2 = nn.LayerNorm(d_embed)

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        """Module forward.

        Args:
            x: torch.Tensor of the input.
        """
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x))
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x))
            x = self.norm2(x + self._ff_block(x))
        return x

    def _sa_block(
        self,
        x: Tensor,
    ) -> Tensor:
        """Attention block forward.

        Args:
            x: torch.Tensor of the input.
        """
        return self.dropout1(self.self_attn(x))

    def _ff_block(self, x: Tensor) -> Tensor:
        """Feed-forward block forward.

        Args:
            x: torch.Tensor of the input.
        """
        return self.dropout2(self.ffn(x))


class NyonicDecoder(nn.Module):
    """Encoder stack module with API for context stuffing."""

    def __init__(
        self,
        decoder_layer: NyonicDecoderLayer,
        num_layers: int,
        norm: Optional[nn.Module] = None,
    ) -> None:
        """Initializes the encoder layer stack.

        Args:
            decoder_layer: encoder layer that will be cloned.
            num_layers: int of the depth of the stack.
            norm: (optional) nn.Module of a normalization op. Defaults to None.
        """
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layer = copy.deepcopy(decoder_layer)
            layers.append(layer)

        self.layers = nn.ModuleList(layers)
        self.norm = norm

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        """Module forward.

        Args:
            x: torch.Tensor of the input.
            attn_mask: optional bool-valued tensor of the attention mask.
            is_causal: bool indicating if the attention op is causal. The exact
                effect depends on this module's attn_type.
        """
        output = x
        for mod in self.layers:
            output = mod(output)

        if self.norm is not None:
            output = self.norm(output)

        return output


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
        self.bias = model_args.bias

        if self.pos_embed_type == "learnable_pe":
            self.embedding = LearnablePositionalEmbeddings(model_args)
        elif self.pos_embed_type == "rotary_pe":
            self.embedding = SimpleTokenEmbedding(
                model_args.vocab_size, model_args.d_embed
            )
        else:
            raise ValueError(
                f"Unsupported positional embedding type: {self.pos_embed_type}"
            )

        final_norm = None
        if self.enable_final_norm:
            final_norm = nn.LayerNorm(model_args.d_embed)

        nyonic_layer: nn.Module = NyonicDecoderLayer(
            d_embed=model_args.d_embed,
            num_heads=model_args.n_heads,
            d_ff=model_args.d_ff,
            dropout=model_args.dropout,
            activation=model_args.activation,
            norm_first=True,
            rotary=self.pos_embed_type == "rotary_pe",
            bias=model_args.bias,
            qk_layer_norm=model_args.qk_layer_norm,
        )

        self.nyonic_decoder = NyonicDecoder(
            nyonic_layer, model_args.n_layers, norm=final_norm
        )

        self.linear = nn.Linear(
            model_args.d_embed, model_args.vocab_size, bias=model_args.bias
        )

    def forward(
        self,
        x: torch.Tensor,
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
        x = self.embedding(x, pos_idx)
        x = self.nyonic_decoder(x)
        x = self.linear(x)
        return x
