import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.models.vision_transformer import Mlp
from typing import Optional, Tuple
from einops import einsum, rearrange, repeat
from hmr4d.network.base_arch.embeddings.rotary_embedding import ROPE


class RoPEAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1, attention_impl="dense", attention_chunk_size=128):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.attention_impl = attention_impl
        self.attention_chunk_size = attention_chunk_size

        self.rope = ROPE(self.head_dim, max_seq_len=4096)

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def _local_attention(self, xq, xk, xv, window_size, key_padding_mask):
        """Evaluate the existing shifted local window without an L x L score tensor."""
        B, _, L, _ = xq.shape
        outputs = []
        half_window = window_size // 2
        for query_start in range(0, L, self.attention_chunk_size):
            query_end = min(L, query_start + self.attention_chunk_size)
            query_ids = torch.arange(query_start, query_end, device=xq.device)
            starts = torch.clamp(query_ids - half_window, min=0)
            ends = torch.clamp(query_ids + half_window, max=L)
            ends = torch.clamp(ends, min=window_size)
            starts = torch.clamp(starts, max=L - window_size)
            key_start = int(starts[0])
            key_end = int(ends[-1])

            score = einsum(
                xq[:, :, query_start:query_end],
                xk[:, :, key_start:key_end],
                "b n i c, b n j c -> b n i j",
            ) / math.sqrt(self.head_dim)
            key_ids = torch.arange(key_start, key_end, device=xq.device)
            local_mask = (key_ids[None] < starts[:, None]) | (key_ids[None] >= ends[:, None])
            score = score.masked_fill(local_mask[None, None], float("-inf"))
            if key_padding_mask is not None:
                padding = key_padding_mask[:, None, None, key_start:key_end]
                score = score.masked_fill(padding, float("-inf"))
            score = torch.softmax(score, dim=-1)
            score = self.dropout(score)
            output = einsum(
                score,
                xv[:, :, key_start:key_end],
                "b n i j, b n j c -> b n i c",
            )
            outputs.append(output)
        return torch.cat(outputs, dim=2)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        # x: (B, L, C)
        # attn_mask: (L, L)
        # key_padding_mask: (B, L)
        B, L, _ = x.shape
        xq, xk, xv = self.query(x), self.key(x), self.value(x)

        xq = xq.reshape(B, L, self.num_heads, -1).transpose(1, 2)
        xk = xk.reshape(B, L, self.num_heads, -1).transpose(1, 2)
        xv = xv.reshape(B, L, self.num_heads, -1).transpose(1, 2)

        xq = self.rope.rotate_queries_or_keys(xq)  # B, N, L, C
        xk = self.rope.rotate_queries_or_keys(xk)  # B, N, L, C

        if isinstance(attn_mask, tuple) and attn_mask[0] == "local":
            output = self._local_attention(xq, xk, xv, attn_mask[1], key_padding_mask)
        else:
            attn_score = einsum(xq, xk, "b n i c, b n j c -> b n i j") / math.sqrt(self.head_dim)
            if attn_mask is not None:
                attn_mask = attn_mask.reshape(1, 1, L, L).expand(B, self.num_heads, -1, -1)
                attn_score = attn_score.masked_fill(attn_mask, float("-inf"))
            if key_padding_mask is not None:
                padding = key_padding_mask.reshape(B, 1, 1, L).expand(-1, self.num_heads, L, -1)
                attn_score = attn_score.masked_fill(padding, float("-inf"))

            attn_score = torch.softmax(attn_score, dim=-1)
            attn_score = self.dropout(attn_score)
            output = einsum(attn_score, xv, "b n i j, b n j c -> b n i c")  # B, N, L, C
        output = output.transpose(1, 2).reshape(B, L, -1)  # B, L, C
        output = self.proj(output)  # B, L, C
        return output


class EncoderRoPEBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        dropout=0.1,
        attention_impl="dense",
        attention_chunk_size=128,
        **block_kwargs,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        self.attn = RoPEAttention(
            hidden_size,
            num_heads,
            dropout,
            attention_impl=attention_impl,
            attention_chunk_size=attention_chunk_size,
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=dropout)

        self.gate_msa = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.gate_mlp = nn.Parameter(torch.zeros(1, 1, hidden_size))

        # Zero-out adaLN modulation layers
        nn.init.constant_(self.gate_msa, 0)
        nn.init.constant_(self.gate_mlp, 0)

    def forward(self, x, attn_mask=None, tgt_key_padding_mask=None):
        x = x + self.gate_msa * self._sa_block(
            self.norm1(x), attn_mask=attn_mask, key_padding_mask=tgt_key_padding_mask
        )
        x = x + self.gate_mlp * self.mlp(self.norm2(x))
        return x

    def _sa_block(self, x, attn_mask=None, key_padding_mask=None):
        # x: (B, L, C)
        x = self.attn(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        return x
