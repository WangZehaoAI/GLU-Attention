import torch
from torch import Tensor
import torch.nn as nn


class Transformer(nn.Module):
    def __init__(
        self,
        nlayer: int,
        dim_model: int,
        num_head: int,
        glu_attn,
        dropout=0.0,
        rope=False,
        max_seq_len: int = 4096,
    ) -> None:
        super().__init__()
        self.rope = (
            RotaryPositionalEmbeddings(dim_model // num_head, max_seq_len)
            if rope
            else None
        )
        self.pe = PositionalEncoding(dim_model, max_seq_len) if not rope else None
        self.layers = nn.ModuleList(
            [
                TransformerLayer(dim_model, num_head, dropout, self.rope, glu_attn)
                for i in range(nlayer)
            ]
        )

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        if self.pe is not None:
            x = self.pe(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x


class TransformerLayer(nn.Module):
    def __init__(
        self,
        dim_model: int,
        num_head: int,
        dropout: float,
        rope: nn.Module,
        glu_attn,
    ) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(dim_model)
        self.attn = MultiHeadSelfAttention(dim_model, num_head, dropout, rope, glu_attn)
        self.ffn_norm = nn.LayerNorm(dim_model)
        self.ffn = GLUFeedForward(dim_model, dropout=dropout)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        x = x + self.attn(self.attn_norm(x), mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


def gated_linear_unit(x: Tensor) -> Tensor:
    gated, gate = x.chunk(2, dim=-1)
    return gated * nn.functional.silu(gate)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        dim_model: int,
        num_head: int,
        dropout: float,
        rope: nn.Module,
        glu: bool,
    ) -> None:
        super().__init__()
        self.num_head = num_head
        self.dim_qk = dim_model // num_head
        self.glu_v = glu
        if glu:
            self.dim_v = dim_model * 2 // num_head // 3
        else:
            self.dim_v = dim_model // num_head
        self.wq = nn.Linear(dim_model, self.dim_qk * num_head)
        self.wk = nn.Linear(dim_model, self.dim_qk * num_head)
        if glu:
            self.wv = nn.Linear(dim_model, self.dim_v * num_head * 2)
        else:
            self.wv = nn.Linear(dim_model, self.dim_v * num_head)
        self.wo = nn.Linear(self.dim_v * num_head, dim_model)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.scale = self.dim_qk**-0.5
        self.rope = rope

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor = None) -> Tensor:
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        if self.glu_v:
            v = gated_linear_unit(v)
        # q,k,v: (b,s,d) -> (b,s,h,d)
        q = q.unflatten(-1, (self.num_head, -1))
        k = k.unflatten(-1, (self.num_head, -1))
        v = v.unflatten(-1, (self.num_head, -1))
        if self.rope is not None:
            q = self.rope(q)
            k = self.rope(k)
        # q,k,v: (b,s,h,d) -> (b,h,s,d)
        q = q.transpose(-2, -3)
        k = k.transpose(-2, -3)
        v = v.transpose(-2, -3)
        score: Tensor = q @ k.mT * self.scale
        if mask is not None:
            if mask.dtype == torch.bool:
                score.masked_fill_(mask, float("-inf"))
            else:
                score += mask
        o = score.softmax(-1) @ v
        result = self.wo(o.transpose(-2, -3).flatten(-2))
        return result if self.dropout is None else self.dropout(result)


class MultiHeadSelfAttention(MultiHeadAttention):
    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        return super().forward(x, x, x, mask)


class GLUFeedForward(nn.Module):
    def __init__(self, dim_model: int, dim_hidden=0, dropout=0.0):
        super().__init__()
        if dim_hidden == 0:
            dim_hidden = dim_model * 8 // 3
        self.linear1 = nn.Linear(dim_model, dim_hidden * 2)
        self.linear2 = nn.Linear(dim_hidden, dim_model)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear1(x)
        x = gated_linear_unit(x)
        x = self.linear2(x)
        return x if self.dropout is None else self.dropout(x)


class PositionalEncoding(nn.Module):
    def __init__(
        self,
        dim,
        max_seq_len: int = 4096,
        base: int = 10_000,
    ):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_seq_len, dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        theta = base ** -(torch.arange(0, dim, 2).float() / dim)
        pe[:, 0::2] = torch.sin(position * theta)
        pe[:, 1::2] = torch.cos(position * theta)

        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(-2), :]


class RotaryPositionalEmbeddings(nn.Module):

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 4096,
        base: int = 10_000,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        self.rope_init()

    def rope_init(self):
        theta = self.base ** -(torch.arange(0, self.dim, 2).float() / self.dim)

        self.register_buffer("theta", theta, persistent=False)
        self.build_rope_cache(self.max_seq_len)

    def build_rope_cache(self, max_seq_len: int = 4096) -> None:
        seq_idx = torch.arange(
            max_seq_len, dtype=self.theta.dtype, device=self.theta.device
        )
        idx_theta = torch.einsum("i, j -> ij", seq_idx, self.theta).float()
        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
        self.register_buffer("cache", cache, persistent=False)

    def forward(
        self, x: torch.Tensor, *, input_pos: torch.Tensor = None
    ) -> torch.Tensor:
        seq_len = x.size(1)
        rope_cache = (
            self.cache[:seq_len] if input_pos is None else self.cache[input_pos]
        )
        xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
        rope_cache = rope_cache.view(-1, xshaped.size(1), 1, xshaped.size(3), 2)

        x_out = torch.stack(
            [
                xshaped[..., 0] * rope_cache[..., 0]
                - xshaped[..., 1] * rope_cache[..., 1],
                xshaped[..., 1] * rope_cache[..., 0]
                + xshaped[..., 0] * rope_cache[..., 1],
            ],
            -1,
        )
        x_out = x_out.flatten(3)
        return x_out.type_as(x)
