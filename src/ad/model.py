import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class Config:
    num_arms: int = 10
    embedding_dim: int = 64
    n_filters: int = 64
    hidden_dim: int = 512
    num_layers: int = 4
    num_heads: int = 4
    seq_len: int = 25
    stretch_factor: int = 4
    attention_dropout: float = 0.5
    residual_dropout: float = 0.1
    embedding_dropout: float = 0.3


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2) * (-math.log(10000.0) / hidden_dim)
        )
        pe = torch.zeros(1, max_len, hidden_dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pos_emb", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [batch_size, seq_len, embedding_dim]
        x = x + self.pos_emb[:, : x.size(1)]
        return x


class TransformerBlock(nn.Module):
    def __init__(
        self,
        seq_len: int,
        hidden_dim: int,
        num_heads: int,
        attention_dropout: float,
        residual_dropout: float,
        stretch_factor: int = 4,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(residual_dropout)

        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads, attention_dropout, batch_first=True
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, stretch_factor * hidden_dim),
            nn.GELU(),
            nn.Linear(stretch_factor * hidden_dim, hidden_dim),
            nn.Dropout(residual_dropout),
        )
        # True value indicates that the corresponding position is not allowed to attend
        self.register_buffer(
            "causal_mask", ~torch.tril(torch.ones(seq_len, seq_len)).to(bool)
        )
        self.seq_len = seq_len

    # [batch_size, seq_len, emb_dim] -> [batch_size, seq_len, emb_dim]
    def forward(
        self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        causal_mask = self.causal_mask[: x.shape[1], : x.shape[1]]

        norm_x = self.norm1(x)
        attention_out = self.attention(
            query=norm_x,
            key=norm_x,
            value=norm_x,
            attn_mask=causal_mask,
            key_padding_mask=padding_mask,
            need_weights=False,
        )[0]
        # by default pytorch attention does not use dropout
        # after final attention weights projection, while minGPT does:
        # https://github.com/karpathy/minGPT/blob/7218bcfa527c65f164de791099de715b81a95106/mingpt/model.py#L70 # noqa
        x = x + self.drop(attention_out)
        x = x + self.mlp(self.norm2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        
        self.seq_len = config.seq_len
        self.embedding_dim = config.embedding_dim
        self.hidden_dim = config.hidden_dim
        self.num_actions = config.num_arms

        self.pos_emb = PositionalEncoding(
            hidden_dim=self.embedding_dim, max_len=2 * self.seq_len
        )
        self.emb_drop = nn.Dropout(config.embedding_dropout)

        self.action_emb = nn.Embedding(self.num_actions, self.embedding_dim)
        self.reward_emb = nn.Embedding(2, self.embedding_dim)

        self.emb2hid = nn.Linear(self.embedding_dim, self.hidden_dim)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    seq_len=2 * self.seq_len,
                    hidden_dim=self.hidden_dim,
                    num_heads=config.num_heads,
                    stretch_factor=config.stretch_factor,
                    attention_dropout=config.attention_dropout,
                    residual_dropout=config.residual_dropout,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.action_head = nn.Linear(self.hidden_dim, self.num_actions)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(
        self,
        actions: torch.Tensor,  # [batch_size, seq_len]
        rewards: torch.Tensor,  # [batch_size, seq_len]
        padding_mask: Optional[torch.Tensor] = None,  # [batch_size, seq_len],
    ) -> torch.FloatTensor:
        
        batch_size, seq_len = rewards.shape[0], rewards.shape[1]

        act_emb = self.action_emb(actions)
        rew_emb = self.reward_emb(rewards)

        assert act_emb.shape == rew_emb.shape
        # [batch_size, 2 * seq_len, emb_dim], (a_0, r_0, a_1, r_1, ...)
        sequence = (
            torch.stack([act_emb, rew_emb], dim=1)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, 2 * seq_len, self.embedding_dim)
        )
        sequence = self.pos_emb(sequence)
        sequence = self.emb2hid(sequence)

        if padding_mask is not None:
            # [batch_size, 2 * seq_len], stack mask identically to fit the sequence
            padding_mask = (
                torch.stack([padding_mask, padding_mask], dim=1)
                .permute(0, 2, 1)
                .reshape(batch_size, 2 * seq_len)
            )

        out = self.emb_drop(sequence)
        for block in self.blocks:
            out = block(out, padding_mask=padding_mask)

        # [batch_size, seq_len, num_actions]
        # predict actions only from state embeddings
        out = self.action_head(out[:, 0::2])

        return out