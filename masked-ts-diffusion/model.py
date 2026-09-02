from __future__ import annotations

import math

import torch
from torch import nn


def sinusoidal_embedding(values: torch.Tensor, dimension: int) -> torch.Tensor:
    """Embed integer/float positions with the standard sine/cosine encoding."""
    if dimension < 2:
        raise ValueError("embedding dimension must be at least 2")
    half = dimension // 2
    scale = math.log(10_000) / max(half - 1, 1)
    frequencies = torch.exp(
        -scale * torch.arange(half, device=values.device, dtype=torch.float32)
    )
    angles = values.float().unsqueeze(-1) * frequencies
    embedding = torch.cat((angles.sin(), angles.cos()), dim=-1)
    if dimension % 2:
        embedding = torch.nn.functional.pad(embedding, (0, 1))
    return embedding


class CausalAttentionBlock(nn.Module):
    """Pre-norm Transformer block with an explicit future-token mask."""

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        dropout: float,
        feedforward_multiplier: int = 4,
    ) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(model_dim)
        self.attention = nn.MultiheadAttention(
            model_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.feedforward_norm = nn.LayerNorm(model_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(model_dim, feedforward_multiplier * model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_multiplier * model_dim, model_dim),
            nn.Dropout(dropout),
        )

    @staticmethod
    def causal_mask(length: int, device: torch.device) -> torch.Tensor:
        # True entries are blocked by MultiheadAttention.
        return torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self.attention_norm(x)
        mask = self.causal_mask(x.shape[1], x.device)
        attended, _ = self.attention(
            normalized, normalized, normalized, attn_mask=mask, need_weights=False
        )
        x = x + attended
        return x + self.feedforward(self.feedforward_norm(x))


class CausalTimeSeriesDenoiser(nn.Module):
    """Predict DDPM noise for tensors shaped ``(batch, time, features)``."""

    def __init__(
        self,
        num_features: int,
        model_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if model_dim % num_heads:
            raise ValueError("model_dim must be divisible by num_heads")
        self.model_dim = model_dim
        self.input_projection = nn.Linear(num_features, model_dim)
        self.diffusion_projection = nn.Sequential(
            nn.Linear(model_dim, model_dim * 4),
            nn.SiLU(),
            nn.Linear(model_dim * 4, model_dim),
        )
        self.blocks = nn.ModuleList(
            CausalAttentionBlock(model_dim, num_heads, dropout)
            for _ in range(num_layers)
        )
        self.output_norm = nn.LayerNorm(model_dim)
        self.output_projection = nn.Linear(model_dim, num_features)

    def forward(self, noisy_series: torch.Tensor, diffusion_step: torch.Tensor) -> torch.Tensor:
        if noisy_series.ndim != 3:
            raise ValueError("noisy_series must have shape (batch, time, features)")
        batch, length, _ = noisy_series.shape
        if diffusion_step.shape != (batch,):
            raise ValueError("diffusion_step must have shape (batch,)")

        sequence_positions = torch.arange(length, device=noisy_series.device)
        position_embedding = sinusoidal_embedding(
            sequence_positions, self.model_dim
        ).unsqueeze(0)
        step_embedding = self.diffusion_projection(
            sinusoidal_embedding(diffusion_step, self.model_dim)
        ).unsqueeze(1)

        hidden = self.input_projection(noisy_series) + position_embedding + step_embedding
        for block in self.blocks:
            hidden = block(hidden)
        return self.output_projection(self.output_norm(hidden))

