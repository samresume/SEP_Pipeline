from __future__ import annotations

import copy
import random
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from model import CausalTimeSeriesDenoiser


@dataclass(frozen=True)
class TSGenerationConfig:
    """Training, model, and diffusion parameters."""

    epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 2e-4
    diffusion_steps: int = 100
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    model_dim: int = 128
    num_layers: int = 4
    num_heads: int = 4
    dropout: float = 0.1
    ema_decay: float = 0.995
    gradient_clip: float | None = 1.0
    seed: int = 42
    device: str = "auto"
    verbose: bool = True

    @classmethod
    def from_value(
        cls, value: Mapping[str, Any] | "TSGenerationConfig" | None
    ) -> "TSGenerationConfig":
        if value is None:
            result = cls()
        elif isinstance(value, cls):
            result = value
        elif isinstance(value, Mapping):
            unknown = set(value) - set(asdict(cls()))
            if unknown:
                raise ValueError(f"unknown parameters: {sorted(unknown)}")
            result = cls(**value)
        else:
            raise TypeError("parameters must be a mapping, TSGenerationConfig, or None")
        result.validate()
        return result

    def validate(self) -> None:
        if self.epochs < 1 or self.batch_size < 1 or self.diffusion_steps < 2:
            raise ValueError("epochs/batch_size must be positive and diffusion_steps >= 2")
        if not 0 < self.beta_start < self.beta_end < 1:
            raise ValueError("betas must satisfy 0 < beta_start < beta_end < 1")
        if self.model_dim % self.num_heads:
            raise ValueError("model_dim must be divisible by num_heads")
        if not 0 <= self.ema_decay < 1:
            raise ValueError("ema_decay must be in [0, 1)")


def _dataset_tensor(dataset: Any) -> torch.Tensor:
    """Materialize common array and PyTorch Dataset inputs as float32 [N,T,F]."""
    if isinstance(dataset, torch.Tensor):
        tensor = dataset.detach().cpu()
    elif isinstance(dataset, np.ndarray):
        tensor = torch.from_numpy(dataset)
    elif isinstance(dataset, Dataset):
        samples = []
        for index in range(len(dataset)):
            sample = dataset[index]
            if isinstance(sample, (tuple, list)):
                sample = sample[0]
            samples.append(torch.as_tensor(sample))
        if not samples:
            raise ValueError("dataset is empty")
        tensor = torch.stack(samples)
    else:
        tensor = torch.as_tensor(dataset)

    tensor = tensor.to(dtype=torch.float32)
    if tensor.ndim != 3:
        raise ValueError(
            "dataset must have shape (num_examples, num_timesteps, num_features)"
        )
    if tensor.shape[0] < 1 or tensor.shape[1] < 1 or tensor.shape[2] < 1:
        raise ValueError("all dataset dimensions must be non-zero")
    if not torch.isfinite(tensor).all():
        raise ValueError("dataset contains NaN or infinite values")
    return tensor.contiguous()


class _Standardizer:
    def fit(self, data: torch.Tensor) -> "_Standardizer":
        self.mean = data.mean(dim=(0, 1), keepdim=True)
        self.std = data.std(dim=(0, 1), keepdim=True, unbiased=False).clamp_min(1e-6)
        return self

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        return (data - self.mean) / self.std

    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        return data * self.std.to(data.device) + self.mean.to(data.device)


class _LinearSchedule(nn.Module):
    def __init__(self, steps: int, beta_start: float, beta_end: float) -> None:
        super().__init__()
        betas = torch.linspace(beta_start, beta_end, steps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bars_previous = torch.cat((torch.ones(1), alpha_bars[:-1]))
        posterior_variance = betas * (1 - alpha_bars_previous) / (1 - alpha_bars)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("posterior_variance", posterior_variance.clamp_min(1e-20))

    @staticmethod
    def extract(values: torch.Tensor, steps: torch.Tensor, shape: torch.Size) -> torch.Tensor:
        return values.gather(0, steps).reshape(steps.shape[0], *((1,) * (len(shape) - 1)))


class TimeSeriesDDPM:
    """Train and sample a causal-attention DDPM."""

    def __init__(self, parameters: Mapping[str, Any] | TSGenerationConfig | None = None):
        self.config = TSGenerationConfig.from_value(parameters)
        self.device = self._resolve_device(self.config.device)
        self.model: CausalTimeSeriesDenoiser | None = None
        self.ema_model: CausalTimeSeriesDenoiser | None = None
        self.scaler: _Standardizer | None = None
        self.schedule: _LinearSchedule | None = None
        self.sequence_length: int | None = None
        self.num_features: int | None = None
        self.loss_history: list[float] = []

    @staticmethod
    def _resolve_device(requested: str) -> torch.device:
        if requested != "auto":
            return torch.device(requested)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _seed_everything(self) -> None:
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)

    @staticmethod
    @torch.no_grad()
    def _ema_update(ema_model: nn.Module, model: nn.Module, decay: float) -> None:
        for ema_parameter, parameter in zip(ema_model.parameters(), model.parameters()):
            ema_parameter.lerp_(parameter, 1 - decay)
        for ema_buffer, buffer in zip(ema_model.buffers(), model.buffers()):
            ema_buffer.copy_(buffer)

    def fit(self, dataset: Any) -> "TimeSeriesDDPM":
        self._seed_everything()
        data = _dataset_tensor(dataset)
        _, self.sequence_length, self.num_features = data.shape
        self.scaler = _Standardizer().fit(data)
        normalized = self.scaler.transform(data)
        loader_generator = torch.Generator().manual_seed(self.config.seed)
        loader = DataLoader(
            TensorDataset(normalized),
            batch_size=self.config.batch_size,
            shuffle=True,
            generator=loader_generator,
        )

        self.model = CausalTimeSeriesDenoiser(
            self.num_features,
            model_dim=self.config.model_dim,
            num_layers=self.config.num_layers,
            num_heads=self.config.num_heads,
            dropout=self.config.dropout,
        ).to(self.device)
        self.ema_model = copy.deepcopy(self.model).eval().requires_grad_(False)
        self.schedule = _LinearSchedule(
            self.config.diffusion_steps,
            self.config.beta_start,
            self.config.beta_end,
        ).to(self.device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate)

        self.loss_history = []
        for epoch in range(self.config.epochs):
            self.model.train()
            losses = []
            for (clean,) in loader:
                clean = clean.to(self.device)
                steps = torch.randint(
                    self.config.diffusion_steps, (clean.shape[0],), device=self.device
                )
                noise = torch.randn_like(clean)
                alpha_bar = self.schedule.extract(
                    self.schedule.alpha_bars, steps, clean.shape
                )
                noisy = alpha_bar.sqrt() * clean + (1 - alpha_bar).sqrt() * noise
                predicted_noise = self.model(noisy, steps)
                loss = torch.nn.functional.mse_loss(predicted_noise, noise)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if self.config.gradient_clip is not None:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                optimizer.step()
                self._ema_update(self.ema_model, self.model, self.config.ema_decay)
                losses.append(loss.detach().item())

            epoch_loss = float(np.mean(losses))
            self.loss_history.append(epoch_loss)
            if self.config.verbose and (
                epoch == 0
                or (epoch + 1) % 100 == 0
                or epoch + 1 == self.config.epochs
            ):
                print(f"epoch {epoch + 1:03d}/{self.config.epochs:03d} loss={epoch_loss:.6f}")
        return self

    @torch.inference_mode()
    def sample(self, num_samples: int) -> np.ndarray:
        if num_samples < 1:
            raise ValueError("num_samples must be positive")
        if any(value is None for value in (self.ema_model, self.schedule, self.scaler)):
            raise RuntimeError("call fit(dataset) before sample(num_samples)")
        assert self.ema_model is not None and self.schedule is not None
        assert self.scaler is not None and self.sequence_length is not None
        assert self.num_features is not None

        model = self.ema_model.eval()
        shape = (num_samples, self.sequence_length, self.num_features)
        series = torch.randn(shape, device=self.device)
        for step in reversed(range(self.config.diffusion_steps)):
            steps = torch.full((num_samples,), step, device=self.device, dtype=torch.long)
            predicted_noise = model(series, steps)
            beta = self.schedule.extract(self.schedule.betas, steps, series.shape)
            alpha = self.schedule.extract(self.schedule.alphas, steps, series.shape)
            alpha_bar = self.schedule.extract(self.schedule.alpha_bars, steps, series.shape)
            mean = (series - beta * predicted_noise / (1 - alpha_bar).sqrt()) / alpha.sqrt()
            if step > 0:
                variance = self.schedule.extract(
                    self.schedule.posterior_variance, steps, series.shape
                )
                series = mean + variance.sqrt() * torch.randn_like(series)
            else:
                series = mean

        return self.scaler.inverse_transform(series).cpu().numpy().astype(np.float32)


def train_and_generate(
    dataset: Any,
    parameters: Mapping[str, Any] | TSGenerationConfig | None,
    num_samples: int,
) -> np.ndarray:
    """Train on ``dataset`` and return ``(num_samples, timesteps, features)`` data."""
    return TimeSeriesDDPM(parameters).fit(dataset).sample(num_samples)
