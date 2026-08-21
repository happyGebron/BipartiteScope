"""Validated, label-free configuration for build and query operations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AffinityConfig:
    """Configuration for the structure-attribute higher-order affinity graph."""

    beta: float = 0.5
    restart: float = 0.3
    steps: int = 3
    top_k: int = 64

    def __post_init__(self) -> None:
        if not 0.0 <= self.beta <= 1.0:
            raise ValueError("beta must be in [0, 1]")
        if not 0.0 < self.restart < 1.0:
            raise ValueError("restart must be in (0, 1)")
        if self.steps < 1 or self.top_k < 1:
            raise ValueError("steps and top_k must be positive")


@dataclass(frozen=True, slots=True)
class EncoderConfig:
    hidden_dim: int = 64
    layers: int = 2
    latent_groups: int = 8
    epochs: int = 100
    learning_rate: float = 0.01
    lambda_b: float = 0.5
    lambda_o: float = 0.1
    seed: int = 7

    def __post_init__(self) -> None:
        if min(self.hidden_dim, self.layers, self.latent_groups, self.epochs) < 1:
            raise ValueError("encoder dimensions and epochs must be positive")
        if self.learning_rate <= 0 or self.lambda_b < 0 or self.lambda_o < 0:
            raise ValueError("learning rate must be positive and loss weights nonnegative")


@dataclass(frozen=True, slots=True)
class BuildConfig:
    affinity: AffinityConfig = AffinityConfig()
    encoder: EncoderConfig = EncoderConfig()
    semantic_recall_budget: int = 100

    def __post_init__(self) -> None:
        if self.semantic_recall_budget < 1:
            raise ValueError("semantic_recall_budget must be positive")


@dataclass(frozen=True, slots=True)
class QueryConfig:
    size_budget: int = 20
    rho: float = 0.5
    eta_s: float = 0.65
    delta_0: float = 0.02
    semantic_recall_budget: int | None = None

    def __post_init__(self) -> None:
        if self.size_budget < 1:
            raise ValueError("size_budget must be positive")
        if not 0.0 <= self.rho <= 1.0 or not 0.0 <= self.eta_s <= 1.0:
            raise ValueError("rho and eta_s must be in [0, 1]")
        if self.delta_0 < 0:
            raise ValueError("delta_0 must be nonnegative")
        if self.semantic_recall_budget is not None and self.semantic_recall_budget < 1:
            raise ValueError("semantic_recall_budget must be positive when supplied")
