"""Differentiable 9-d intent game refiner.

The refiner targets intra-vector consistency in cached LLM intent scores. It
does not call any external service; it only transforms local intent vectors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

DEFAULT_INTENT_NAMES = [
    "public_oriented",
    "emotion_driven",
    "individual_focused",
    "popularize",
    "clout_seeking",
    "conflict_creation",
    "smearing",
    "bias_injection",
    "connection_seeking",
]

DEFAULT_COMPATIBILITY_EDGES: List[Tuple[str, str, float]] = [
    ("public_oriented", "popularize", 0.70),
    ("public_oriented", "connection_seeking", 0.25),
    ("emotion_driven", "individual_focused", 0.55),
    ("emotion_driven", "popularize", 0.35),
    ("emotion_driven", "clout_seeking", 0.35),
    ("individual_focused", "connection_seeking", 0.30),
    ("popularize", "clout_seeking", 0.60),
    ("conflict_creation", "smearing", 0.70),
    ("conflict_creation", "bias_injection", 0.55),
    ("smearing", "bias_injection", 0.65),
]

DEFAULT_CONFLICT_EDGES: List[Tuple[str, str, float]] = [
    ("public_oriented", "conflict_creation", 0.55),
    ("public_oriented", "smearing", 0.70),
    ("public_oriented", "bias_injection", 0.60),
    ("public_oriented", "clout_seeking", 0.20),
    ("connection_seeking", "smearing", 0.20),
]


@dataclass
class IntentGameConfig:
    intent_names: Sequence[str] = field(default_factory=lambda: list(DEFAULT_INTENT_NAMES))
    external_weight: float = 1.0
    anchor_weight: float = 12.0
    compatibility_weight: float = 0.05
    conflict_weight: float = 2.5
    sparsity_weight: float = 0.08
    step_size: float = 0.12
    damping: float = 0.35
    temperature: float = 1.0
    iterations: int = 4
    eps: float = 1e-5


def _edge_matrix(
    intent_names: Sequence[str],
    edges: Iterable[Tuple[str, str, float]],
) -> torch.Tensor:
    name_to_idx = {name: idx for idx, name in enumerate(intent_names)}
    n = len(intent_names)
    matrix = torch.zeros(n, n, dtype=torch.float32)
    for left, right, weight in edges:
        if left not in name_to_idx or right not in name_to_idx:
            raise ValueError(f"Unknown intent edge: {left!r}, {right!r}")
        i = name_to_idx[left]
        j = name_to_idx[right]
        matrix[i, j] = float(weight)
        matrix[j, i] = float(weight)
    return matrix


def safe_logit(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x = x.clamp(eps, 1.0 - eps)
    return torch.log(x) - torch.log1p(-x)


class DifferentiableIntentGameRefiner(nn.Module):
    """Potential-game-style fixed-point refiner for explicit intent scores."""

    def __init__(
        self,
        config: Optional[IntentGameConfig] = None,
        compatibility_edges: Optional[Iterable[Tuple[str, str, float]]] = None,
        conflict_edges: Optional[Iterable[Tuple[str, str, float]]] = None,
    ):
        super().__init__()
        self.config = config or IntentGameConfig()
        self.intent_names = list(self.config.intent_names)
        self.register_buffer(
            "compatibility_matrix",
            _edge_matrix(self.intent_names, compatibility_edges or DEFAULT_COMPATIBILITY_EDGES),
        )
        self.register_buffer(
            "conflict_matrix",
            _edge_matrix(self.intent_names, conflict_edges or DEFAULT_CONFLICT_EDGES),
        )

    @property
    def num_intents(self) -> int:
        return len(self.intent_names)

    def interaction_scores(self, activations: torch.Tensor) -> Dict[str, torch.Tensor]:
        compatibility = 0.5 * torch.sum(
            activations * torch.matmul(activations, self.compatibility_matrix),
            dim=-1,
        )
        conflict = 0.5 * torch.sum(
            activations * torch.matmul(activations, self.conflict_matrix),
            dim=-1,
        )
        return {"compatibility": compatibility, "conflict": conflict}

    def potential(self, raw_intents: torch.Tensor, activations: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        interactions = self.interaction_scores(activations)
        external = cfg.external_weight * torch.sum(raw_intents * activations, dim=-1)
        anchor = cfg.anchor_weight * torch.sum((activations - raw_intents) ** 2, dim=-1)
        sparsity = cfg.sparsity_weight * torch.sum(activations, dim=-1)
        return (
            external
            + cfg.compatibility_weight * interactions["compatibility"]
            - cfg.conflict_weight * interactions["conflict"]
            - anchor
            - sparsity
        )

    def utility_gradient(self, raw_intents: torch.Tensor, activations: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        compatibility = cfg.compatibility_weight * torch.matmul(
            activations, self.compatibility_matrix
        )
        conflict = cfg.conflict_weight * torch.matmul(activations, self.conflict_matrix)
        anchor = 2.0 * cfg.anchor_weight * (activations - raw_intents)
        return (
            cfg.external_weight * raw_intents
            + compatibility
            - conflict
            - anchor
            - cfg.sparsity_weight
        )

    def forward(self, raw_intents: torch.Tensor, return_history: bool = False) -> Dict[str, torch.Tensor]:
        if raw_intents.size(-1) != self.num_intents:
            raise ValueError(f"Expected {self.num_intents} dimensions, got {raw_intents.size(-1)}")

        cfg = self.config
        raw = raw_intents.float().clamp(cfg.eps, 1.0 - cfg.eps)
        state = raw.clone()
        potentials = [self.potential(raw, state)]

        for _ in range(cfg.iterations):
            utility = self.utility_gradient(raw, state)
            proposal = torch.sigmoid(
                safe_logit(state, cfg.eps) + cfg.step_size * utility / cfg.temperature
            )
            state = ((1.0 - cfg.damping) * state + cfg.damping * proposal).clamp(
                cfg.eps, 1.0 - cfg.eps
            )
            if return_history:
                potentials.append(self.potential(raw, state))

        before = self.interaction_scores(raw)
        after = self.interaction_scores(state)
        result: Dict[str, torch.Tensor] = {
            "refined_intents": state,
            "potential_before": potentials[0],
            "potential_after": self.potential(raw, state),
            "compatibility_before": before["compatibility"],
            "compatibility_after": after["compatibility"],
            "conflict_before": before["conflict"],
            "conflict_after": after["conflict"],
            "l1_shift": torch.mean(torch.abs(state - raw), dim=-1),
            "l2_shift": torch.linalg.vector_norm(state - raw, ord=2, dim=-1),
        }
        if return_history:
            result["potential_history"] = torch.stack(potentials, dim=0)
        return result

    def refine(self, raw_intents: torch.Tensor) -> torch.Tensor:
        return self.forward(raw_intents)["refined_intents"]
