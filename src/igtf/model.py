"""IGTF model: text MoE encoder + 9-d intent game calibration + fusion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from .intent_game import DifferentiableIntentGameRefiner, IntentGameConfig


@dataclass
class IGTFConfig:
    bert_model: str = "bert-base-uncased"
    hidden_dim: int = 768
    cnn_dim: int = 256
    intent_dim: int = 9
    fusion_dim: int = 256
    dropout: float = 0.3
    num_labels: int = 2
    use_intent_game: bool = True
    game_iterations: int = 4


class TextMoEEncoder(nn.Module):
    """BERT encoder with CLS, mean, CNN, and document-attention experts."""

    def __init__(self, config: IGTFConfig):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(config.bert_model)
        hidden = self.backbone.config.hidden_size
        self.hidden_size = hidden

        self.cnn = nn.ModuleList(
            [nn.Conv1d(hidden, config.cnn_dim, kernel_size=k, padding=k // 2) for k in (2, 3, 4)]
        )
        self.cnn_projection = nn.Linear(config.cnn_dim * 3, hidden)
        self.doc_attention = nn.Linear(hidden, 1)

        self.expert_projection = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(4)])
        self.router = nn.Linear(hidden, 4)
        self.norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(config.dropout)

    def masked_mean(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).float()
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

    def doc_attention_pool(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        scores = self.doc_attention(hidden).squeeze(-1)
        scores = scores.masked_fill(attention_mask == 0, -1e9)
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
        return (hidden * weights).sum(dim=1)

    def cnn_pool(self, hidden: torch.Tensor) -> torch.Tensor:
        x = hidden.transpose(1, 2)
        pooled = []
        for conv in self.cnn:
            z = F.relu(conv(x))
            pooled.append(F.adaptive_max_pool1d(z, 1).squeeze(-1))
        return self.cnn_projection(torch.cat(pooled, dim=-1))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        output = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden = output.last_hidden_state
        cls = hidden[:, 0]
        experts = [
            cls,
            self.masked_mean(hidden, attention_mask),
            self.cnn_pool(hidden),
            self.doc_attention_pool(hidden, attention_mask),
        ]
        projected = torch.stack(
            [proj(expert) for proj, expert in zip(self.expert_projection, experts)],
            dim=1,
        )
        router_weights = torch.softmax(self.router(cls), dim=-1)
        fused = torch.sum(projected * router_weights.unsqueeze(-1), dim=1)
        return {
            "text_feature": self.dropout(self.norm(fused)),
            "router_weights": router_weights,
        }


class IGTFClassifier(nn.Module):
    def __init__(self, config: IGTFConfig):
        super().__init__()
        self.config = config
        self.text_encoder = TextMoEEncoder(config)
        game_cfg = IntentGameConfig(iterations=config.game_iterations)
        self.intent_game = DifferentiableIntentGameRefiner(game_cfg)
        text_dim = self.text_encoder.hidden_size
        self.intent_projection = nn.Sequential(
            nn.Linear(config.intent_dim * 2, config.fusion_dim),
            nn.LayerNorm(config.fusion_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )
        self.text_projection = nn.Sequential(
            nn.Linear(text_dim, config.fusion_dim),
            nn.LayerNorm(config.fusion_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )
        self.classifier = nn.Sequential(
            nn.Linear(config.fusion_dim * 2, config.fusion_dim),
            nn.LayerNorm(config.fusion_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.fusion_dim, config.num_labels),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        intent_vector: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        text_result = self.text_encoder(input_ids, attention_mask)
        raw_intents = intent_vector.float().clamp(0.0, 1.0)
        if self.config.use_intent_game:
            game_result = self.intent_game(raw_intents)
            refined = game_result["refined_intents"]
        else:
            game_result = {}
            refined = raw_intents

        text_z = self.text_projection(text_result["text_feature"])
        intent_z = self.intent_projection(torch.cat([raw_intents, refined], dim=-1))
        logits = self.classifier(torch.cat([text_z, intent_z], dim=-1))

        result: Dict[str, torch.Tensor] = {
            "logits": logits,
            "predictions": torch.argmax(logits, dim=-1),
            "raw_intents": raw_intents,
            "refined_intents": refined,
            "router_weights": text_result["router_weights"],
        }
        result.update(game_result)
        if labels is not None:
            result["loss"] = F.cross_entropy(logits, labels)
        return result
