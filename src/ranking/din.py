"""DIN ranker with candidate-aware attention over the user's item history.

Attention is computed against the current candidate item, not by mean-pooling
the history.  Padding index 0 is masked.  Item category/price buckets for
history items come from registered lookup buffers built from the Phase 1 train
feature encoder.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import nn


def _mlp(
    input_dim: int,
    hidden_dims: Sequence[int],
    dropout: float,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous = input_dim
    for hidden in hidden_dims:
        layers.extend(
            [nn.Linear(previous, hidden), nn.ReLU(), nn.Dropout(dropout)]
        )
        previous = hidden
    layers.append(nn.Linear(previous, 1))
    return nn.Sequential(*layers)


class DIN(nn.Module):
    def __init__(
        self,
        field_cardinalities: Sequence[int],
        *,
        embedding_dim: int = 16,
        hidden_dims: Sequence[int] = (256, 128, 64),
        attention_hidden_dim: int = 64,
        dropout: float = 0.2,
        item_category_lookup: np.ndarray,
        item_price_lookup: np.ndarray,
    ) -> None:
        super().__init__()
        self.field_cardinalities = list(field_cardinalities)
        self.embedding_dim = embedding_dim
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(cardinality, embedding_dim, padding_idx=0)
                for cardinality in field_cardinalities
            ]
        )
        for embedding in self.embeddings:
            nn.init.normal_(embedding.weight, std=0.01)
            with torch.no_grad():
                embedding.weight[0].zero_()
        self.attention = nn.Sequential(
            nn.Linear(embedding_dim * 4, attention_hidden_dim),
            nn.ReLU(),
            nn.Linear(attention_hidden_dim, 1),
        )
        self.dnn = _mlp(
            len(field_cardinalities) * embedding_dim + embedding_dim,
            hidden_dims,
            dropout,
        )
        self.register_buffer(
            "item_category_lookup",
            torch.from_numpy(item_category_lookup.astype(np.int64)),
            persistent=True,
        )
        self.register_buffer(
            "item_price_lookup",
            torch.from_numpy(item_price_lookup.astype(np.int64)),
            persistent=True,
        )

    def _history_vector(
        self, history: torch.Tensor, candidate: torch.Tensor
    ) -> torch.Tensor:
        item_embedding = self.embeddings[1]
        category_embedding = self.embeddings[2]
        price_embedding = self.embeddings[4]
        history_category = self.item_category_lookup[history]
        history_price = self.item_price_lookup[history]
        history_vector = (
            item_embedding(history)
            + category_embedding(history_category)
            + price_embedding(history_price)
        )
        candidate_expanded = candidate.unsqueeze(1).expand_as(history_vector)
        attention_input = torch.cat(
            [
                history_vector,
                candidate_expanded,
                history_vector * candidate_expanded,
                torch.abs(history_vector - candidate_expanded),
            ],
            dim=-1,
        )
        scores = self.attention(attention_input).squeeze(-1)
        mask = history.ne(0)
        scores = scores.masked_fill(~mask, -1e9)
        weights = torch.softmax(scores, dim=1)
        weights = torch.where(mask, weights, torch.zeros_like(weights))
        return (weights.unsqueeze(-1) * history_vector).sum(dim=1)

    def forward(
        self, fields: torch.Tensor, history: torch.Tensor
    ) -> torch.Tensor:
        if fields.ndim != 2 or fields.shape[1] != len(self.embeddings):
            raise ValueError("fields shape does not match field cardinalities")
        embedded = torch.stack(
            [
                embedding(fields[:, index])
                for index, embedding in enumerate(self.embeddings)
            ],
            dim=1,
        )
        candidate_item = (
            embedded[:, 1] + embedded[:, 2] + embedded[:, 4]
        )
        interest = self._history_vector(history, candidate_item)
        deep = self.dnn(
            torch.cat([embedded.flatten(start_dim=1), interest], dim=-1)
        ).squeeze(-1)
        return deep

    def predict_proba(
        self, fields: torch.Tensor, history: torch.Tensor
    ) -> torch.Tensor:
        return torch.sigmoid(self.forward(fields, history))
