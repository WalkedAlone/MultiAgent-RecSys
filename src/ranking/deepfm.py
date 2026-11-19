"""DeepFM ranker (FM first/second order + DNN) for pointwise preference."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
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


class DeepFM(nn.Module):
    """DeepFM with per-field embeddings and a binary logit output."""

    def __init__(
        self,
        field_cardinalities: Sequence[int],
        *,
        embedding_dim: int = 16,
        hidden_dims: Sequence[int] = (256, 128, 64),
        dropout: float = 0.2,
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
        self.linear_embeddings = nn.ModuleList(
            [
                nn.Embedding(cardinality, 1, padding_idx=0)
                for cardinality in field_cardinalities
            ]
        )
        self.bias = nn.Parameter(torch.zeros(1))
        for embedding in self.embeddings:
            nn.init.normal_(embedding.weight, std=0.01)
            with torch.no_grad():
                embedding.weight[0].zero_()
        for embedding in self.linear_embeddings:
            nn.init.zeros_(embedding.weight)
        self.dnn = _mlp(
            len(field_cardinalities) * embedding_dim,
            hidden_dims,
            dropout,
        )

    def forward(self, fields: torch.Tensor) -> torch.Tensor:
        if fields.ndim != 2 or fields.shape[1] != len(self.embeddings):
            raise ValueError("fields shape does not match field cardinalities")
        embedded = torch.stack(
            [
                embedding(fields[:, index])
                for index, embedding in enumerate(self.embeddings)
            ],
            dim=1,
        )
        linear = self.bias
        for index, embedding in enumerate(self.linear_embeddings):
            linear = linear + embedding(fields[:, index]).squeeze(-1)
        summed = embedded.sum(dim=1)
        square_summed = embedded.pow(2).sum(dim=1)
        fm_second = 0.5 * (summed.pow(2) - square_summed).sum(dim=1)
        deep = self.dnn(embedded.flatten(start_dim=1)).squeeze(-1)
        return linear + fm_second + deep

    def predict_proba(self, fields: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(fields))
