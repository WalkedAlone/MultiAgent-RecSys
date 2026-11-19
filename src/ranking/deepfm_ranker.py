"""Inference wrapper for the trained DeepFM comparison ranker."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import polars as pl
import torch

from src.core.config import AppConfig
from src.core.training import resolve_device
from src.ranking.deepfm import DeepFM
from src.ranking.features import RankingFeatureEncoder


class DeepFMRanker:
    """Batch-score candidates with a checkpoint trained in Phase 2.

    DeepFM is a pointwise comparison baseline, not an exposure/CTR model.  The
    returned ``ranking_score`` is a model preference score in ``[0, 1]``.
    """

    def __init__(
        self,
        *,
        model: DeepFM,
        encoder: RankingFeatureEncoder,
        device: torch.device,
        item_id_to_idx: dict[str, int],
        user_id_to_idx: dict[str, int],
        history_len_by_user: dict[str, int],
    ) -> None:
        self.model = model
        self.encoder = encoder
        self.device = device
        self.item_id_to_idx = item_id_to_idx
        self.user_id_to_idx = user_id_to_idx
        self.history_len_by_user = history_len_by_user
        self.model.eval()

    @classmethod
    def from_artifacts(
        cls,
        root: Path,
        config: AppConfig,
        train: pl.DataFrame,
    ) -> "DeepFMRanker":
        encoder = RankingFeatureEncoder.load(
            root / "checkpoints/ranking_encoder.json",
            root / "checkpoints/ranking_encoder.npz",
        )
        checkpoint = torch.load(
            root / "checkpoints/deepfm_best.pt",
            map_location="cpu",
            weights_only=False,
        )
        model_config = checkpoint["config"]
        model = DeepFM(
            model_config["field_cardinalities"],
            embedding_dim=model_config["embedding_dim"],
            hidden_dims=model_config["hidden_dims"],
            dropout=model_config["dropout"],
        )
        model.load_state_dict(checkpoint["model_state"])
        device = resolve_device(config.model.device)
        model.to(device)
        model.eval()

        item_map = pl.read_parquet(
            root / "data/processed/items/item_id_map.parquet"
        )
        user_map = pl.read_parquet(
            root / "data/processed/users/user_id_map.parquet"
        )
        item_id_to_idx = {
            str(row["parent_asin"]): int(row["item_idx"])
            for row in item_map.iter_rows(named=True)
        }
        user_id_to_idx = {
            str(row["user_id"]): int(row["user_idx"])
            for row in user_map.iter_rows(named=True)
        }
        history_frame = (
            train.select(["user_id", "timestamp_ms"])
            .sort(["user_id", "timestamp_ms"])
            .group_by("user_id", maintain_order=True)
            .len()
        )
        history_len_by_user = {
            str(row["user_id"]): int(row["len"])
            for row in history_frame.iter_rows(named=True)
        }
        return cls(
            model=model,
            encoder=encoder,
            device=device,
            item_id_to_idx=item_id_to_idx,
            user_id_to_idx=user_id_to_idx,
            history_len_by_user=history_len_by_user,
        )

    def score(
        self, user_id: str, item_ids: Sequence[str]
    ) -> list[dict[str, Any]]:
        user_idx = self.user_id_to_idx.get(str(user_id))
        if user_idx is None:
            return []
        pairs: list[tuple[str, int]] = []
        for item_id in item_ids:
            item_idx = self.item_id_to_idx.get(str(item_id))
            if item_idx is not None and item_idx > 0:
                pairs.append((str(item_id), int(item_idx)))
        if not pairs:
            return []
        history_len = int(self.history_len_by_user.get(str(user_id), 0))
        item_idxs = np.asarray([item_idx for _, item_idx in pairs], dtype=np.int64)
        user_idxs = np.full(item_idxs.shape[0], user_idx, dtype=np.int64)
        history_lengths = np.full(
            item_idxs.shape[0], history_len, dtype=np.float64
        )
        fields = self.encoder.encode_points(
            user_idxs, item_idxs, history_lengths
        )
        scores: list[float] = []
        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(fields), 4096):
                batch = torch.from_numpy(fields[start : start + 4096]).to(
                    self.device
                )
                logits = self.model(batch)
                scores.extend(torch.sigmoid(logits).cpu().numpy().tolist())
        return [
            {
                "item_id": item_id,
                "item_idx": item_idx,
                "ranking_score": float(score),
                "history_len": history_len,
            }
            for (item_id, item_idx), score in zip(pairs, scores)
        ]
