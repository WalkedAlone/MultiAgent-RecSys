#!/usr/bin/env python
"""Train and evaluate the DeepFM ranker with sampled validation negatives."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from src.core.config import get_app_config  # noqa: E402
from src.core.logger import setup_logging  # noqa: E402
from src.core.training import (  # noqa: E402
    resolve_device,
    save_checkpoint,
    save_json,
    set_seed,
)
from src.evaluation.ranking_metrics import evaluate_binary  # noqa: E402
from src.ranking.deepfm import DeepFM  # noqa: E402
from src.ranking.features import FIELD_NAMES, RankingFeatureEncoder  # noqa: E402


def predict(model: DeepFM, fields: np.ndarray, device: torch.device, batch_size: int = 4096) -> np.ndarray:
    model.eval()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(fields), batch_size):
            batch = torch.from_numpy(fields[start : start + batch_size]).to(device)
            outputs.append(torch.sigmoid(model(batch)).cpu().numpy())
    return np.concatenate(outputs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train DeepFM")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--negatives", type=int, default=4)
    args = parser.parse_args(argv)

    setup_logging(level="INFO")
    config = get_app_config()
    set_seed(config.model.seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    root = config.project_root
    train = pl.read_parquet(root / "data/splits/train.parquet")
    samples = pl.read_parquet(
        root / "data/processed/interactions/train_samples.parquet"
    )
    valid_eval = pl.read_parquet(root / "data/splits/valid_eval.parquet")
    user_features = pl.read_parquet(root / "data/features/user_features.parquet")
    item_features = pl.read_parquet(root / "data/features/item_features.parquet")

    encoder = RankingFeatureEncoder.fit(user_features, item_features)
    encoder.save(
        root / "checkpoints/ranking_encoder.json",
        root / "checkpoints/ranking_encoder.npz",
    )
    train_arrays = encoder.encode_samples(samples, include_history=False)
    valid_arrays = encoder.build_validation_samples(
        valid_eval,
        train,
        num_negatives=args.negatives,
        seed=config.data.negative_sampling.seed,
    )
    print(
        f"DeepFM: train={len(train_arrays['labels'])} "
        f"valid_sampled={len(valid_arrays['labels'])} "
        f"fields={len(encoder.field_cardinalities)}"
    )

    model_cfg = config.model.deepfm
    model = DeepFM(
        encoder.field_cardinalities,
        embedding_dim=model_cfg.embedding_dim,
        hidden_dims=model_cfg.hidden_dims,
        dropout=model_cfg.dropout,
    )
    device = resolve_device(config.model.device)
    model.to(device)
    batch_size = args.batch_size or model_cfg.batch_size
    epochs = args.epochs or model_cfg.epochs
    learning_rate = args.learning_rate or model_cfg.learning_rate
    dataset = TensorDataset(
        torch.from_numpy(train_arrays["fields"]),
        torch.from_numpy(train_arrays["labels"]).float(),
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.BCEWithLogitsLoss()

    best_auc = -1.0
    best_state = None
    best_epoch = 0
    stale = 0
    history: list[dict] = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        batches = 0
        for fields, labels in loader:
            fields = fields.to(device)
            labels = labels.to(device)
            logits = model(fields)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            batches += 1
        probabilities = predict(model, valid_arrays["fields"], device)
        metrics = evaluate_binary(valid_arrays["labels"], probabilities)
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / max(1, batches),
                "valid_auc": metrics["auc"],
                "valid_log_loss": metrics["log_loss"],
            }
        )
        print(
            f"epoch {epoch}/{epochs} loss={total_loss/max(1,batches):.5f} "
            f"valid_auc={metrics['auc']:.4f} logloss={metrics['log_loss']:.4f}"
        )
        if metrics["auc"] > best_auc:
            best_auc = metrics["auc"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                print(f"early stopping at epoch {epoch}")
                break
    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    probabilities = predict(model, valid_arrays["fields"], device)
    valid_metrics = evaluate_binary(valid_arrays["labels"], probabilities)
    valid_metrics["users"] = float(
        len({int(value) for value in valid_arrays["user_idx"]})
    )
    valid_metrics["rows"] = float(len(probabilities))
    save_checkpoint(
        root / "checkpoints/deepfm_best.pt",
        model_state=best_state,
        config={
            "field_cardinalities": encoder.field_cardinalities,
            "field_names": FIELD_NAMES,
            "embedding_dim": model_cfg.embedding_dim,
            "hidden_dims": list(model_cfg.hidden_dims),
            "dropout": model_cfg.dropout,
        },
        extra={"best_epoch": best_epoch, "valid_metrics": valid_metrics},
    )
    save_json(valid_metrics, root / "logs/metrics_deepfm_valid.json")
    save_json(
        {
            "model": "deepfm",
            "train_rows": int(len(train_arrays["labels"])),
            "valid_rows": int(len(valid_arrays["labels"])),
            "negatives_per_positive": args.negatives,
            "epochs_requested": epochs,
            "best_epoch": best_epoch,
            "history": history,
            "valid_metrics": valid_metrics,
        },
        root / "logs/phase2_deepfm.json",
    )
    print("\nDeepFM valid sampled metrics:")
    for key in ("auc", "log_loss", "accuracy", "users", "rows"):
        print(f"  {key}: {valid_metrics.get(key)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
