#!/usr/bin/env python
"""Train and evaluate the Two-Tower BPR retriever on the sample dataset."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
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

from src.core.config import get_app_config  # noqa: E402
from src.core.logger import setup_logging  # noqa: E402
from src.core.training import (  # noqa: E402
    resolve_device,
    save_checkpoint,
    save_json,
    set_seed,
)
from src.evaluation.embedding_eval import evaluate_embedding_retrieval  # noqa: E402
from src.retrieval.faiss_index import VectorIndex  # noqa: E402
from src.retrieval.two_tower import (  # noqa: E402
    TwoTowerFeatureEncoder,
    TwoTowerModel,
    bpr_loss,
    build_bpr_pairs,
    build_user_exclusions,
    compute_two_tower_vectors,
    make_loader,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train Two-Tower BPR")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--eval-subset-every", type=int, default=5)
    parser.add_argument(
        "--objective", choices=["softmax", "bpr"], default="softmax"
    )
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument(
        "--eval-split", choices=["valid", "test", "both"], default="both"
    )
    args = parser.parse_args(argv)

    setup_logging(level="INFO")
    config = get_app_config()
    set_seed(config.model.seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    root = config.project_root

    train = pl.read_parquet(root / "data/splits/train.parquet")
    user_features = pl.read_parquet(root / "data/features/user_features.parquet")
    item_features = pl.read_parquet(root / "data/features/item_features.parquet")
    item_map = pl.read_parquet(root / "data/processed/items/item_id_map.parquet")
    user_map = pl.read_parquet(root / "data/processed/users/user_id_map.parquet")
    samples = pl.read_parquet(
        root / "data/processed/interactions/train_samples.parquet"
    )
    if args.objective == "bpr":
        pairs = build_bpr_pairs(samples)
        if pairs.height == 0:
            raise RuntimeError("No BPR pairs found in train_samples.parquet")
        train_rows = pairs.height
    else:
        pairs = pl.DataFrame()
        positives = (
            samples.filter(pl.col("sample_type") == "positive")
            .select(["user_idx", "item_idx"])
            .unique()
        )
        if positives.height == 0:
            raise RuntimeError("No positive rows found in train_samples.parquet")
        train_rows = positives.height
    print(
        f"Two-Tower: users={user_map.height} items={item_map.height} "
        f"objective={args.objective} train_rows={train_rows}"
    )

    encoder = TwoTowerFeatureEncoder.fit(user_features, item_features)
    encoder.save(
        root / "checkpoints/two_tower_encoder.json",
        root / "checkpoints/two_tower_encoder.npz",
    )
    model_cfg = config.model.two_tower
    embedding_dim = model_cfg.user_embedding_dim
    model = TwoTowerModel(
        num_users=encoder.num_users,
        num_items=encoder.num_items,
        user_numeric_dim=encoder.user_numeric.shape[1],
        item_numeric_dim=encoder.item_numeric.shape[1],
        num_categories=encoder.num_categories,
        num_stores=encoder.num_stores,
        embedding_dim=embedding_dim,
        hidden_dims=model_cfg.hidden_dims,
        dropout=model_cfg.dropout,
    )
    device = resolve_device(config.model.device)
    model.to(device)
    batch_size = args.batch_size or model_cfg.batch_size
    epochs = args.epochs or model_cfg.epochs
    learning_rate = args.learning_rate or model_cfg.learning_rate
    patience = args.patience if args.patience is not None else model_cfg.patience
    if args.objective == "bpr":
        loader = make_loader(pairs, batch_size=batch_size, shuffle=True)
    else:
        from torch.utils.data import DataLoader, TensorDataset

        positive_dataset = TensorDataset(
            torch.from_numpy(
                positives["user_idx"].to_numpy().astype(np.int64)
            ),
            torch.from_numpy(
                positives["item_idx"].to_numpy().astype(np.int64)
            ),
        )
        loader = DataLoader(
            positive_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=False,
        )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    user_numeric = torch.from_numpy(encoder.user_numeric).to(device)
    user_category = torch.from_numpy(encoder.user_category).to(device)
    item_numeric = torch.from_numpy(encoder.item_numeric).to(device)
    item_category = torch.from_numpy(encoder.item_category).to(device)
    item_store = torch.from_numpy(encoder.item_store).to(device)
    exclusions = build_user_exclusions(train)
    valid_eval = pl.read_parquet(root / "data/splits/valid_eval.parquet")
    subset = valid_eval.gather_every(max(1, args.eval_subset_every))

    best_metric = -1.0
    best_state = None
    best_epoch = 0
    stale = 0
    history: list[dict] = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        batches = 0
        for batch in loader:
            if args.objective == "bpr":
                user_idx, positive, negative = batch
                user_idx = user_idx.to(device)
                positive = positive.to(device)
                negative = negative.to(device)
                user_vector = model.user_vector(
                    user_idx,
                    user_numeric[user_idx],
                    user_category[user_idx],
                )
                positive_vector = model.item_vector(
                    positive,
                    item_numeric[positive],
                    item_category[positive],
                    item_store[positive],
                )
                negative_vector = model.item_vector(
                    negative,
                    item_numeric[negative],
                    item_category[negative],
                    item_store[negative],
                )
                loss = bpr_loss(
                    user_vector, positive_vector, negative_vector
                )
            else:
                user_idx, positive = batch
                user_idx = user_idx.to(device)
                positive = positive.to(device)
                user_vector = model.user_vector(
                    user_idx,
                    user_numeric[user_idx],
                    user_category[user_idx],
                )
                item_vector = model.item_vector(
                    positive,
                    item_numeric[positive],
                    item_category[positive],
                    item_store[positive],
                )
                logits = user_vector @ item_vector.T / args.temperature
                labels = torch.arange(
                    logits.shape[0], device=device
                )
                loss = torch.nn.functional.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            batches += 1
        model.eval()
        user_vectors, item_vectors = compute_two_tower_vectors(
            model, encoder, device
        )
        subset_metrics = evaluate_embedding_retrieval(
            user_vectors, item_vectors, subset, exclusions
        )
        ndcg10 = float(subset_metrics.get("ndcg@10", 0.0))
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / max(1, batches),
                "subset_ndcg@10": ndcg10,
            }
        )
        print(
            f"epoch {epoch}/{epochs} loss={total_loss/max(1,batches):.5f} "
            f"subset_ndcg@10={ndcg10:.4f}"
        )
        if ndcg10 > best_metric:
            best_metric = ndcg10
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                print(f"early stopping at epoch {epoch}")
                break

    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    user_vectors, item_vectors = compute_two_tower_vectors(model, encoder, device)

    index_dir = root / "indexes"
    index_dir.mkdir(parents=True, exist_ok=True)
    np.save(index_dir / "two_tower_user_vectors.npy", user_vectors)
    np.save(index_dir / "two_tower_item_vectors.npy", item_vectors)
    item_ids = item_map.sort("item_idx")["parent_asin"].to_list()
    (index_dir / "two_tower_item_ids.json").write_text(
        json.dumps(item_ids, ensure_ascii=False), encoding="utf-8"
    )
    index = VectorIndex.build(
        item_vectors[1:],
        item_ids,
        root / "indexes/two_tower.faiss",
        use_faiss=True,
    )
    save_checkpoint(
        root / "checkpoints/two_tower_best.pt",
        model_state=best_state,
        config={
            "embedding_dim": embedding_dim,
            "hidden_dims": list(model_cfg.hidden_dims),
            "dropout": model_cfg.dropout,
            "num_users": encoder.num_users,
            "num_items": encoder.num_items,
            "num_categories": encoder.num_categories,
            "num_stores": encoder.num_stores,
        },
        extra={"best_epoch": best_epoch, "subset_ndcg@10": best_metric},
    )

    split_names = (
        ["valid", "test"] if args.eval_split == "both" else [args.eval_split]
    )
    metrics: dict[str, dict] = {}
    for split in split_names:
        eval_frame = pl.read_parquet(root / f"data/splits/{split}_eval.parquet")
        split_metrics = evaluate_embedding_retrieval(
            user_vectors, item_vectors, eval_frame, exclusions
        )
        split_metrics["runtime_seconds"] = 0.0
        metrics[split] = split_metrics
        print(f"\n[{split}] Two-Tower metrics:")
        for key in (
            "users",
            "recall@10",
            "recall@20",
            "recall@50",
            "recall@100",
            "ndcg@10",
            "ndcg@20",
            "ndcg@50",
            "hit_rate@10",
            "mrr",
            "coverage@50",
        ):
            print(f"  {key}: {split_metrics.get(key)}")
        save_json(
            split_metrics, root / f"logs/metrics_two_tower_{split}.json"
        )
    save_json(
        {
            "model": "two_tower",
            "backend": index.backend,
            "index_size": index.size,
            "objective": args.objective,
            "train_rows": train_rows,
            "temperature": args.temperature,
            "batch_size": batch_size,
            "epochs_requested": epochs,
            "best_epoch": best_epoch,
            "history": history,
            "metrics": metrics,
        },
        root / "logs/phase2_two_tower.json",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
