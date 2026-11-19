#!/usr/bin/env python
"""Train and evaluate LightGCN on the sample bipartite graph."""

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
from src.retrieval.lightgcn import (  # noqa: E402
    LightGCNModel,
    build_normalized_adjacency,
    compute_lightgcn_vectors,
)
from src.retrieval.two_tower import (  # noqa: E402
    bpr_loss,
    build_bpr_pairs,
    build_user_exclusions,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train LightGCN")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--embedding-dim", type=int, default=None)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--eval-subset-every", type=int, default=5)
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
    user_map = pl.read_parquet(root / "data/processed/users/user_id_map.parquet")
    item_map = pl.read_parquet(root / "data/processed/items/item_id_map.parquet")
    samples = pl.read_parquet(
        root / "data/processed/interactions/train_samples.parquet"
    )
    pairs = build_bpr_pairs(samples)
    if pairs.height == 0:
        raise RuntimeError("No BPR pairs found")
    adjacency, num_users, num_items, node_count = build_normalized_adjacency(
        train, user_map, item_map
    )
    device = resolve_device(config.model.device)
    adjacency = adjacency.to(device)
    user_nodes = torch.from_numpy(
        pairs["user_idx"].to_numpy().astype(np.int64)
    ).to(device)
    positive_nodes = torch.from_numpy(
        (num_users + pairs["positive_item"].to_numpy()).astype(np.int64)
    ).to(device)
    negative_nodes = torch.from_numpy(
        (num_users + pairs["negative_item"].to_numpy()).astype(np.int64)
    ).to(device)
    print(
        f"LightGCN: users={num_users} items={num_items} nodes={node_count} "
        f"pairs={pairs.height} edges={adjacency._nnz() // 2}"
    )

    model_cfg = config.model.lightgcn
    embedding_dim = args.embedding_dim or model_cfg.embedding_dim
    layers = args.layers or model_cfg.num_layers
    epochs = args.epochs or model_cfg.epochs
    learning_rate = args.learning_rate or model_cfg.learning_rate
    patience = args.patience if args.patience is not None else model_cfg.patience
    model = LightGCNModel(
        node_count=node_count, embedding_dim=embedding_dim, num_layers=layers
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=1e-5
    )
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
        optimizer.zero_grad()
        final = model.propagate(adjacency)
        user_vector = final[user_nodes]
        positive_vector = final[positive_nodes]
        negative_vector = final[negative_nodes]
        loss = bpr_loss(user_vector, positive_vector, negative_vector)
        loss.backward()
        optimizer.step()
        model.eval()
        user_vectors, item_vectors = compute_lightgcn_vectors(
            model, adjacency, num_users, num_items
        )
        subset_metrics = evaluate_embedding_retrieval(
            user_vectors, item_vectors, subset, exclusions
        )
        ndcg10 = float(subset_metrics.get("ndcg@10", 0.0))
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(loss.item()),
                "subset_ndcg@10": ndcg10,
            }
        )
        print(f"epoch {epoch}/{epochs} loss={loss.item():.5f} ndcg@10={ndcg10:.4f}")
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
    user_vectors, item_vectors = compute_lightgcn_vectors(
        model, adjacency, num_users, num_items
    )

    index_dir = root / "indexes"
    index_dir.mkdir(parents=True, exist_ok=True)
    np.save(index_dir / "lightgcn_user_vectors.npy", user_vectors)
    np.save(index_dir / "lightgcn_item_embeddings.npy", item_vectors[1:])
    item_ids = item_map.sort("item_idx")["parent_asin"].to_list()
    (index_dir / "lightgcn_item_ids.json").write_text(
        json.dumps(item_ids, ensure_ascii=False), encoding="utf-8"
    )
    index = VectorIndex.build(
        item_vectors[1:], item_ids, root / "indexes/lightgcn.faiss"
    )
    save_checkpoint(
        root / "checkpoints/lightgcn_best.pt",
        model_state=best_state,
        config={
            "node_count": node_count,
            "embedding_dim": embedding_dim,
            "num_layers": layers,
            "num_users": num_users,
            "num_items": num_items,
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
        metrics[split] = split_metrics
        print(f"\n[{split}] LightGCN metrics:")
        for key in (
            "users",
            "skipped",
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
        save_json(split_metrics, root / f"logs/metrics_lightgcn_{split}.json")
    save_json(
        {
            "model": "lightgcn",
            "backend": index.backend,
            "index_size": index.size,
            "node_count": node_count,
            "edges_directed": int(adjacency._nnz()),
            "pairs": pairs.height,
            "epochs_requested": epochs,
            "best_epoch": best_epoch,
            "history": history,
            "metrics": metrics,
        },
        root / "logs/phase2_lightgcn.json",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
