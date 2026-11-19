# AgentRec

**Backend for a multi-agent, deep-learning personalized recommendation platform.**

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-EE4C2C?logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?logo=fastapi&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-1.2-1F2937)
![DeepSeek](https://img.shields.io/badge/LLM-DeepSeek-4B5563)
![Dataset](https://img.shields.io/badge/Dataset-Amazon%20Reviews%202023-FF9900)

AgentRec combines four-way candidate retrieval, DIN ranking, MMR diversity
re-ranking, a LangGraph multi-agent workflow, persistent user memory and
retrieval-augmented review evidence.

> **Backend-only repository.** The Streamlit frontend, the local test suite,
> the Amazon dataset, and trained checkpoints/indexes are intentionally not
> included. See [Data](#data) for the official dataset links and
> [Model artifacts](#model-artifacts) for the training commands.

---

## Contents

- [Highlights](#highlights)
- [Architecture](#architecture)
- [Recommendation pipeline](#recommendation-pipeline)
- [Agent workflow](#agent-workflow)
- [Repository layout](#repository-layout)
- [Data](#data)
- [Quick start](#quick-start)
- [Model artifacts](#model-artifacts)
- [API](#api)
- [Design notes](#design-notes)
- [What is not included](#what-is-not-included)
- [Citation](#citation)

---

## Highlights

- **Real data, no synthetic catalog** — every returned product comes from
  Amazon Reviews 2023 Electronics metadata. 
- **Four recall channels** — ItemCF, Two-Tower, LightGCN and a deterministic
  Popularity channel, each with per-channel scores/ranks preserved for audit.
- **DIN ranking + MMR re-ranking** — candidate-aware attention over user
  history, followed by category-diversity constraints.
- **Catalog fallback** — when collaborative recall is too sparse for a hard
  category/budget query, the engine fills the gap from real item metadata and
  sends the candidates through the same DIN/MMR/Critic pipeline.
- **LangGraph agents** — Intent, Profile, Recommendation, Review RAG, Critic
  and Memory update nodes with an explicit retry loop.
- **Persistent memory** — SQLite structured preferences plus optional Chroma
  semantic memory; temporary query constraints are never persisted.
- **Grounded explanations** — BGE review retrieval and a Critic Agent reject
  unsupported claims; evidence is scoped by `parent_asin`.
- **Reproducible training/evaluation scripts** — data download, preprocessing,
  model training, index building and offline evaluation are all in `scripts/`.

---

## Architecture

```mermaid
flowchart TD
    Client[API client] --> API[FastAPI]
    API -->|/api/v1/recommend/user/{user_id}| Engine[Recommendation Engine]
    API -->|/api/v1/recommend/chat| Graph[LangGraph Workflow]

    Engine --> ItemCF[ItemCF recall]
    Engine --> TwoTower[Two-Tower recall]
    Engine --> LightGCN[LightGCN recall]
    Engine --> Popularity[Popularity recall]

    ItemCF --> Fusion[Fusion: quota / RRF]
    TwoTower --> Fusion
    LightGCN --> Fusion
    Popularity --> Fusion

    Fusion --> Filters[Hard filters: history, category, budget]
    Filters --> Fallback[Catalog fallback if needed]
    Fallback --> DIN[DIN ranking]
    DIN --> MMR[MMR re-ranking]
    MMR --> Engine

    Graph --> Intent[Intent Agent]
    Graph --> Profile[Profile Agent]
    Graph --> Recommendation[Recommendation Agent]
    Recommendation --> Engine
    Graph --> ReviewRAG[Review RAG]
    Graph --> Critic[Critic Agent]
    Graph --> Memory[Memory update]
```

The direct recommendation endpoint is LLM-free. The chat endpoint wraps the
same engine in the LangGraph workflow for intent parsing, memory, evidence and
semantic validation.

---

## Recommendation pipeline

### 1. Four-way recall

| Channel | Signal | Output |
|---|---|---|
| **ItemCF** | Positive train interactions; cosine item-user similarity | Neighbour items from the user's history |
| **Two-Tower** | User/item features + ID embeddings; Faiss FlatIP | Vector nearest neighbours |
| **LightGCN** | User-item bipartite graph propagation; Faiss FlatIP | Graph-based nearest neighbours |
| **Popularity** | Train interaction counts, excluding the target user's history | Non-personalised sanity-check / cold-start candidates |

Each channel returns up to 100 candidates. Every candidate retains
`source_scores`, `raw_source_scores`, `source_ranks` and `retrieval_sources`.

### 2. Fusion

`quota` is the default strategy; `rrf` and `quota_rrf` are configurable.
Channel weights are validation-calibrated, and the Popularity channel is
weighted separately as a sanity-check channel. The fused candidate list is
capped at 300 by default.

### 3. Hard filters

Before ranking:

- exclude all of the user's train interactions
- exclude explicit `dislike` / `not_interested` / `purchased` feedback
- category matching by path segments (so `laptop` does not match
  `Laptop Accessories`)
- `budget_min` / `budget_max`
- unknown prices are excluded when a budget constraint is present

### 4. Catalog fallback

Collaborative recall is intentionally limited to a few hundred candidates. If
a category/budget constraint leaves fewer candidates than `top_k`, the engine
scans the real catalog, applies the same constraints, and ranks the resulting
real items through DIN/MMR. Fallback items are marked
`retrieval_sources=["catalog_fallback"]`.

### 5. DIN ranking

`src/ranking/ranker.py` loads `checkpoints/din_best.pt` and the matching feature
encoder. It scores real candidates using user history and item features.
`ranking_score` is a model preference score, **not a CTR estimate**.

### 6. MMR re-ranking

`src/ranking/reranker.py` applies category-aware MMR with `lambda=0.9` and a
per-category cap. Diversity is an explicit trade-off; it usually lowers
Recall@10 while increasing category diversity.

---

## Agent workflow

The chat endpoint runs a LangGraph state machine:

```text
Intent → Profile → Recommendation → Review RAG → Critic
                                      │
                                      ├─ rejected items → retry Recommendation
                                      └─ approved / retry limit → Finalize
                                                                  → Memory update
```

| Node | Responsibility |
|---|---|
| **Intent Agent** | DeepSeek JSON extraction: category, budget, preferences, brands, clarification, explicit memory updates |
| **Profile Agent** | Real train facts + SQLite memory + optional Chroma semantic memory |
| **Recommendation Agent** | Calls the controlled `recommend_products` tool; attaches reasons from real facts |
| **Review RAG** | BGE-small + 16 Chroma shards; evidence scoped by `parent_asin` |
| **Critic Agent** | Deterministic checks + DeepSeek semantic review; rejects unsupported claims |
| **Finalize** | Visible recommendations, status, assistant message, trace |
| **Memory update** | Applies only explicit durable preferences; temporary constraints are ignored |

Retry behavior: rejected item IDs are excluded and the Recommendation Agent is
called again, up to the configured `max_critic_retries` (default 2).


---

## Repository layout

```text
.
├── src/
│   ├── api/             FastAPI application
│   ├── agents/          LangGraph workflow, tools, prompts
│   ├── retrieval/       ItemCF, Two-Tower, LightGCN, Popularity, fusion
│   ├── ranking/         DIN/DeepFM rankers, feature encoder, MMR
│   ├── memory/          SQLite structured memory + Chroma semantic memory
│   ├── rag/             BGE embeddings and review retrieval
│   ├── data/            data loaders and preprocessing utilities
│   ├── evaluation/      retrieval/ranking metrics
│   └── services/        end-to-end RecommendationEngine
├── scripts/             download / preprocess / train / evaluate / acceptance
├── configs/             YAML configuration
├── docs/                experiment notes
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

The `frontend/`, `tests/`, `data/`, `checkpoints/`, `indexes/` and `logs/`
directories are intentionally absent from this repository.

---

## Data

The project uses the official **Amazon Reviews 2023 — Electronics** dataset
(McAuley Lab). The dataset is not redistributed here.

| Resource | Official link | Expected local path |
|---|---|---|
| Dataset home | https://amazon-reviews-2023.github.io/ | — |
| Hugging Face mirror | https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023 | — |
| 5-core train | [Electronics.train.csv.gz](https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/benchmark/5core/last_out_w_his/Electronics.train.csv.gz) | `data/raw/benchmark/5core/Electronics.train.csv.gz` |
| 5-core valid | [Electronics.valid.csv.gz](https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/benchmark/5core/last_out_w_his/Electronics.valid.csv.gz) | `data/raw/benchmark/5core/Electronics.valid.csv.gz` |
| 5-core test | [Electronics.test.csv.gz](https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/benchmark/5core/last_out_w_his/Electronics.test.csv.gz) | `data/raw/benchmark/5core/Electronics.test.csv.gz` |
| Metadata | [meta_Electronics.jsonl.gz](https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_Electronics.jsonl.gz) | `data/raw/metadata/meta_Electronics.jsonl.gz` |
| Reviews | [Electronics.jsonl.gz](https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Electronics.jsonl.gz) | `data/raw/reviews/Electronics.jsonl.gz` |

The included downloader uses the same URLs and verifies SHA-256 checksums:

```bash
python scripts/download_data.py --group all
```

### Current sample scale

The public development run uses `sample` mode, not the complete Electronics
category:

| Item | Scale |
|---|---:|
| Users | 20,000 |
| User-item interactions | 187,217 |
| Catalog items | 86,074 |
| Train interactions | 147,217 |
| Train-scored items | 72,447 |
| Valid / test targets | 20,000 / 20,000 |
| Scorable valid / test users | 10,073 / 9,258 |
| Cleaned reviews | 3,543,233 |
| Review RAG documents | 144,894 |

### Price and time boundary

- Prices are **historical USD** values from Amazon Reviews 2023, not live
  prices. Missing-price rate in the catalog is about **47.52%**.
- Interaction timestamps run from 1999-12-18 to 2023-08-31; review timestamps
  run from 1999-06-13 to 2023-09-12.
- The repository does not claim live inventory, real-time pricing, or online
  CTR improvement.

---

## Quick start

### 1. Clone

```bash
git clone https://github.com/WalkedAlone/MultiAgent-RecSys.git
cd MultiAgent-RecSys
```

### 2. Python environment

Python **3.11** is recommended.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1
python -m pip install -r requirements.txt
```

### 3. Configure

```bash
cp .env.example .env
# Edit .env and set DEEPSEEK_API_KEY
```

`DEEPSEEK_API_KEY` is required to start the full API/Agent workflow. The direct
`/api/v1/recommend/user/{user_id}` endpoint itself does not call an LLM, but the
application still initializes the Agent client at startup.

### 4. Prepare data

```bash
python scripts/download_data.py --group all
python scripts/preprocess_data.py --mode sample
```

The raw download is about **7.8 GB compressed / 30.2 GB decompressed**. A
sample run selects 20,000 users and builds the features/splits used by the
models.

### 5. Train models and build indexes

```bash
python scripts/train_itemcf.py --neighbors 100 --eval-split both
python scripts/train_two_tower.py \
  --objective softmax --temperature 0.07 --epochs 20 \
  --batch-size 4096 --patience 5 --eval-subset-every 1
python scripts/train_lightgcn.py \
  --learning-rate 0.05 --epochs 50 --patience 10 --eval-subset-every 5
python scripts/build_faiss_index.py --model all
python scripts/train_deepfm.py --epochs 4 --batch-size 1024 --negatives 4
python scripts/train_din.py --epochs 3 --batch-size 1024 --negatives 4 --max-history 50
python scripts/build_sharded_review_index.py --shards 16 --max-reviews-per-item 2 --rebuild
```

Training is CPU-compatible but not fast. The scripts write artifacts to
`checkpoints/` and `indexes/`; these directories are ignored by Git.

### 6. Run the API

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

OpenAPI docs: http://127.0.0.1:8000/docs

### 7. Docker

```bash
cp .env.example .env   # if not already created
docker compose up --build -d
```

The Compose file mounts `data/`, `checkpoints/`, `indexes/` and `logs/` from
the host. They must exist before starting the container.

---

## Model artifacts

The following files are **not** in the repository:

```text
checkpoints/
  itemcf_similarity.npz
  itemcf_meta.json
  two_tower_best.pt
  lightgcn_best.pt
  deepfm_best.pt
  din_best.pt
  ranking_encoder.json
  ranking_encoder.npz

indexes/
  two_tower.faiss
  two_tower_user_vectors.npy
  lightgcn.faiss
  lightgcn_user_vectors.npy
  review_vectors/
  chroma_memory/
```

Generate them with the commands in [Quick start](#quick-start), or provide your
own artifacts with the same filenames. The API will not start correctly without
the model checkpoints and retrieval indexes.

A lightweight artifact check is available:

```bash
python scripts/acceptance_check.py --skip-api
```

This checks that the expected files and memory database exist and runs the
local recommendation engine without calling DeepSeek.

---

## API

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/v1/health` | Component health and review-index count |
| `POST` | `/api/v1/recommend/user/{user_id}` | Pure algorithm recommendation (no LLM) |
| `POST` | `/api/v1/recommend/chat` | Full LangGraph Agent recommendation |
| `GET` | `/api/v1/user/{user_id}/profile` | Real profile facts + memory |
| `POST` | `/api/v1/user/{user_id}/preferences` | Add an explicit long-term preference |
| `DELETE` | `/api/v1/user/{user_id}/preferences/{preference_id}` | Delete a long-term preference |
| `POST` | `/api/v1/feedback` | `like` / `dislike` / `purchased` / `not_interested` |
| `GET` | `/api/v1/products/{item_id}` | Real product metadata |
| `GET` | `/api/v1/metrics` | Request count, errors and latency percentiles |
| `GET` | `/api/v1/metrics/model` | Offline evaluation JSON |

### Pure recommendation

```bash
curl -X POST http://127.0.0.1:8000/api/v1/recommend/user/AGAV7IY4HWGYRUIAGUZRXYUJ22DA \
  -H "Content-Type: application/json" \
  -d '{
    "top_k": 5,
    "category": "laptop",
    "budget_max": 1000,
    "use_mmr": true,
    "use_memory": false
  }'
```

### Agent recommendation

```bash
curl -X POST http://127.0.0.1:8000/api/v1/recommend/chat \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "AGAV7IY4HWGYRUIAGUZRXYUJ22DA",
    "query": "推荐 1500 美元以内的笔记本电脑",
    "thread_id": "demo-1"
  }'
```

The chat response includes `intent`, `profile`, `recommendations`, `critic`,
`review_evidence`, `agent_trace`, `memory_update` and `assistant_message`.

### Authentication

Set `RECSYS_API_KEY` in `.env` to require `X-API-Key` on protected endpoints.
Leave it empty for local development.

---

## Design notes

- **No fabricated products.** The LLM never returns item IDs or prices. The
  recommendation engine and metadata tables are the only sources of products.
- **Controlled tools.** `ProductToolkit` exposes a typed `ToolRegistry`; the
  workflow executes registered Python functions, not model-generated code.
- **Critic loop.** Deterministic checks cover existence, price, category,
  provenance and review-parent matching. DeepSeek adds semantic relevance
  review. Rejected items are excluded before retrying.
- **Grounding.** Review evidence is retrieved per `parent_asin`; missing
  evidence is reported as `no_evidence`, never invented.
- **Memory.** Only explicit durable preferences or feedback are written.
  Temporary budget/category constraints do not enter long-term memory.
- **Currency.** Dataset prices are historical USD. The Intent prompt asks for a
  USD restatement when a non-USD budget is supplied; no exchange-rate
  conversion is performed.
- **Catalog fallback.** Hard constraints can exhaust the collaborative recall
  pool. The fallback scans real metadata and marks those items so the source is
  auditable.

---

## What is not included

This repository is intentionally backend-only:

- **Frontend:** the Streamlit UI is not included.
- **Tests:** the local test suite is not included.
- **Data:** raw and processed Amazon files are not included; use the official
  links above.
- **Artifacts:** checkpoints, Faiss indexes, review vectors, SQLite memory and
  logs are not included.
- **Secrets:** `.env` is ignored; only `.env.example` is committed.

This keeps the repository focused on the API, recommendation engine, Agent
workflow and training/evaluation scripts.

---

## Citation

If you use the Amazon Reviews 2023 dataset, please cite:

```bibtex
@article{hou2024bridging,
  title={Bridging Language and Items for Retrieval and Recommendation},
  author={Hou, Yupeng and Li, Jiacheng and He, Zhankui and Yan, An and Chen, Xiusi and McAuley, Julian},
  journal={arXiv preprint arXiv:2403.03952},
  year={2024}
}
```

## Acknowledgements

- [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) and the McAuley Lab
- [BAAI BGE](https://huggingface.co/BAAI/bge-small-en-v1.5) for local embeddings
- [DeepSeek](https://www.deepseek.com/) for the LLM backend
- FastAPI, PyTorch, Faiss, LangGraph, Chroma and SQLite

## License

No license has been specified for this repository. Add an explicit license
before redistributing or using it commercially.
