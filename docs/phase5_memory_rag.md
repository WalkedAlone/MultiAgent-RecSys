# Phase 5：用户记忆与评论 RAG 设计记录

日期：2026-11-18

## 1. 结构化记忆（SQLite / SQLAlchemy）

数据库：`data/processed/user_profiles.sqlite`

| 表 | 内容 |
|---|---|
| `user_profiles` | 用户画像行、训练交互数、更新时间、来源/置信度 |
| `memory_preferences` | kind/value、`explicit_user` / `feedback` / `inferred_from_train`、置信度、active、时间 |
| `recommendation_feedback` | like / dislike / purchased / not_interested、item_id、query、时间 |

规则：

1. `UserProfileStore` 的真实 train 特征只作为 **inferred_from_train** 种子；
   不会自动变成硬过滤条件。
2. 只有显式 durable 语句或反馈才写入 `explicit_user` / `feedback`。
3. 当前查询中的预算/类别/用途不会写入长期记忆。
4. 只有显式 memory 才进入 `effective_constraints`（category/budget fallback）。
5. 修改和删除采用 `active=false` 软删除；来源、置信度、更新时间保留。
6. 反馈会写回 preference；dislike / not_interested / purchased 会在下一次
   推荐时作为额外排除项，like 会参与 soft brand ordering。

## 2. 语义记忆（Chroma + BGE）

- 本地 `BAAI/bge-small-en-v1.5`，384 维，完全不调用 DeepSeek embedding；
- Chroma 持久目录：`indexes/chroma_memory`；
- collection：`user_semantic_memory`；
- 每条文档带 `user_id`、kind、value、source、confidence、updated_at 元数据；
- 查询时必须使用 `where={"user_id": ...}`，不会跨用户检索；
- 删除按 deterministic memory id 同步删除。

## 3. 评论 RAG

### 3.1 真实评论预处理

- 源：`data/processed/reviews/reviews.parquet`（Phase 1 真实评论）；
- DuckDB 窄列窗口查询选出每个商品最多 N 条评论，避免把 1.3 GB 文本
  全量载入内存；
- 选择顺序：verified purchase、helpful_vote、最新时间、review_id；
- 每篇 review 独立向量，metadata 保留 review_id / parent_asin / rating /
  timestamp / helpful_vote / verified_purchase；
- 仅索引 **train item map 中的商品**（推荐引擎可能推荐的 72,447 个商品），
  默认最多 2 条/商品；
- 全量 3.5M 评论没有全部索引；这是 CPU 和时间约束下的明确范围，不是伪造。

### 3.2 本地 BGE + Chroma

- BGE 模型懒加载；
- Chroma 分片（16 个 shard，每个约 9k 向量），避免在 Windows/Chroma 0.5
  上单个超大 HNSW 持久文件不可重载的问题；
- 商品通过 `parent_asin` 哈希到固定 shard；
- 查询时 Chroma `where={"parent_asin": item_id}`，Python 侧再次校验
  metadata 的 parent_asin，禁止把 A 商品评论用于 B 商品；
- 没有匹配证据时明确返回 `no_evidence`，不生成评论内容。

### 3.3 工作流中的证据

`Review RAG` 节点：

1. 对 Top-10 推荐逐商品查询真实评论；
2. 每条推荐附加 `review_evidence` 列表和 `evidence_summary`；
3. Critic 接收证据文本，检查解释是否有真实依据；
4. 最终响应保留 review_id、评分、时间、verified、文本片段；
5. 索引缺失/为空时明确 `review_status=index_empty/unavailable`。

## 4. 复现命令

```bash
# 构建评论分片索引（默认 16 shards，2 reviews/train item）
python scripts/build_sharded_review_index.py --shards 16 --workers 2 --rebuild

# 查询真实评论证据
python scripts/search_reviews.py --item-id B00OBRE5UE \
  --query "Samsung SSD reliability performance" --top-k 2

# 结构化 + 语义记忆
python scripts/manage_memory.py --user-id <USER_ID> --show
python scripts/manage_memory.py --user-id <USER_ID> --add brand ASUS
python scripts/manage_memory.py --user-id <USER_ID> --feedback B00OBRE5UE like
python scripts/manage_memory.py --user-id <USER_ID> --delete-id 3

# 完整 Agent（Phase 5 证据会进入最终推荐）
python scripts/run_agent.py --user-id <USER_ID> \
  --query "我想买一个 SSD 固态硬盘，预算不超过 200。"
```
