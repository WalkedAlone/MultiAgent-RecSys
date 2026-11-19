# Phase 7：召回、排序与 Agent 消融实验

日期：2026-11-18

## 1. 实验范围与口径

- 数据：Amazon Reviews 2023 Electronics 5-Core；直接使用 Phase 1 的官方
  train/valid/test 划分与 `valid_eval.parquet` / `test_eval.parquet`。
- 目标：每个用户的可评分留出正样本 `parent_asin`。
- 用户：valid 10,073；test 9,258；商品目录 72,447。
- 召回候选：四路各取 100（ItemCF / Two-Tower / LightGCN / Popularity），
  quota 融合最多 300；训练历史排除。Popularity 是无个性化基线通道，不是
  训练模型。
- 目标若不在候选列表：rank = `catalog_size + 1`（72,448）。
- 指标：Recall@K、NDCG@K、MRR、Coverage@K、CategoryDiversity@K。
- **所有指标是偏好排序指标，不是 CTR、曝光转化率或线上收益。**
- 模型分数是模型偏好分数；AUC 是采样诊断值，不能描述为 CTR。
- Bootstrap：按用户配对重采样 1,000 次，seed=42，报告 treatment−baseline
  的均值差与 95% 分位区间；区间跨 0 时不声称显著。

## 2. 召回消融

### 2.1 单通道与融合

| 变体 | valid R@10 | valid NDCG@10 | test R@10 | test NDCG@10 |
|---|---:|---:|---:|---:|
| ItemCF | 0.00278 | 0.00148 | 0.00173 | 0.00097 |
| Two-Tower | 0.00596 | 0.00286 | 0.00378 | 0.00188 |
| LightGCN | 0.00546 | 0.00285 | 0.00346 | 0.00173 |
| **Popularity（基线）** | **0.02978** | **0.01462** | **0.02592** | **0.01270** |
| quota，四通道，校准权重 | 0.00844 | 0.00382 | 0.00518 | 0.00241 |
| RRF，四通道，校准权重 | 0.01092 | 0.00645 | 0.00767 | 0.00390 |
| quota-RRF，四通道 | 0.01092 | 0.00645 | 0.00767 | 0.00390 |
| quota，三通道，无 Popularity | 0.00814 | 0.00372 | 0.00508 | 0.00238 |
| quota，四通道，等权 | 0.00645 | 0.00269 | 0.00540 | 0.00232 |
| quota，ItemCF+Two-Tower | 0.00655 | 0.00305 | 0.00400 | 0.00195 |
| quota，ItemCF+LightGCN | 0.00566 | 0.00292 | 0.00367 | 0.00180 |
| quota，Two-Tower+LightGCN | 0.00804 | 0.00369 | 0.00508 | 0.00238 |

补充 Coverage@50（valid/test）：

- ItemCF 0.7521 / 0.7444
- Two-Tower 0.1305 / 0.1283
- LightGCN 0.7928 / 0.7777
- Popularity 0.0008 / 0.0008（覆盖率低但命中率高）
- quota 四通道 0.7502 / 0.7373
- RRF 四通道 0.7056 / 0.6854
- quota 三通道 0.7569 / 0.7446
- equal-weights quota 四通道 0.7902 / 0.7785

解释：

- **Popularity 单通道 R@10 高于任何训练通道和融合结果**；这是必须如实
  报告的基线，而不是可以省略的对比。
- 四通道 quota 融合的 R@10 仅比三通道高 0.0003/0.0001；但它把 R@50 从
  0.0163/0.0134 提升到 0.0216/0.0187，给 DIN 提供了更大的候选池。
- RRF 的融合层 R@10/NDCG 高于 quota，但端到端 DIN/MMR 与 quota 差异很小；
  项目默认保留 quota，不是声称 RRF 全面更差。

### 2.2 配对 Bootstrap

| 比较（treatment − baseline） | valid delta [95% CI] | test delta [95% CI] |
|---|---:|---:|
| 校准权重 − 等权（NDCG@10） | -0.001130 [-0.001901, -0.000378] | -0.000093 [-0.000927, +0.000653] |
| 融合 − ItemCF（NDCG@10） | +0.002340 [+0.001444, +0.003327] | +0.001447 [+0.000661, +0.002339] |
| 融合 − Two-Tower（NDCG@10） | +0.000968 [+0.000439, +0.001528] | +0.000534 [+0.000143, +0.000978] |
| 融合 − LightGCN（NDCG@10） | +0.000973 [+0.000256, +0.001783] | +0.000688 [-0.000018, +0.001487] |
| RRF − quota（NDCG@10） | +0.002630 [+0.001613, +0.003586] | +0.001481 [+0.000559, +0.002440] |
| Two-Tower+LightGCN − 四通道（R@50） | -0.004864 [-0.006651, -0.003276] | -0.004861 [-0.006589, -0.003024] |
| Popularity 单通道 − 无 Popularity 融合（R@10） | +0.021642 [+0.018465, +0.024819] | +0.020847 [+0.017496, +0.024195] |
| 四通道融合 − 无 Popularity 融合（R@10） | +0.000298 [0.0, +0.000695] | +0.000108 [0.0, +0.000324] |
| 四通道融合 − 无 Popularity 融合（R@50） | +0.005361 [+0.003872, +0.006949] | +0.005293 [+0.003781, +0.006808] |

结论：

1. 校准权重相对等权在 valid 有正向配对区间；test 区间跨 0，不能外推到
   所有数据。
2. 融合比三个训练通道的 NDCG@10 都有提升；LightGCN 的 test 区间仍跨 0。
3. RRF 的 NDCG@10 在本轮显著高于 quota，但端到端 DIN/MMR 差异很小；
   不能据此声称 RRF 全面更好。
4. Popularity 单通道显著高于无 Popularity 融合；四通道融合的 R@10 提升
   几乎为 0，但 R@50 提升显著。收益主要是候选池变深，不是 Top-10 本身。

## 3. 排序与重排消融

同一 candidate pool（四通道 quota 融合，校准权重，Popularity=0.5）上比较：

| 变体 | valid R@10 | valid NDCG@10 | valid Div@10 | test R@10 | test NDCG@10 | test Div@10 |
|---|---:|---:|---:|---:|---:|---:|
| 融合顺序（无精排） | 0.00844 | 0.00382 | 0.3405 | 0.00518 | 0.00241 | 0.3412 |
| DeepFM | 0.02224 | 0.01007 | 0.2768 | 0.02041 | 0.00918 | 0.2779 |
| DIN | **0.02541** | **0.01224** | 0.3810 | **0.02106** | **0.01062** | 0.3823 |
| DeepFM + MMR | 0.01638 | 0.00774 | 0.5711 | 0.01620 | 0.00768 | 0.5701 |
| DIN + MMR | 0.01976 | 0.00949 | **0.5682** | 0.01761 | 0.00899 | **0.5670** |

| 比较（treatment − baseline） | valid delta [95% CI] | test delta [95% CI] |
|---|---:|---:|
| DIN − 融合（R@10） | +0.016976 [+0.014293, +0.019954] | +0.015878 [+0.012635, +0.018903] |
| DeepFM − DIN（R@10） | -0.003177 [-0.005460, -0.000988] | -0.000648 [-0.003024, +0.001620] |
| DIN+MMR − DIN（R@10） | -0.005659 [-0.007346, -0.003872] | -0.003456 [-0.005187, -0.001728] |
| DIN+MMR − DIN（Div@10） | +0.187203 [+0.184801, +0.189755] | +0.184694 [+0.182003, +0.187211] |
| DeepFM+MMR − DeepFM（R@10） | -0.005857 [-0.007942, -0.003870] | -0.004213 [-0.006265, -0.002052] |

解释：

- DIN 相对纯融合顺序显著提升 R@10，两个 split 的区间都不跨 0；加入
  Popularity 候选池后，DIN 的 valid/test R@10 从 0.01787/0.01102 提升到
  0.02541/0.02106，但仍低于 Popularity 单通道 0.02978/0.02592。
- DeepFM 与 DIN 在 valid 上 DIN 更好；test 上差异区间跨 0，不能外推为
  所有数据上的普遍结论。
- Phase 2 的采样验证 AUC：DeepFM 0.8318，DIN 0.8346；这是 1 正:4 负
  采样诊断，不是 CTR。
- MMR 以 Recall 下降换取类别多样性，两个 split 都明确；不能表述为
  “MMR 同时提升相关性和多样性”。

## 4. Agent 消融

### 4.1 变体

| 变体 | 实现 |
|---|---|
| `full` | Phase 6 完整 workflow：Intent + Profile + Recommendation + Review RAG + Critic + Memory |
| `no_rag` | `enable_review_rag=False`，Review RAG 节点明确 skipped，不生成证据 |
| `no_memory` | `memory_manager=None`；ProfileAgent `use_persistent_memory=False`，只保留真实训练事实 |
| `no_critic` | `enable_critic=False`，图中不添加 Critic 节点，直接 finalize；**不是生产模式** |

固定查询（单次运行，temperature=0.2）：

1. SSD：`我想买一个用于日常办公的 SSD，预算不超过 200 美元。`
2. 耳机：`推荐一款头戴式降噪耳机，预算 150 美元以内。`
3. 键盘：`推荐一款适合办公的键盘。`（无显式预算；用于观察长期记忆）

### 4.2 每次运行的真实结果

| 查询 | full | no_rag | no_memory | no_critic |
|---|---|---|---|---|
| SSD | 7 个商品，证据覆盖率 1.0 | 7 个，证据 0.0 | 7 个，证据 1.0 | 10 个，证据 1.0 |
| 耳机 | 2 个，证据 1.0，`completed_with_warnings` | 1 个，证据 0.0 | 2 个，证据 1.0 | 10 个，证据 1.0 |
| 键盘 | 10 个，证据 1.0 | 9 个，证据 0.0 | 10 个，证据 1.0 | 10 个，证据 1.0 |

聚合：

| 变体 | runs | recommendations | evidence coverage | budget/category violations | memory constraint runs | mean tokens | mean latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| full | 3 | 19 | 1.000 | 0 / 0 | 3 | 24,557 | 54,012 ms |
| no_rag | 3 | 17 | 0.000 | 0 / 0 | 3 | 23,585 | 52,441 ms |
| no_memory | 3 | 19 | 1.000 | 0 / 0 | 0 | 25,914 | 57,724 ms |
| no_critic | 3 | 30 | 1.000 | 0 / 0 | 3 | 5,714 | 12,166 ms |

### 4.3 观察

> 本轮 Agent 消融于 2026-11-18 重新运行，已包含 Popularity 通道和
> `catalog_fallback`；下表是当前引擎的真实结果，旧结果不再使用。

1. **Review RAG**：`full` 的 19 个最终商品全部有真实评论证据；
   `no_rag` 为 0/17。SSD 数量相同（7/7），耳机与键盘数量不同
   （2 vs 1、10 vs 9），说明 RAG 还通过 Critic/重试链路影响最终候选；
   本组实验只能证明证据可追溯性，不声称 RAG 提升了 Recall。
2. **Memory**：`full` 与 `no_memory` 最终都是 19 个商品；full 的 3 次
   运行应用了记忆约束，no_memory 为 0。本组查询没有显示最终推荐数量
   差异，不能声称记忆提升了相关性；它证明的是候选过滤条件确实不同。
3. **Critic**：`full` 在耳机查询中得到 2 个商品并标记
   `completed_with_warnings`，no_critic 直接返回 10 个；SSD 7 vs 10、
   键盘 10 vs 10。Critic 会拒绝语义不匹配的候选并触发重试，这是
   确定性别名/类目过滤之外的语义安全检查。
4. **代价**：`full` 平均 24,557 tokens / 54.0 秒；`no_critic`
   5,714 tokens / 12.2 秒；`no_rag` 23,585 tokens / 52.4 秒；
   `no_memory` 25,914 tokens / 57.7 秒。Critic、重试和 RAG 有明确成本。
5. 本实验每组查询只运行一次，没有人类相关性标注，不能计算“回答质量分数”；
   以上结论仅限可验证的计数、证据、约束和 latency/token。

## 5. 复现命令

```bash
# 完整检索/排序消融（valid，10,073 用户；约 3.5 分钟）
python scripts/run_ablation.py --split valid --users 0 --ranking-users 0

# test（9,258 用户）
python scripts/run_ablation.py --split test --users 0 --ranking-users 0

# Agent 消融（真实 DeepSeek；3 查询 × 4 变体）
python scripts/run_agent_ablation.py

# 单元测试
python -m pytest -q
```

实测测试结果：`144 passed, 69 warnings`；warning 均为 Faiss/Chroma 依赖弃用提示。

输出：

- `logs/phase7_ablation_valid.json`
- `logs/phase7_ablation_test.json`
- `logs/phase7_agent_ablation.json`
- `logs/phase7_agent_ablation_raw.json`

## 6. 限制

1. 召回/排序目标是留出正样本，没有真实曝光和点击，不能计算 CTR。
2. 只评估一个随机种子；bootstrap 只度量给定用户集上的抽样不确定性。
3. 多个比较未做多重检验校正；区间应作为探索性证据，而不是确认性结论。
4. 排序候选池来自 quota 融合；DeepFM/DIN 的对比限定在这个候选池。
5. Agent 消融只有 3 个查询、单次运行，LLM 在 temperature=0.2 下仍有
   波动；不构成统计显著性或线上收益结论。
6. `no_critic` 只是消融变体，不是生产配置。
7. 未做 Docker、线上 A/B 或大规模超参数搜索；这些不属于 Phase 7 已验证范围。
