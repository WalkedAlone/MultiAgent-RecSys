# Phase 2 ItemCF 评估校正记录

日期：2026-11-18（Phase 3 接入推荐引擎时发现）

## 现象

Phase 2 首次报告 ItemCF 的 valid/test Recall@10 为 0.0706 / 0.0701，
显著高于 Two-Tower / LightGCN。Phase 3 用实际引擎复现时，ItemCF 的
top-100 列表中几乎找不到目标商品，无法得到该数量级的命中。

## 根因

`src/evaluation/retrieval_metrics.py::target_rank` 对并列分数采用
“只统计严格更高分”的乐观策略。ItemCF 的稀疏相似度矩阵中，大量商品分数
恰好为 `0`，表示“没有共现证据”。当某用户的所有 ItemCF 分数都为 `0`
时，目标分数也是 `0`，旧逻辑返回 rank=1，导致这些用户被错误计为命中。

这不是训练或相似度矩阵的问题，而是**评估口径**问题：服务接口
`ItemCFRetriever.retrieve` 只返回正相似度商品，不会把零分商品放进 Top-K。

## 修复

`scripts/train_itemcf.py::evaluate_itemcf` 改为直接评估服务实际返回的
top-100 候选列表：

- 目标在列表中：rank = 列表中的 1-based 位置；
- 目标不在列表中：rank = catalog_size + 1；
- 覆盖率仍按实际推荐列表统计。

未重新训练模型；ItemCF 相似度矩阵和超参数保持不变。

## 校正前后

| Split | 旧 Recall@10 | 校正后 Recall@10 | 旧 Recall@50 | 校正后 Recall@50 |
|---|---:|---:|---:|---:|
| valid | 0.0705847 | 0.0027797 | 0.1824680 | 0.0072471 |
| test | 0.0701015 | 0.0017282 | 0.1818967 | 0.0048607 |

旧指标文件保留在 `logs/metrics_itemcf_*_legacy.json`；
新指标写入 `logs/metrics_itemcf_valid.json`、`logs/metrics_itemcf_test.json`
和 `logs/phase2_itemcf.json`。

## 影响

- “ItemCF 是最强单通道”的结论不再成立。校正后 valid Recall@10：
  Two-Tower 0.0060 > LightGCN 0.0055 > ItemCF 0.0028。
- Phase 3 融合权重改为按校正后的 valid Recall@10 比例校准：
  ItemCF 0.47 / Two-Tower 1.0 / LightGCN 0.92，而不是以 ItemCF 为主。
- Phase 2 的 Two-Tower / LightGCN / DeepFM / DIN 指标不受该问题影响
  （它们的分数不是大量精确零并列；已用各自实际 top-K 接口复核）。
- 该问题仅影响指标解读，不影响 Phase 1 数据划分、特征或模型产物。

## 可复现命令

```bash
python scripts/train_itemcf.py --neighbors 100 --eval-split both
python scripts/evaluate.py --split valid
python scripts/evaluate.py --split test
```
