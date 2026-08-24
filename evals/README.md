# IxDL 回答质量评测集

`ixdl_answer_quality_v1.jsonl` 是根据工作区根目录中的 `用户输入.xlsx` 和
`output_frames_标注.xlsx` 生成并经人工确认策略的第一版评测集，共 33 条。

## 当前状态

- 29 条单轮意图可评测问题；
- 3 条纯语料回答；
- 26 条“语料依据 + 明确标注的设计推导”回答；
- 4 条多轮反馈问题，其中 3 条仍需补齐前置 assistant 回答；
- Excel 中的期望内容暂时标为 `pending_corpus_audit`，不能直接当作已验证的事实金标准。
- `mechanism_registry.json` 已从 `手势词典大纲定稿.xlsx` 核对生成，包含完整的 36 个机制编号。

## 回答类型

- `corpus_only`：所有事实必须直接来自本轮检索语料，并使用有效 `[n]` 引用。
- `corpus_plus_labeled_reasoning`：优先采用书中已有案例或类比；语料不足时，才允许增加单独标注的设计推导块。
- `conversation_feedback`：评测多轮上下文适应、引用补充、长度调整和前后一致性。

## 强约束

1. `[n]` 只允许指向本轮 API 响应 `chunks[n-1]`，且引用片段必须直接支持陈述。
2. 中文术语必须来自 `knowledge_index/terminology.json`；机制中文名、英文名和编号必须匹配 `mechanism_registry.json`。
3. 回答首次提到某个交互机制时必须使用“编号 中文名”，例如 `2-a 拖拽`。
4. API 使用 `answer_blocks[]` 返回“语料依据”和可选的“设计推导（仅供参考）”，前端按 block type 区分样式。
5. 外部产品事实和具体参数不是“合理推导”，在没有独立来源时不能作为确定事实输出。

下一步应补齐每条问题的期望证据 chunk，并审核 Excel 标注中的事实与编号；当前硬约束评分器不把未经审计的内容标准当作事实金标准。

## 运行硬约束评测

配置好 DeepSeek 后，可对指定样本运行本地完整链路：

```bash
.venv/bin/python scripts/evaluate_answer_quality.py \
  --ids ixdl-ui-001 ixdl-ui-014 ixdl-ui-015 \
  --output evals/results/smoke.json
```

评分器检查意图/subtype、`answer_blocks`、引用范围、机制注册表、输出验证器和
grounding 状态。内容覆盖度与“推导是否确有必要”仍需在语料审计完成后加入语义评分。
