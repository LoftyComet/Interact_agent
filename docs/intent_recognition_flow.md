# 意图识别流程

本文档描述手势词典 Agent 当前的意图（intent）识别流程，覆盖从用户输入到产出
`QuestionStructure` 的全过程。涉及的核心代码：

| 文件 | 职责 |
|------|------|
| `gesture_agent/cli.py` | 入口、参数解析、交互循环、单次提问 |
| `gesture_agent/learning/session.py` | 多轮会话、澄清状态机、记忆 |
| `gesture_agent/learning/question_parser.py` | 规则意图识别、术语/层级/焦点抽取 |
| `gesture_agent/learning/llm_intent.py` | LLM 辅助意图判定与回退 |
| `gesture_agent/core/models.py` | `IntentResolution` / `QuestionStructure` 等数据模型 |

## 1. 意图全集

意图定义在 `core/models.py` 的 `Intent` 字面量类型，共 13 类，标签见
`question_parser.py:INTENT_LABELS`：

- `basic_interaction_mechanism` 基础交互机制
- `advanced_interaction_mechanism` 高级交互机制
- `control_form` 控件形态
- `basic_property` 基础属性
- `multimodal_interaction` 多模态交互
- `voice_interaction` 语音交互
- `podcast_content` 播客内容
- `interaction_compare` 交互机制对比
- `background_knowledge` 背景知识
- `case_analysis` 理解交互案例
- `design_evaluation` 设计方案评估
- `dictionary_methodology` 词典方法论
- `open_ended` 开放问题（与词典无关的兜底类别）

## 2. 总体流程

意图识别有两条入口路径，都收敛到同一套「规则识别 + LLM 辅助」内核：

```
                         ┌─────────────────────────────────────────┐
用户输入 ─┬─ 交互式 ──▶  │ ConversationSession.receive()             │
          │             │  · 多轮记忆 / 追问指代补全                  │
          │             │  · 澄清状态机（最多 3 轮）                  │
          │             │  · _carried_intent 延续上一轮意图           │
          │             └───────────────┬───────────────────────────┘
          │                             │
          └─ 单次提问 ─▶ resolve_structure() ──┐
                                          │     │
                                          ▼     ▼
                              ┌───────────────────────────────┐
                              │ IntentResolver.resolve()        │
                              │  规则候选 → (可选)LLM 修正       │
                              └───────────────┬─────────────────┘
                                              │ IntentResolution
                          needs_clarification?─┤
                            是 ──▶ 反问澄清     │ 否
                                              ▼
                              QuestionParser.parse(forced_intent)
                                              │
                                              ▼  QuestionStructure
                                       下游检索 + Prompt 构建
```

核心数据流转：规则层先产出 `IntentResolution`（意图 + 置信度 + 候选列表 +
是否需要澄清），确定意图后再由 `QuestionParser.parse` 产出完整的
`QuestionStructure`（层级、术语、焦点、输出框架等）供下游使用。

## 3. 规则层意图识别（QuestionParser）

`QuestionParser._intent_candidates`（`question_parser.py:137`）是规则层核心，
对每个意图打分后取最高分。两类信号源：

### 3.1 正则信号

为每类意图维护一个关键词正则，命中即赋一个固定分值（`add(intent, score, reason)`，
同一意图取最高分）：

| 正则 | 意图 | 分值 |
|------|------|------|
| `DESIGN_EVALUATION_RE` 评估/评价/方案合理/优化建议 | `design_evaluation` | 0.99 |
| `COMPARE_RE` 对比/区别/差异/vs | `interaction_compare` | 0.98 |
| `PODCAST_RE` 播客/脚本/口播 | `podcast_content` | 0.95 |
| `MULTIMODAL_RE` 多模态/跨模态 | `multimodal_interaction` | 0.95 |
| 提供了图片 | `case_analysis` | 0.94 |
| `DICTIONARY_METHODOLOGY_RE` 词典分类逻辑/编写动机 | `dictionary_methodology` | 0.93 |
| `CASE_RE` 案例/图片/分析/拆解 | `case_analysis` | 0.92 |
| `ADVANCED_MECHANISM_RE` 高级/组合/冲突调和 | `advanced_interaction_mechanism` | 0.9 |
| `CONTROL_FORM_RE` 控件/按钮/摇杆 | `control_form` | 0.88 |
| `VOICE_RE` 语音/声控/口令 | `voice_interaction` | 0.86 |
| `BACKGROUND_RE` 背景/操控力/IxDL | `background_knowledge` | 0.86 |
| `MECHANISM_RE` 点击/长按/拖拽（或命中机制术语） | 基础/高级机制 | 0.86 |
| `PROPERTY_RE` 属性/力属性/形变 | `basic_property` | 0.84 |

基础 vs 高级机制由 `_detect_mechanism_intent` 区分：命中
`ADVANCED_MECHANISM_RE` 或章节标题形如 `3-x`/`4-x` 判为高级，`1-x`/`2-x` 判为基础。

### 3.2 检索信号

调用 `kb.search` 取 top-5，分数 ≥ 3.0 的命中章节按其 `layer` 给对应意图补
0.74 分（`question_parser.py:172`）。这让没有命中正则、但语义落在某章节的问题
也能被归类。

所有候选按分数降序排列，`_detect_intent` 取第一个；无任何候选时兜底为
`background_knowledge`。

## 4. 是否需要澄清（resolve_intent）

`resolve_intent`（`question_parser.py:104`）把候选列表转成 `IntentResolution`，
并判定是否需要反问澄清。三个触发条件（任一满足即 `needs_clarification=True`）：

- **低置信**：最高分 `< CONFIDENCE_THRESHOLD`（0.65）
- **胶着**：存在次优候选，且最高分 `< 0.9`，且与次优分差 `< CLOSE_CALL_MARGIN`（0.15）
- **关键信息缺失**：`_blocking_missing_info` 返回非空

`_blocking_missing_info` 针对特定意图检查阻塞性缺失：

| 意图 | 缺失条件 |
|------|----------|
| `case_analysis` | 无图片且问题文本 < 30 字 |
| `interaction_compare` | 命中术语少于 2 个 |
| `podcast_content` | 问题文本 < 20 字 |
| `design_evaluation` | 缺产品场景/用户目标/交互机制，或既无图片又无可拆解的控件与机制 |

澄清问题由 `_build_clarification_question` 生成：缺信息时按意图给定向追问，
否则列出 top-3 候选让用户选择。

## 5. LLM 辅助层（llm_intent.py）

两个 Resolver 包装规则层，决定是否调用 LLM：

- **`ClarificationIntentResolver`**（默认）：仅当规则层
  `needs_clarification=True` 或 `intent is None` 时才调 LLM；规则足够明确就直接
  返回，省一次 API 调用。对应 CLI 的 `--llm-clarify`（默认开）。
- **`LLMIntentResolver`**：每次都调 LLM 复核。对应 `--llm-intent`。

LLM 调用（`_resolve_with_rule` / `_build_prompt`）把可选意图、对话记忆、用户
输入、规则候选一起喂给模型，要求返回 JSON：`intent` / `confidence` /
`needs_clarification` / `clarification_question` / `reason`。Prompt 中的关键
规则：

1. 追问（「它」「这个」「那它和长按的区别」）必须结合对话记忆补全指代。
2. 上一轮是 `design_evaluation` 时，「这个方案/如何应用/主要风险/怎么改」应延续
   判为 `design_evaluation`，而非机械判为 `case_analysis`。
3. 与词典 12 类都不沾边（闲聊、编程、天气等）才判 `open_ended`，并设
   `needs_clarification=false`。
4. LLM 返回的意图会插到候选列表最前，与规则候选去重后取前 5。

**回退**：LLM API 失败或 JSON 解析失败时，记 warning 并回退到规则层的
`IntentResolution`（`llm_intent.py:47-52`），不会中断流程。

## 6. 多轮会话与澄清状态机（session.py）

交互式模式下 `ConversationSession.receive` 驱动整个流程：

### 6.1 首轮

1. `memory_summary()` 汇总最近 4 轮（intent、术语、输出框架、回答摘要）。
2. `_contextualize_query`：若当前输入命中 `FOLLOW_UP_RE`（它/这个/区别/为什么…）
   或可延续意图，则把对话记忆拼到查询里供 LLM 补全指代。
3. `_carried_intent`：上一轮是 `design_evaluation` 且当前命中
   `DESIGN_EVALUATION_FOLLOW_UP_RE`（这个方案/主要风险/怎么改…），直接强制延续
   该意图，置信度 0.93，跳过重新识别。
4. 否则走 `_try_resolve` → Resolver → 得到 `IntentResolution`。

### 6.2 澄清循环

若结果 `status="clarify"`，会创建 `PendingClarification` 记下原始查询和候选。
后续每轮用户输入被当作补充信息追加（`collected_details`），与原问题拼成
`combined_query` 重新解析。最多尝试 `MAX_CLARIFICATION_ATTEMPTS`（3）次，超过则
放弃并请用户重述。

### 6.3 产出

`status="ready"` 时，`_try_resolve` 用确定的意图调
`QuestionParser.parse(forced_intent=...)` 得到 `QuestionStructure`，并把
`raw_query` 还原为用户原文。`open_ended` 或开启了 `output_frame_resolver` 时，
还会用 LLM 动态生成输出框架覆盖默认模板。

## 7. 单次提问路径

非交互的单次提问走 `resolve_structure`（`cli.py:348`），逻辑更简单：

- `--dry-run` 或未开 `--llm-intent`：直接 `parser.parse`，纯本地规则。
- 开启 `--llm-intent`：`build_intent_resolver` 构造 Resolver，`resolve` 得到
  意图；若 `needs_clarification` 仅在 stderr 打印提示（单次模式不进入澄清循环），
  仍用识别出的意图继续 `parser.parse`。

构造 Resolver 时若 API 客户端不可用（缺 key 等），打印提示并回退到纯规则。

## 8. 最终产出：QuestionStructure

意图确定后，`QuestionParser.parse` 还会补全这些字段（`core/models.py:128`）：

- `layers`：检索命中的层级 + 意图映射层级 + 关键词补充
- `terms`：`kb.find_terms` 命中的术语
- `focus`：定义/属性/逻辑关系/适用边界等焦点（`_detect_focus`）
- `compare_targets`：对比意图下抽取的对比对象
- `case_modality`：text / image
- `output_frame` + `output_frame_source`：输出框架及其来源（static / cot）
- `design_evaluation`：设计评估意图下的结构化拆解

这个 `QuestionStructure` 随后进入下游的知识检索、Prompt 构建与 API 调用，确保
模型在词典框架内作答。


