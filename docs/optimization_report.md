# 代码质量优化汇报

本文档记录对 `gesture_agent` 的一次系统性优化，涵盖错误处理、意图解析、检索质量、Prompt 构建和测试覆盖五个方向。

---

## 1. 优化目标

随着代码逐步扩展，以下问题开始积累：

- 关键异常被静默吞掉，线上降级无日志可查。
- 置信阈值、评分权重等关键数值散落在多个文件，修改时容易漏改。
- 检索权重是无命名的字面量，无法从文档理解其含义。
- 会话澄清可以无限循环，没有终止保护。
- Prompt 中的术语截断顺序与当前查询无关，浪费上下文窗口。
- 图片、流式输出等边缘路径几乎没有测试。

---

## 2. D. 错误处理

### 2.1 LLM 意图解析异常分类日志

**文件：** `gesture_agent/learning/llm_intent.py:35-47`

**改动前：** 所有异常被一条 `except` 捕获后静默回退，无任何输出。

**改动后：** `SiliconFlowError`（API 连接/鉴权失败）和 `ValueError / json.JSONDecodeError`（响应解析失败）分别记录 `logging.warning`，回退行为不变但现在可查。

```python
except SiliconFlowError as exc:
    logger.warning("LLM intent API call failed, falling back to rule-based: %s", exc)
    return rule_resolution
except (ValueError, json.JSONDecodeError) as exc:
    logger.warning("LLM intent response parse failed, falling back to rule-based: %s", exc)
    return rule_resolution
```

### 2.2 JSON 解析改用 `raw_decode`

**文件：** `gesture_agent/learning/llm_intent.py:112-118`

**改动前：** 用 `text.find("{")` / `text.rfind("}")` 定位边界，当 LLM 在 `reason` 字段中嵌套 JSON 时会截出不合法片段。

**改动后：** 改用 `json.JSONDecoder().raw_decode(text, start)`，正确处理嵌套括号，并增加返回类型校验。

```python
def _parse_json(self, text: str) -> dict[str, Any]:
    start = text.find("{")
    if start < 0:
        raise ValueError("No JSON object found in LLM intent response.")
    obj, _ = json.JSONDecoder().raw_decode(text, start)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object, got {type(obj).__name__}")
    return obj
```

### 2.3 图片处理加文件大小校验

**文件：** `gesture_agent/media/images.py`

**改动前：** 直接 `read_bytes()` 后 base64 编码，4K/RAW 图可超出 API payload 限制或导致 OOM。

**改动后：** 新增 `max_bytes` 参数（默认 10 MB），在读取前用 `stat().st_size` 校验，超限抛出含文件大小信息的 `ValueError`。

```python
MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB

def image_path_to_data_url(path: Union[str, Path], max_bytes: int = MAX_IMAGE_BYTES) -> str:
    ...
    file_size = image_path.stat().st_size
    if file_size > max_bytes:
        raise ValueError(f"Image too large: {file_size / 1024 / 1024:.1f} MB ...")
```

### 2.4 流式输出加中断处理

**文件：** `gesture_agent/cli.py:244-258`

**改动前：** `for delta in client.chat_stream(...)` 无任何中断保护，网络挂起或用户按 Ctrl-C 时行为不可控。

**改动后：** 捕获 `KeyboardInterrupt`，打印 `[已中断]` 后通过 `finally` 块保证换行，不影响已收集内容。

---

## 3. C. 意图解析与会话管理

### 3.1 统一置信阈值常量

**文件：** `gesture_agent/learning/question_parser.py:42-44`

**改动前：** `0.65`（低置信阈值）分别在 `llm_intent.py:54` 和 `question_parser.py:96` 各写了一份，互相独立。

**改动后：** 在 `question_parser.py` 中提取为三个命名常量，`llm_intent.py` 通过 import 引用：

```python
CONFIDENCE_THRESHOLD = 0.65
CLOSE_CALL_MARGIN = 0.15
HIGH_CONFIDENCE_THRESHOLD = 0.9
```

### 3.2 会话澄清最大轮次保护

**文件：** `gesture_agent/learning/session.py`

**改动前：** `ConversationSession` 在 `pending` 状态下可以无限次追加 `collected_details`，没有终止条件。

**改动后：** 新增模块常量 `MAX_CLARIFICATION_ATTEMPTS = 3`，超过次数后清空 `pending` 并提示用户重新描述。

```python
if self.pending.attempts >= MAX_CLARIFICATION_ATTEMPTS:
    self.pending = None
    return SessionResult(status="clarify", message="已多次尝试仍无法确定意图，请重新描述你的问题。", ...)
```

### 3.3 历史长度提为命名常量

**文件：** `gesture_agent/learning/session.py:113`

`self.turns[-6:]` 改为 `self.turns[-MAX_HISTORY_TURNS:]`，`MAX_HISTORY_TURNS = 6` 作为模块常量，修改时不需要查找字面量。

---

## 4. A. 知识检索质量

### 4.1 检索权重提取为命名常量

**文件：** `gesture_agent/knowledge/base.py:22-31`

**改动前：** `search()` 和 `_structured_score()` 中散布着 `12.0`、`5.0`、`8.0`、`0.25`、`0.3` 等字面量，含义不透明。

**改动后：** 全部提取为 `SCORE_*` 命名常量，便于理解和统一调优：

```python
SCORE_TERM_IN_TITLE       = 12.0
SCORE_TERM_IN_TEXT        = 5.0
SCORE_QUERY_TOKEN_IN_TITLE = 2.0
SCORE_QUERY_TOKEN_IN_BODY  = 0.25
SCORE_JACCARD_MULTIPLIER   = 8.0
SCORE_LAYER_MECHANISM_BONUS = 0.3
SCORE_DIC_PENALTY          = 0.4
SCORE_STRUCTURED_EXACT_MATCH   = 8.0
SCORE_STRUCTURED_RELATED_MATCH = 3.0
SCORE_STRUCTURED_TYPE_MATCH    = 1.5
```

### 4.2 `_dedupe` 添加前提注释

**文件：** `gesture_agent/knowledge/base.py:_dedupe`

该方法依赖入参已按分数降序排好的隐含前提，不满足时行为静默错误。新增注释说明这一约束，防止后续重构意外破坏。

### 4.3 `rewrite_query` 扩展一层 `related_terms`

**文件：** `gesture_agent/knowledge/base.py:347-368`

**改动前：** 只展开命中术语的直接字段（aliases / properties / mechanisms 等）。

**改动后：** 收集命中术语的 `related_terms`，再对这些关联术语做一轮额外展开，将其 `term`、`aliases`、`properties`、`mechanisms` 追加到检索 query 中，提升隐式关联术语的召回覆盖。

---

## 5. B. Prompt 构建

### 5.1 chunk 截断改为句子边界

**文件：** `gesture_agent/learning/prompt_builder.py:format_chunk`

**改动前：** `chunk.text[:3600]`，可能在句子中间截断，破坏语义完整性。

**改动后：** 新增 `_truncate_at_sentence(text, limit)` 函数，优先在 `。；\n.;` 等句子边界处截断，仅在找不到合适边界时才做硬截：

```python
def _truncate_at_sentence(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    for sep in ("。", "；", "\n", ".", ";"):
        pos = text.rfind(sep, 0, limit)
        if pos >= limit // 2:
            return text[: pos + 1]
    return text[:limit]
```

### 5.2 term_inventory 按 query 相关性排序

**文件：** `gesture_agent/learning/prompt_builder.py:format_term_inventory`

**改动前：** 术语截断顺序由字典插入顺序决定，与当前查询无关，在截断时容易丢失最相关术语。

**改动后：** 对每个层级/类型的术语列表先按 `(不在 query_terms 中, -len)` 排序，将命中术语置前，再截断到 `max_terms_per_layer`。

---

## 6. E. 测试覆盖

本次新增或扩展了四个测试文件，测试总数从 **47** 增至 **74**。

### 6.1 `tests/test_images.py`（新增，5 个用例）

覆盖 `image_path_to_data_url` 的全部边界路径：

| 用例 | 覆盖点 |
|------|--------|
| `test_valid_png_returns_data_url` | 正常 PNG → data URL，base64 可逆解码验证 |
| `test_missing_file_raises_file_not_found` | 文件不存在 → `FileNotFoundError` |
| `test_non_image_file_raises_value_error` | `.txt` 文件 → `ValueError` |
| `test_oversized_file_raises_value_error` | 超出 `max_bytes` → `ValueError` |
| `test_custom_max_bytes_allows_small_file` | 自定义限制 → 正常通过 |

### 6.2 `tests/test_design_parser.py`（新增，10 个用例）

覆盖 `parse_design_evaluation` 的核心逻辑：

| 用例 | 覆盖点 |
|------|--------|
| `test_detects_control_forms_and_mechanisms` | 旋钮 + 长按拖拽 + 显示反馈提取 |
| `test_detects_risk_for_click_and_drag` | 点拖互斥风险检测 |
| `test_detects_risk_for_single_and_double_click` | 点击缓冲风险检测 |
| `test_short_proposal_reports_missing_info` | 过短描述 → 缺失信息检测 |
| `test_empty_proposal_reports_missing_info` | 空输入 → 缺失信息 |
| `test_image_only_suppresses_length_check` | 提供图片时不触发"描述过短"规则 |
| `test_modality_text_only` | modality 纯文本 |
| `test_modality_image_and_text` | modality 图文混合 |
| `test_extracts_product_context` | 产品上下文提取 |
| `test_risk_incomplete_info_flag` | 信息不完整触发"信息不完整"风险标记 |

### 6.3 `tests/test_llm_intent.py`（扩展，新增 3 个用例）

| 用例 | 覆盖点 |
|------|--------|
| `test_resolver_falls_back_to_rules_on_api_error` | `SiliconFlowError` → 回退到规则，不抛出 |
| `test_resolver_falls_back_to_rules_on_malformed_json` | 无括号响应 → 回退到规则 |
| `test_resolver_handles_nested_braces_in_reason` | reason 字段含嵌套 JSON → 正确解析 |

### 6.4 `tests/test_prompt_builder.py`（扩展，新增 3 个用例）

| 用例 | 覆盖点 |
|------|--------|
| `test_truncate_at_sentence_respects_boundary` | 截断在句子边界，不在字符中间 |
| `test_truncate_at_sentence_short_text_unchanged` | 短文本不被修改 |
| `test_term_inventory_query_terms_appear_first` | 命中术语排在术语列表前半部分 |

---

## 7. 变更文件汇总

| 文件 | 改动类型 |
|------|----------|
| `gesture_agent/learning/llm_intent.py` | 异常分类日志、`_parse_json` 用 `raw_decode`、引用共享阈值常量 |
| `gesture_agent/learning/question_parser.py` | 提取 `CONFIDENCE_THRESHOLD` 等共享常量 |
| `gesture_agent/learning/session.py` | `MAX_CLARIFICATION_ATTEMPTS`、`MAX_HISTORY_TURNS` 常量、最大澄清轮次保护 |
| `gesture_agent/learning/prompt_builder.py` | `_truncate_at_sentence`、term_inventory 按相关性排序 |
| `gesture_agent/knowledge/base.py` | `SCORE_*` 命名常量、`_dedupe` 注释、`rewrite_query` 一层 related_terms 扩展 |
| `gesture_agent/media/images.py` | `MAX_IMAGE_BYTES`、`max_bytes` 参数、文件大小校验 |
| `gesture_agent/cli.py` | 流式输出 `KeyboardInterrupt` 处理 |
| `tests/test_images.py` | 新增，5 个用例 |
| `tests/test_design_parser.py` | 新增，10 个用例 |
| `tests/test_llm_intent.py` | 扩展，新增 3 个用例 |
| `tests/test_prompt_builder.py` | 扩展，新增 3 个用例 |

---

## 8. 验证

```bash
uv run pytest tests/ -q
# 74 passed
```

所有改动均向后兼容，无需修改配置文件或 `.env`。
