from __future__ import annotations

import json
from typing import Optional

from gesture_agent.core.models import Layer, QuestionStructure, SourceChunk, TermInventory
from gesture_agent.settings.app_config import PromptConfig

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gesture_agent.knowledge.image_index import ImageEntry


SYSTEM_PROMPT = """你是“手势词典”的学习与设计评估 Agent，服务对象是交互设计学习者和设计师。

你的工作方式：
1. 先按“问题结构”理解用户意图，再回答，不要把所有问题都当作普通问答。
2. 只依据给定的词典资料回答；资料不足时明确说“当前资料没有直接证据”，再给出谨慎推断。
3. 优先使用词典的结构语言：控件形态、基本属性、交互机制、响应逻辑、交互特性、适用/不适用场景、关联内容。
4. 如果用户提供的是设计方案，先用词典术语重述方案，再评价；不要沿用用户混乱、口语化或不一致的术语。
5. 回答要准确、结构化、简洁，避免泛泛而谈。
6. 不要创造新的手势词典专有名词；控件形态、基础属性、交互机制等专有名词必须来自术语枚举或检索资料标题。
7. 如果"可用参考图片"中有与当前回答相关的图片，在合适的位置用 `![图片描述](image:图片ID)` 引用。只引用确实有助于理解的图片，不要强行引用所有图片。
8. 引用资料时，在对应文字后用方括号编号标注来源，如 `[1]`、`[2]`，编号对应"词典检索资料"中的序号。一句话可以标注多个来源 `[1][3]`。不要用"（来源：xxx）"格式。

输出格式（强制）：
- 始终用 GitHub 风格的 Markdown 写作，面向人类阅读，不要返回 JSON、YAML 或其他结构化数据格式。
- 不要把整个回答放在 ```json 或任何代码块里；正文不要被一对花括号 `{}` 包住。
- 把问题结构中的 `output_frame` 当作章节顺序，每一项用 Markdown 二级标题（`## 标题`）开头，正文用普通段落、列表或表格。
- 表格使用 Markdown 表格语法（`| 列1 | 列2 |`），不要用 JSON 数组表示表格。
- 资料引用使用方括号编号，如 `[1]`、`[2]`，对应检索资料序号，放在引用内容所在句子末尾。
"""


DEFAULT_RESPONSE_INSTRUCTIONS = [
    "输出格式：必须是面向阅读的 Markdown。禁止把整段回答放在 ```json / ```yaml 代码块里，也不要让正文以 `{` 开头、以 `}` 结尾。",
    "`output_frame` 中的条目是 Markdown 二级标题（`## …`），不是 JSON key；表格请使用 Markdown 表格语法。",
    "术语约束：凡是作为“手势词典专有名词”的控件形态、基础属性、交互机制、响应类型或结构词，必须来自上面的术语枚举或检索资料标题；不要创造新的交互机制名或控件名。",
    "如果用户使用了口语化说法，先映射到枚举中的最接近术语；如果枚举和资料中没有对应术语，明确写“词典中没有对应术语”，再用普通描述解释，不要把普通描述包装成新术语。",
    "基础交互机制：强调基础属性、状态/变化序列、响应逻辑和适用边界。",
    "高级交互机制：强调它解决的组合或冲突问题、判定条件、收益与代价。",
    "控件形态：强调控件能承载哪些属性，以及这些属性能组合出哪些交互机制。",
    "基础属性：强调连续性、维度、感知灵敏度和可组合方向。",
    "多模态/语音交互：强调模态分工、识别逻辑、反馈闭环、风险和替代入口。",
    "播客内容：输出适合口播的结构，避免写成论文段落。",
    "交互机制对比：如果词典检索资料中已有对应的对比内容，严格按照词典原文呈现，不要改写或补充；如果对比对象超出词典已有内容，AI 自行理解并给出恰当描述，但必须参考检索资料中已有对比章节的格式（Markdown 表格 + 共同基础/核心差异/适用边界的分组结构）。",
    "案例理解：必须按“控件形态 -> 基础属性 -> 交互机制 -> 响应逻辑 -> 设计判断”拆解。",
    "设计方案评估：先用规范术语复述方案，再按“控件形态 -> 基础属性 -> 交互机制 -> 响应逻辑”拆解；重点指出术语混乱、机制冲突、反馈缺失、适用边界错误，并给出可执行修改建议。",
    "词典方法论：当用户问“为什么这样划分/这本书的意义/切入点”时，回答词典自身的分类逻辑与编写立场。先识别提问切入点（划分逻辑？写作动机？取材范围？），再说明词典从“控件形态 / 基础属性 / 交互机制 / 响应逻辑”切入的理由，对比常见 UI 教材或 HCI 综述的差异，落到学习者收益。这一类问题不要陷入具体机制讲解，要保持元层视角。",
    "开放问题：当 intent 为 `open_ended` 时，问题不属于词典预定义类别，词典检索资料只是弱参考。如果资料不直接相关，可以坦率说明“词典中没有直接对应的内容”，再基于通用知识给出回答；仍按动态生成的 `output_frame` 顺序作为 Markdown 二级标题组织内容，但不必强求使用词典专有术语。",
]


ANSWER_STYLES = {
    "concise": (
        "回答风格：简洁。只给结论和关键要点，每节尽量用一两句话或短列表表达，"
        "省略背景铺垫、举例和重复解释。仍按 output_frame 的章节顺序组织，但每节可以很短；"
        "宁可少写也不要展开。"
    ),
    "detailed": (
        "回答风格：详细。充分展开每一节，解释背景、推理过程、设计权衡和必要举例，"
        "帮助学习者深入理解，但仍保持结构化、避免泛泛而谈。"
    ),
}
DEFAULT_ANSWER_STYLE = "concise"


def format_answer_style(style: Optional[str]) -> str:
    key = (style or DEFAULT_ANSWER_STYLE).strip().lower()
    return ANSWER_STYLES.get(key, ANSWER_STYLES[DEFAULT_ANSWER_STYLE])


def build_user_prompt(
    question: QuestionStructure,
    chunks: list[SourceChunk],
    memory_context: str = "",
    term_inventory: Optional[TermInventory] = None,
    prompt_config: Optional[PromptConfig] = None,
    available_images: Optional[list["ImageEntry"]] = None,
    style: Optional[str] = None,
) -> str:
    context = "\n\n".join(format_chunk(idx + 1, chunk) for idx, chunk in enumerate(chunks))
    structure_json = json.dumps(question.to_dict(), ensure_ascii=False, indent=2)
    memory_section = (
        f"\n对话记忆（只用于理解指代和延续前文，不作为词典证据）：\n{memory_context}\n"
        if memory_context
        else ""
    )
    term_section = format_term_inventory(term_inventory, question, chunks)
    correction_section = format_term_corrections(question)
    instruction_section = format_response_instructions(prompt_config)
    image_section = format_available_images(available_images)
    style_section = format_answer_style(style)
    output_frame = question.output_frame or []
    output_skeleton = "\n".join(f"## {item}\n（这一节的内容）" for item in output_frame) if output_frame else "（按问题结构内的小节自由组织）"
    return f"""用户原问题：
{question.raw_query}
{memory_section}

问题结构（仅供你理解意图，**不要照抄成 JSON 输出**）：
```json
{structure_json}
```

词典检索资料：
{context if context else "未检索到相关资料。"}

术语枚举约束：
{term_section}
{correction_section}{image_section}

输出要求：
- 必须返回 Markdown 正文，禁止把回答整体包成 JSON / YAML / 代码块。
- 按问题结构中的 `output_frame` 顺序作为 Markdown 二级标题（`##`），每节下用段落、列表或 Markdown 表格展开。
- {style_section}
- 章节骨架（请严格按以下顺序产出，每节至少有一段内容）：

{output_skeleton}

并根据 intent 使用相应的分析口径：
{instruction_section}"""


def format_chunk(index: int, chunk: SourceChunk, max_chars: int = 3600) -> str:
    text = _truncate_at_sentence(chunk.text, max_chars)
    return (
        f"[{index}] {chunk.title}\n"
        f"来源：{chunk.citation()}；层级：{chunk.layer}；匹配分：{chunk.score}\n"
        f"{text}"
    )


def _truncate_at_sentence(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    # Try to cut at a sentence boundary (Chinese or Latin punctuation)
    for sep in ("。", "；", "\n", ".", ";"):
        pos = text.rfind(sep, 0, limit)
        if pos >= limit // 2:
            return text[: pos + 1]
    return text[:limit]


def format_term_inventory(
    term_inventory: Optional[TermInventory],
    question: QuestionStructure,
    chunks: list[SourceChunk],
    max_terms_per_layer: int = 80,
) -> str:
    if term_inventory is None:
        return "未提供全局术语枚举；只能使用问题结构和检索资料标题中的术语。"

    layers: list[Layer] = []
    for layer in question.layers:
        if layer not in layers:
            layers.append(layer)
    for chunk in chunks:
        if chunk.layer not in layers:
            layers.append(chunk.layer)

    query_terms = set(question.terms)

    def _rank_terms(terms: list[str]) -> list[str]:
        """Sort terms so query-matched ones appear first, then by length descending."""
        return sorted(terms, key=lambda t: (t not in query_terms, -len(t)))

    lines = [
        "结构术语：" + "、".join(term_inventory.structural_terms),
    ]
    for term_type, terms in term_inventory.by_type.items():
        if not terms:
            continue
        ranked = _rank_terms(terms)
        clipped = ranked[:max_terms_per_layer]
        suffix = f"（另有 {len(terms) - len(clipped)} 项未列出）" if len(terms) > len(clipped) else ""
        lines.append(f"术语类型-{term_type}：" + "、".join(clipped) + suffix)
    for layer in layers:
        terms = term_inventory.by_layer.get(layer, [])
        if not terms:
            continue
        ranked = _rank_terms(terms)
        clipped = ranked[:max_terms_per_layer]
        suffix = f"（另有 {len(terms) - len(clipped)} 项未列出）" if len(terms) > len(clipped) else ""
        lines.append(f"{layer}：" + "、".join(clipped) + suffix)
    return "\n".join(lines)


def format_response_instructions(prompt_config: Optional[PromptConfig] = None) -> str:
    instructions = list(DEFAULT_RESPONSE_INSTRUCTIONS)
    if prompt_config and prompt_config.response_instructions is not None:
        instructions = list(prompt_config.response_instructions)
    if prompt_config:
        instructions.extend(prompt_config.extra_response_instructions)
    return "\n".join(f"- {item}" for item in instructions)


def format_term_corrections(question: QuestionStructure) -> str:
    corrections = getattr(question, "term_corrections", None) or []
    if not corrections:
        return ""
    lines = ["\n术语纠正（已把用户的口语/近义/误写词对齐到枚举，请用纠正后的标准术语作答）："]
    for item in corrections:
        original = item.get("original", "")
        canonical = item.get("canonical", "")
        explanation = item.get("explanation", "")
        suffix = f"（{explanation}）" if explanation else ""
        lines.append(f"- 用户说「{original}」→ 按「{canonical}」理解{suffix}")
    lines.append(
        "请在回答开头用一句话向用户点明这一映射（如「这里按词典术语，"
        "你说的X对应Y」），再继续基于标准术语回答。"
    )
    return "\n".join(lines)


def format_available_images(images: Optional[list["ImageEntry"]]) -> str:
    if not images:
        return ""
    lines = ["\n可用参考图片（在回答中合适位置用 ![描述](image:ID) 引用有助理解的图片）："]
    for img in images:
        lines.append(f"- image:{img.id} — {img.annotation}（章节：{img.heading}）")
    return "\n".join(lines)


def resolve_system_prompt(prompt_config: Optional[PromptConfig] = None) -> str:
    if prompt_config is None:
        return SYSTEM_PROMPT
    system_prompt = prompt_config.system_prompt or SYSTEM_PROMPT
    if prompt_config.extra_system_prompt:
        return f"{system_prompt.rstrip()}\n\n{prompt_config.extra_system_prompt.strip()}\n"
    return system_prompt


def build_messages(
    question: QuestionStructure,
    chunks: list[SourceChunk],
    image_urls: Optional[list[str]] = None,
    memory_context: str = "",
    term_inventory: Optional[TermInventory] = None,
    prompt_config: Optional[PromptConfig] = None,
    available_images: Optional[list["ImageEntry"]] = None,
    style: Optional[str] = None,
) -> list[dict]:
    user_prompt = build_user_prompt(
        question,
        chunks,
        memory_context=memory_context,
        term_inventory=term_inventory,
        prompt_config=prompt_config,
        available_images=available_images,
        style=style,
    )
    system_prompt = resolve_system_prompt(prompt_config)
    if not image_urls:
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    content_parts: list[dict] = [{"type": "text", "text": user_prompt}]
    for url in image_urls:
        content_parts.append({"type": "image_url", "image_url": {"url": url, "detail": "high"}})
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content_parts},
    ]
