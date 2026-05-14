from __future__ import annotations

import json
from typing import Optional

from gesture_agent.core.models import Layer, QuestionStructure, SourceChunk, TermInventory


SYSTEM_PROMPT = """你是“手势词典”的学习与设计评估 Agent，服务对象是交互设计学习者和设计师。

你的工作方式：
1. 先按“问题结构”理解用户意图，再回答，不要把所有问题都当作普通问答。
2. 只依据给定的词典资料回答；资料不足时明确说“当前资料没有直接证据”，再给出谨慎推断。
3. 优先使用词典的结构语言：控件形态、基本属性、交互机制、响应逻辑、交互特性、适用/不适用场景、关联内容。
4. 如果用户提供的是设计方案，先用词典术语重述方案，再评价；不要沿用用户混乱、口语化或不一致的术语。
5. 回答要准确、结构化、简洁，避免泛泛而谈。
6. 不要创造新的手势词典专有名词；控件形态、基础属性、交互机制等专有名词必须来自术语枚举或检索资料标题。
"""


def build_user_prompt(
    question: QuestionStructure,
    chunks: list[SourceChunk],
    memory_context: str = "",
    term_inventory: Optional[TermInventory] = None,
) -> str:
    context = "\n\n".join(format_chunk(idx + 1, chunk) for idx, chunk in enumerate(chunks))
    structure_json = json.dumps(question.to_dict(), ensure_ascii=False, indent=2)
    memory_section = (
        f"\n对话记忆（只用于理解指代和延续前文，不作为词典证据）：\n{memory_context}\n"
        if memory_context
        else ""
    )
    term_section = format_term_inventory(term_inventory, question, chunks)
    return f"""用户原问题：
{question.raw_query}
{memory_section}

问题结构：
```json
{structure_json}
```

词典检索资料：
{context if context else "未检索到相关资料。"}

术语枚举约束：
{term_section}

请严格按问题结构中的 output_frame 组织回答，并根据 intent 使用相应的分析口径：
- 术语约束：凡是作为“手势词典专有名词”的控件形态、基础属性、交互机制、响应类型或结构词，必须来自上面的术语枚举或检索资料标题；不要创造新的交互机制名或控件名。
- 如果用户使用了口语化说法，先映射到枚举中的最接近术语；如果枚举和资料中没有对应术语，明确写“词典中没有对应术语”，再用普通描述解释，不要把普通描述包装成新术语。
- 基础交互机制：强调基础属性、状态/变化序列、响应逻辑和适用边界。
- 高级交互机制：强调它解决的组合或冲突问题、判定条件、收益与代价。
- 控件形态：强调控件能承载哪些属性，以及这些属性能组合出哪些交互机制。
- 基础属性：强调连续性、维度、感知灵敏度和可组合方向。
- 多模态/语音交互：强调模态分工、识别逻辑、反馈闭环、风险和替代入口。
- 播客内容：输出适合口播的结构，避免写成论文段落。
- 交互机制对比：优先用表格或清晰分组说明共同基础、核心差异和选择建议。
- 案例理解：必须按“控件形态 -> 基础属性 -> 交互机制 -> 响应逻辑 -> 设计判断”拆解。
- 设计方案评估：先用规范术语复述方案，再按“控件形态 -> 基础属性 -> 交互机制 -> 响应逻辑”拆解；重点指出术语混乱、机制冲突、反馈缺失、适用边界错误，并给出可执行修改建议。"""


def format_chunk(index: int, chunk: SourceChunk) -> str:
    return (
        f"[{index}] {chunk.title}\n"
        f"来源：{chunk.citation()}；层级：{chunk.layer}；匹配分：{chunk.score}\n"
        f"{chunk.text[:3600]}"
    )


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

    lines = [
        "结构术语：" + "、".join(term_inventory.structural_terms),
    ]
    for term_type, terms in term_inventory.by_type.items():
        if not terms:
            continue
        clipped = terms[:max_terms_per_layer]
        suffix = f"（另有 {len(terms) - len(clipped)} 项未列出）" if len(terms) > len(clipped) else ""
        lines.append(f"术语类型-{term_type}：" + "、".join(clipped) + suffix)
    for layer in layers:
        terms = term_inventory.by_layer.get(layer, [])
        if not terms:
            continue
        clipped = terms[:max_terms_per_layer]
        suffix = f"（另有 {len(terms) - len(clipped)} 项未列出）" if len(terms) > len(clipped) else ""
        lines.append(f"{layer}：" + "、".join(clipped) + suffix)
    return "\n".join(lines)


def build_messages(
    question: QuestionStructure,
    chunks: list[SourceChunk],
    image_urls: Optional[list[str]] = None,
    memory_context: str = "",
    term_inventory: Optional[TermInventory] = None,
) -> list[dict]:
    user_prompt = build_user_prompt(
        question,
        chunks,
        memory_context=memory_context,
        term_inventory=term_inventory,
    )
    if not image_urls:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

    content_parts: list[dict] = [{"type": "text", "text": user_prompt}]
    for url in image_urls:
        content_parts.append({"type": "image_url", "image_url": {"url": url, "detail": "high"}})
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content_parts},
    ]
