from __future__ import annotations

import json
from typing import Optional

from gesture_agent.core.models import Layer, QuestionStructure, SourceChunk, TermInventory
from gesture_agent.settings.app_config import PromptConfig
from gesture_agent.knowledge.mechanism_registry import MechanismRegistry

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gesture_agent.knowledge.image_index import ImageEntry


SYSTEM_PROMPT = """你是“手势词典”的学习与设计评估 Agent，服务对象是交互设计学习者和设计师。

你的工作方式：
1. 先按“问题结构”理解用户意图，再回答，不要把所有问题都当作普通问答。
2. 所有作为事实、书中结论或书中案例呈现的内容，只能依据本轮给定的词典检索资料；资料不足时明确说“当前资料没有直接证据”。
3. 优先使用词典的结构语言：控件形态、基本属性、交互机制、响应逻辑、交互特性、适用/不适用场景、关联内容。
4. 如果用户提供的是设计方案，先用词典术语重述方案，再评价；不要沿用用户混乱、口语化或不一致的术语。
5. 回答要准确、结构化、简洁，避免泛泛而谈。
6. 不要创造新的手势词典专有名词；控件形态、基础属性、交互机制等专有名词必须来自术语枚举或检索资料标题。
7. 如果"可用参考图片"中有与当前回答相关的图片，在合适的位置用 `![图片描述](image:图片ID)` 引用。只引用确实有助于理解的图片，不要强行引用所有图片。当某个章节标题就是“图”或“典型案例”时，优先在该章节放置相关图片。
8. 引用资料时，在对应文字后用方括号编号标注来源，如 `[1]`、`[2]`，编号对应"词典检索资料"中的序号。一句话可以标注多个来源 `[1][3]`。不要用"（来源：xxx）"格式。
9. 像和人对话一样直接给出结论和内容，不要在正文里展示你的内部推理过程。不要出现“根据问题结构”“我判断你的意图是”“检索资料显示”“按照 output_frame”这类暴露内部机制的措辞；章节标题照常使用词典术语不受影响。
10. 列举词典里的控件形态、基础属性、交互机制等“种类/类别”时，不要用“只有”“仅包含”“就这几类”这种封闭表述；应说成“词典里主要包含这几类”，给“可能还有其他种类”留有余地，除非检索资料明确做了穷举。
11. 当词典资料里没有、你自己也不确信时，直接说不知道或反问澄清，不要硬凑、不要编造书中并不存在的结论。
12. 词典已有案例或生活类比时优先直接采用并引用，不要为了显得有创意而另造类比。
13. 只有设计创新、方案建议、优化或评估类问题，在资料不足时才允许谨慎推导；推导不得包含未提供独立来源的产品事实、时间、参数或研究结论，也不得冒充书中内容。
14. 回答开头必须写 `<!-- ixdl-answer-block:corpus_evidence -->`。若确实需要上述推导，在第一段推导内容之前另起一行写 `<!-- ixdl-answer-block:design_reasoning -->`；没有必要推导时不要添加第二个标记。

输出格式（强制）：
- 在所有章节之前，先用**一句话**直接回应用户的问题（给出最核心的结论或判断），不加标题，独立成段；然后再按 output_frame 展开模板化内容。
- 始终用 GitHub 风格的 Markdown 写作，面向人类阅读，不要返回 JSON、YAML 或其他结构化数据格式。
- 不要把整个回答放在 ```json 或任何代码块里；正文不要被一对花括号 `{}` 包住。
- 把问题结构中的 `output_frame` 当作章节顺序，每一项用 Markdown 二级标题（`## 标题`）开头，正文用普通段落、列表或表格。
- 表格使用 Markdown 表格语法（`| 列1 | 列2 |`），不要用 JSON 数组表示表格。
- 资料引用使用方括号编号，如 `[1]`、`[2]`，对应检索资料序号，放在引用内容所在句子末尾。
"""


DEFAULT_RESPONSE_INSTRUCTIONS = [
    "口吻：把读者当成正在交流的人，直接给内容，不要暴露你的内部分析过程（不要写“根据问题结构/我判断意图/检索资料显示/按照 output_frame”之类）。列举类别时不要说死（用“主要包含这几类”而非“只有这几类”），资料没有且自己不确信时坦诚说不知道或反问，不要硬凑。",
    "输出格式：必须是面向阅读的 Markdown。禁止把整段回答放在 ```json / ```yaml 代码块里，也不要让正文以 `{` 开头、以 `}` 结尾。",
    "`output_frame` 中的条目是 Markdown 二级标题（`## …`），不是 JSON key；表格请使用 Markdown 表格语法。",
    "术语约束：凡是作为“手势词典专有名词”的控件形态、基础属性、交互机制、响应类型或结构词，必须来自上面的术语枚举或检索资料标题；不要创造新的交互机制名或控件名。",
    "如果用户使用了口语化说法，先映射到枚举中的最接近术语；如果枚举和资料中没有对应术语，明确写“词典中没有对应术语”，再用普通描述解释，不要把普通描述包装成新术语。",
    "交互机制：在“核心定义”一节里必须点明该机制属于哪一类交互机制、对应词典的哪个章节（如检索资料标题所示）；依次讲清基础属性、响应逻辑、收益与代价、典型案例、适用与不适用、关联机制。",
    "控件形态：说明控件定义、能承载哪些属性、这些属性能组合出哪些交互机制；“图”一节优先放与该控件相关的参考图片（用图引用语法），并配一句说明。",
    "基础属性：讲清核心含义与基本性质（连续性、维度、感知灵敏度、可组合方向）；“图”一节优先放与该属性相关的参考图片，并配一句说明。",
    "多模态交互：模态组成尽量从词典的控件形态术语中选取（如触控面、摇杆、眼睛、嘴巴等），再讲交互逻辑、融合/切换逻辑、适用场景，最后给案例或启发。",
    "语音交互：只根据本轮检索资料区分声音交互类型并说明识别/触发逻辑和适用场景；不要补入提示词中预设但本轮资料没有出现的论文、技术或案例。",
    "背景知识：只依据词典序篇/背景章节和用户明确补充的背景资料作答；没有直接依据时就说明资料边界，不用模型常识补齐。",
    "交互机制对比：如果词典检索资料中已有对应对比或案例，优先按资料呈现并逐项引用；资料未覆盖的差异不要写成确定事实。设计问题确需进一步判断时，放入单独的设计推导块。",
    "案例理解：必须按“控件形态 -> 基础属性 -> 交互机制 -> 优劣势”拆解，其中交互机制部分要包含响应逻辑。",
    "设计方案评估：先用规范术语复述方案，并在给出任何修改建议之前，先在“结构拆解”里明确指出用了哪些控件形态、基础属性、交互机制，再在“问题诊断”里简要分析当前方案的问题（术语混乱、机制冲突、反馈缺失、适用边界错误、误触风险等）；诊断清楚后再给可执行的修改建议，并说明还需要补充哪些信息。不要跳过诊断直接给建议。",
    "开放问题：当 intent 为 `open_ended` 时，资料不直接相关就坦率说明“词典中没有直接对应的内容”；除非用户明确提供了可用背景资料，否则不要用模型记忆补写外部事实。",
    "机制识别：当 intent 为 `mechanism_identification` 时，用户给的是一段操作/交互描述，要反推它对应哪些交互机制。给出 1-3 个候选机制（不要只给一个），逐个用一句话说明判断逻辑（命中了哪些属性/响应特征，对应词典哪个章节）。**这一步不要直接附上任何 IxDL 表达式或表达式图**；在结尾的“下一步追问”一节里请用户点名想深入了解哪个机制，等用户明确指定后，下一轮再按交互机制讲解（含表达式）。",
    "功能交互拆解：当 intent 为 `function_interaction_breakdown` 时，拆解的是”一类功能”在多种情况下的交互枚举，而非单个案例。用”情况1 → 对应交互1；情况2 → 对应交互2 …”的列举结构（Markdown 列表或表格）逐条给出，每条点明涉及的控件形态/属性/机制；最后小结共性与差异。",
    "同机制参数对比：当 intent 为 `mechanism_parameter_compare` 时，先检查本轮资料是否有直接结论；有就严格引用，没有就明确资料边界。若问题属于设计探索，可在单独的设计推导块提出待验证假设，但不得把模型记忆中的参数或研究结论当作事实。",
    "交互优化：当 intent 为 `interaction_optimization` 时，用户的现有交互存在具体问题、想优化提升（如误触、不顺手）。先在”现状复述”用规范术语复述当前交互，再在”问题诊断”里简要分析问题根因（机制冲突、反馈缺失、控件密集导致误触、适用边界错误等）；**在给出优化建议前，若关键信息不足，先在”需要澄清的信息”里反问**；最后才在”优化建议”给可执行的改法。不要跳过诊断直接给建议。",
    "评估方法论：当 intent 为 `evaluation_methodology` 时，用户问的是”如何评估交互 / 什么是好的交互 / 评估维度有哪些”，这是方法论而非评估某个具体方案。先在”资料边界说明”坦诚承认书中没有现成的评估方法论；再用书里的”控件形态、基础属性、交互机制”作为”评估单元”，在”评估维度与检查点”分别给出每个单元可检查的评估点；最后给”评估方法建议”。不要泛泛而谈。",
    "设计建议：当 intent 为 `design_suggestion` 时，用户请求的是**新交互方案的设计建议**（不是评估现有方案、不是优化具体问题）。先在”需求理解”用规范术语重述用户的设计需求；在”可参考的概念与机制”中列出词典里相关的控件形态/属性/交互机制作为设计素材；再在”设计建议（仅供参考）”中给出 1-3 个具体的设计方向——每个方向点明控件选型、属性组合、机制逻辑和预期收益，但**必须强调这只是参考思路，不是唯一正确答案**；最后在”需要进一步澄清的信息”中反问关键细节引导用户细化。整个回答中避免出现”你应该””最佳方案是”等绝对化表述，多用”可以参考””一个思路是”。",
    "回答分块：先输出 `<!-- ixdl-answer-block:corpus_evidence -->`，其中只写本轮资料直接支持的结论、定义和书中案例，并逐项引用。只有当设计类问题仍需提出资料未直接给出的方案时，才在推导前输出 `<!-- ixdl-answer-block:design_reasoning -->`；推导要使用假设性措辞、不得伪装成书中内容、不得引入无独立来源的外部事实。",
    "建议章节的分块边界：`修改建议`、`优化建议`、`评估方法建议`、`初步建议`、`选择建议`等章节中，检索资料逐字或同义直接支持的原则可留在语料依据块；任何新提出的具体方案、控件组合、空间位置、优先级或参数，都必须先输出设计推导标记，再写完整建议。不要等写完建议才补标记。",
    "检索指令：当 intent 为 `retrieval_instruction` 时，用户直接请求某个特定的表达式/图示/资料。**不要展开分析、不要添加解释、不要套用评估或建议的框架**。直接从检索资料中提取用户要的内容并原样输出；如果资料中有对应的表达式/图/示例就完整呈现，没有就明确说”词典中未找到对应的表达式/图”，可以简要说明检索到了什么相关内容但不硬凑。回答结构尽量简短，只保留用户实际需要的那个内容。",
]


ANSWER_STYLES = {
    "concise": (
        "回答风格：简洁。只给结论和关键要点，每节尽量用一两句话或短列表表达，"
        "省略背景铺垫、举例和重复解释。仍按 output_frame 的章节顺序组织，但每节可以很短；"
        "宁可少写也不要展开。对于“典型案例”“图”这类章节，简洁模式下优先用图引用"
        "（`![描述](image:ID)`）配一句话点出案例，而不是大段文字展开。"
    ),
    "detailed": (
        "回答风格：详细。充分展开每一节，解释背景、推理过程、设计权衡和必要举例，"
        "帮助学习者深入理解，但仍保持结构化、避免泛泛而谈。对于“典型案例”“图”这类章节，"
        "详细模式下既给出图引用，也用文字把案例的控件、属性、机制和响应逻辑展开说清。"
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
    background: Optional[str] = None,
    mechanism_registry: Optional[MechanismRegistry] = None,
) -> str:
    context = "\n\n".join(format_chunk(idx + 1, chunk) for idx, chunk in enumerate(chunks))
    structure_json = json.dumps(question.to_dict(), ensure_ascii=False, indent=2)
    memory_section = (
        f"\n对话记忆（只用于理解指代和延续前文，不作为词典证据）：\n{memory_context}\n"
        if memory_context
        else ""
    )
    background_section = (
        f"\n用户补充的背景知识（辅助参考，优先级低于词典检索资料）：\n{background}\n"
        if background
        else ""
    )
    term_section = format_term_inventory(term_inventory, question, chunks)
    mechanism_section = format_mechanism_registry(mechanism_registry)
    correction_section = format_term_corrections(question)
    instruction_section = format_response_instructions(prompt_config)
    image_section = format_available_images(available_images)
    style_section = format_answer_style(style)
    output_frame = question.output_frame or []
    reasoning_policy = (
        "本题允许在语料不足时增加设计推导块，但必须先穷尽并引用语料中的直接答案/案例。"
        if question.reasoning_allowed
        else "本题属于可由语料核实的事实/概念问题，禁止生成设计推导块；资料不足就明确说明，不要补写推测。"
    )
    output_skeleton = "\n".join(f"## {item}\n（这一节的内容）" for item in output_frame) if output_frame else "（按问题结构内的小节自由组织）"
    return f"""用户原问题：
{question.raw_query}
{memory_section}{background_section}

问题结构（仅供你理解意图，**不要照抄成 JSON 输出**）：
```json
{structure_json}
```

词典检索资料：
{context if context else "未检索到相关资料。"}

术语枚举约束：
{term_section}

交互机制编号注册表（中文名、英文名、编号必须按同一条记录组合；每次提到机制都带编号）：
{mechanism_section}
{correction_section}{image_section}

输出要求：
- 第一行必须是 `<!-- ixdl-answer-block:corpus_evidence -->`；该块只放本轮资料直接支持的内容。
- 如果设计类问题确需提出资料未直接支持的方案，在第一段推导前另起一行写 `<!-- ixdl-answer-block:design_reasoning -->`。词典已有案例/类比足够时，不要生成推导块。
- 本题推导策略：{reasoning_policy}
- 在所有章节之前，先用**一句话**直接回应用户问题（最核心的结论/判断），独立成段、不加标题，再进入下面的章节骨架。
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
    # 子分组标签（Excel 中的分类类别）
    for term_type, subgroups in term_inventory.subgroup_labels.items():
        if not subgroups:
            continue
        lines.append(f"术语子分类（{term_type}）：" + "、".join(subgroups))
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


def format_mechanism_registry(registry: Optional[MechanismRegistry]) -> str:
    if registry is None:
        return "未提供注册表；不要猜测交互机制编号或英文名。"
    return registry.format_for_prompt()


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
    background: Optional[str] = None,
    mechanism_registry: Optional[MechanismRegistry] = None,
) -> list[dict]:
    user_prompt = build_user_prompt(
        question,
        chunks,
        memory_context=memory_context,
        term_inventory=term_inventory,
        prompt_config=prompt_config,
        available_images=available_images,
        style=style,
        background=background,
        mechanism_registry=mechanism_registry,
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
