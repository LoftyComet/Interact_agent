from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Literal, Optional


Intent = Literal[
    "basic_interaction_mechanism",
    "control_form",
    "basic_property",
    "multimodal_interaction",
    "voice_interaction",
    "interaction_compare",
    "background_knowledge",
    "case_analysis",
    "design_evaluation",
    "open_ended",
    "mechanism_identification",
    "control_form_compare",
    "function_interaction_breakdown",
    "mechanism_parameter_compare",
    "control_form_application",
    "interaction_optimization",
    "evaluation_methodology",
]

Layer = Literal[
    "basic_property",
    "interaction_mechanism",
    "control_form",
    "interaction_case",
    "multimodal_interaction",
    "voice_interaction",
    "background_knowledge",
    "design_evaluation",
    "unknown",
]


@dataclass
class SourceChunk:
    id: str
    title: str
    source: str
    start_line: int
    end_line: int
    text: str
    layer: Layer = "unknown"
    terms: list[str] = field(default_factory=list)
    score: float = 0.0

    def citation(self) -> str:
        return f"{self.source}:{self.start_line}-{self.end_line}"


@dataclass
class StructuredKnowledgeItem:
    id: str
    term: str
    term_type: str
    layer: Layer
    definition: str = ""
    aliases: list[str] = field(default_factory=list)
    properties: list[str] = field(default_factory=list)
    mechanisms: list[str] = field(default_factory=list)
    control_forms: list[str] = field(default_factory=list)
    response_logic: str = ""
    related_terms: list[str] = field(default_factory=list)
    source: str = ""
    evidence: str = ""


@dataclass
class TermInventory:
    by_layer: dict[Layer, list[str]] = field(default_factory=dict)
    by_type: dict[str, list[str]] = field(default_factory=dict)
    aliases: dict[str, str] = field(default_factory=dict)
    structural_terms: list[str] = field(default_factory=list)
    subgroup_labels: dict[str, list[str]] = field(default_factory=dict)
    source: str = "generated"

    def all_terms(self) -> list[str]:
        terms: list[str] = []
        for term in self.structural_terms:
            if term not in terms:
                terms.append(term)
        for layer_terms in self.by_layer.values():
            for term in layer_terms:
                if term not in terms:
                    terms.append(term)
        for type_terms in self.by_type.values():
            for term in type_terms:
                if term not in terms:
                    terms.append(term)
        for subgroup_list in self.subgroup_labels.values():
            for term in subgroup_list:
                if term not in terms:
                    terms.append(term)
        for alias, canonical in self.aliases.items():
            if alias not in terms:
                terms.append(alias)
            if canonical not in terms:
                terms.append(canonical)
        return terms


@dataclass
class IntentOutputFrames:
    frames: dict[Intent, list[str]] = field(default_factory=dict)
    source: str = "defaults"

    def frame_for(self, intent: Intent) -> list[str]:
        if intent == "open_ended":
            return list(self.frames.get("open_ended", []))
        frame = self.frames.get(intent)
        if frame is None:
            raise KeyError(f"Missing output frame for intent `{intent}`.")
        return list(frame)


@dataclass
class DesignEvaluationStructure:
    raw_proposal: str
    modality: list[str]
    product_context: str = ""
    user_goal: str = ""
    control_forms: list[str] = field(default_factory=list)
    basic_properties: list[str] = field(default_factory=list)
    mechanisms: list[str] = field(default_factory=list)
    system_feedback: list[str] = field(default_factory=list)
    risk_points: list[str] = field(default_factory=list)
    missing_info: list[str] = field(default_factory=list)


@dataclass
class QuestionStructure:
    raw_query: str
    intent: Intent
    layers: list[Layer]
    terms: list[str]
    focus: list[str]
    compare_targets: list[str] = field(default_factory=list)
    case_modality: list[str] = field(default_factory=list)
    missing_info: list[str] = field(default_factory=list)
    term_corrections: list[dict[str, str]] = field(default_factory=list)
    output_frame: list[str] = field(default_factory=list)
    output_frame_source: str = "static"
    design_evaluation: Optional[DesignEvaluationStructure] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class IntentCandidate:
    intent: Intent
    score: float
    reason: str


@dataclass
class IntentResolution:
    intent: Optional[Intent]
    confidence: float
    candidates: list[IntentCandidate] = field(default_factory=list)
    needs_clarification: bool = False
    clarification_question: str = ""
    missing_info: list[str] = field(default_factory=list)


@dataclass
class SessionResult:
    status: Literal["clarify", "ready"]
    message: str = ""
    user_query: str = ""
    resolved_query: str = ""
    memory_context: str = ""
    structure: Optional[QuestionStructure] = None
    resolution: Optional[IntentResolution] = None
    # 结构化追问选项：非空时，前端把它们渲染成可点击按钮（点击即回填 value 作为下一轮输入）。
    options: list[dict[str, str]] = field(default_factory=list)


class TopicRelation(str, Enum):
    """当前输入与历史对话的话题关系（维度 A：输入侧）。

    与「可答性」（维度 B：信息是否充分）正交——本枚举只回答“这句话和上文什么关系”，
    不回答“能不能直接作答”。
    """

    FOLLOW_UP = "follow_up"  # 延续上一个已答完的话题，应注入对话记忆
    NEW_TOPIC = "new_topic"  # 全新话题，不注入记忆；若处于澄清态则放弃 pending
    CLARIFY_REPLY = "clarify_reply"  # 对当前澄清问题的回答（仅在 pending 时可能）


@dataclass
class TurnDecision:
    """TurnClassifier 对一轮输入的话题关系判定结果。"""

    relation: TopicRelation
    confidence: float
    reason: str
    # 命中了哪些信号，便于调试与测试。
    signals: list[str] = field(default_factory=list)
    # 追问需沿用的上一轮意图（如 design_evaluation 延续）；无则 None。
    carried_intent: Optional[Intent] = None


@dataclass
class ConversationTurn:
    """一轮已完成的问答，构成会话短期记忆。"""

    user_query: str
    resolved_query: str
    structure: QuestionStructure
    answer_summary: str = ""


@dataclass
class PendingClarification:
    """一次尚未解决的澄清状态。"""

    original_query: str
    candidates: list[IntentCandidate] = field(default_factory=list)
    collected_details: list[str] = field(default_factory=list)
    attempts: int = 0
    last_question: str = ""
    question_history: list[str] = field(default_factory=list)
    memory_context: str = ""
    original_intent: Optional[Intent] = None
    original_terms: list[str] = field(default_factory=list)

    def combined_query(self) -> str:
        if not self.collected_details:
            return self.original_query
        details = "；".join(self.collected_details)
        return f"{self.original_query}\n补充信息：{details}"
