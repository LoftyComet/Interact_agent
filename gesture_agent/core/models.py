from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Optional


Intent = Literal[
    "basic_interaction_mechanism",
    "advanced_interaction_mechanism",
    "control_form",
    "basic_property",
    "multimodal_interaction",
    "voice_interaction",
    "podcast_content",
    "interaction_compare",
    "background_knowledge",
    "case_analysis",
    "design_evaluation",
]

Layer = Literal[
    "basic_property",
    "interaction_mechanism",
    "control_form",
    "interaction_case",
    "multimodal_interaction",
    "voice_interaction",
    "podcast_content",
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
    output_frame: list[str] = field(default_factory=list)
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
