from gesture_agent.core.models import (
    ConversationTurn,
    PendingClarification,
    QuestionStructure,
    TopicRelation,
    TurnDecision,
)
from gesture_agent.knowledge import KnowledgeBase
from gesture_agent.learning import QuestionParser, TurnClassifier


def _make_classifier(relation_resolver=None) -> TurnClassifier:
    kb = KnowledgeBase.load("data")
    parser = QuestionParser(kb)
    return TurnClassifier(parser, relation_resolver=relation_resolver)


def _turn(classifier: TurnClassifier, query: str, intent: str) -> ConversationTurn:
    structure = classifier.parser.parse(query, forced_intent=intent)
    return ConversationTurn(user_query=query, resolved_query=query, structure=structure)


def test_anaphora_is_follow_up() -> None:
    clf = _make_classifier()
    turns = [_turn(clf, "什么是单击？", "basic_interaction_mechanism")]

    decision = clf.classify("它和长按有什么区别？", turns, None)

    assert decision.relation == TopicRelation.FOLLOW_UP
    assert "anaphora" in decision.signals


def test_new_term_no_overlap_is_new_topic() -> None:
    clf = _make_classifier()
    turns = [_turn(clf, "什么是单击？", "basic_interaction_mechanism")]

    decision = clf.classify("语音交互的唤醒词是什么？", turns, None)

    assert decision.relation == TopicRelation.NEW_TOPIC


def test_design_evaluation_follow_up_carries_intent() -> None:
    clf = _make_classifier()
    turns = [_turn(clf, "请评估这个长按拖拽调音量的方案", "design_evaluation")]

    decision = clf.classify("主要风险有哪些", turns, None)

    assert decision.relation == TopicRelation.FOLLOW_UP
    assert decision.carried_intent == "design_evaluation"


def test_pending_supplement_is_clarify_reply() -> None:
    clf = _make_classifier()
    pending = PendingClarification(
        original_query="帮我对比一下",
        original_intent="interaction_compare",
        original_terms=[],
    )

    decision = clf.classify("单击和长按", turns=[], pending=pending)

    assert decision.relation == TopicRelation.CLARIFY_REPLY


def test_pending_unrelated_new_term_switches_topic() -> None:
    clf = _make_classifier()
    # 原问题真正模糊（intent=None），自带无重合新术语、无指代词 → 切话题。
    pending = PendingClarification(
        original_query="这个怎么用？",
        original_intent=None,
        original_terms=[],
    )

    decision = clf.classify("语音交互为什么需要唤醒词？", turns=[], pending=pending)

    assert decision.relation == TopicRelation.NEW_TOPIC


def test_generic_cue_without_anchor_triggers_llm_fallback() -> None:
    stub = _StubRelationResolver(
        TurnDecision(
            relation=TopicRelation.FOLLOW_UP,
            confidence=0.7,
            reason="stub",
            signals=["llm_fallback"],
        )
    )
    clf = _make_classifier(relation_resolver=stub)
    turns = [_turn(clf, "什么是单击？", "basic_interaction_mechanism")]

    # 无指代词、无术语重合、规则默认置信度 0.5 < 阈值 → 降级到 LLM。
    decision = clf.classify("能不能再展开讲讲", turns, None)

    assert stub.calls == 1
    assert decision.relation == TopicRelation.FOLLOW_UP
    assert "llm_fallback" in decision.signals


def test_llm_fallback_none_falls_back_to_rule_default() -> None:
    stub = _StubRelationResolver(None)
    clf = _make_classifier(relation_resolver=stub)
    turns = [_turn(clf, "什么是单击？", "basic_interaction_mechanism")]

    decision = clf.classify("随便聊点别的吧", turns, None)

    assert stub.calls == 1
    assert decision.relation == TopicRelation.NEW_TOPIC


class _StubRelationResolver:
    def __init__(self, decision) -> None:
        self._decision = decision
        self.calls = 0

    def resolve(self, user_text, turns):
        self.calls += 1
        return self._decision
