from gesture_agent.knowledge import KnowledgeBase
from gesture_agent.learning import QuestionParser
from gesture_agent.learning.prompt_builder import (
    _truncate_at_sentence,
    build_messages,
    format_term_inventory,
)
from gesture_agent.settings import PromptConfig


def test_prompt_includes_term_inventory_constraints() -> None:
    kb = KnowledgeBase.load("data")
    parser = QuestionParser(kb)
    structure = parser.parse("请评估这个设计方案：用户长按音量旋钮后拖动来调节音量。")
    chunks = kb.search(structure.raw_query, top_k=3, prefer_terms=structure.terms)

    messages = build_messages(structure, chunks, term_inventory=kb.term_inventory)
    prompt = messages[1]["content"]

    assert "术语枚举约束" in prompt
    assert "不要创造新的交互机制名或控件名" in prompt
    assert "长按拖拽" in prompt
    assert "旋钮" in prompt


def test_prompt_config_can_override_system_and_append_instructions() -> None:
    kb = KnowledgeBase.load("data")
    parser = QuestionParser(kb)
    structure = parser.parse("什么是单击？")

    messages = build_messages(
        structure,
        [],
        prompt_config=PromptConfig(
            system_prompt="自定义系统提示词",
            extra_response_instructions=["自定义回答规则"],
        ),
    )
    prompt = messages[1]["content"]

    assert messages[0]["content"] == "自定义系统提示词"
    assert "自定义回答规则" in prompt
    assert "交互机制：在“核心定义”一节里必须点明该机制属于哪一类" in prompt


def test_truncate_at_sentence_respects_boundary() -> None:
    text = "这是第一句。这是第二句。这是第三句。"
    result = _truncate_at_sentence(text, 10)
    assert result.endswith("。")
    assert len(result) <= 10


def test_truncate_at_sentence_short_text_unchanged() -> None:
    text = "短文本"
    assert _truncate_at_sentence(text, 100) == text


def test_term_inventory_query_terms_appear_first() -> None:
    kb = KnowledgeBase.load("data")
    parser = QuestionParser(kb)
    structure = parser.parse("长按和单击有什么区别？")
    chunks = kb.search(structure.raw_query, top_k=3, prefer_terms=structure.terms)

    section = format_term_inventory(kb.term_inventory, structure, chunks)
    # "长按" and "单击" should appear before generic terms in the layer sections
    # Find the first layer line and check matched terms come early
    lines = section.splitlines()
    for line in lines:
        if "interaction_mechanism" in line or "基础交互机制" in line:
            terms_part = line.split("：", 1)[-1]
            positions = {t: terms_part.find(t) for t in ["长按", "单击"] if t in terms_part}
            # Both matched terms should appear within the first half of the term list
            mid = len(terms_part) // 2
            for term, pos in positions.items():
                assert pos < mid, f"Expected '{term}' in first half, pos={pos}"
            break

