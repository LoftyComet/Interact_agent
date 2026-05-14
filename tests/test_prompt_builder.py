from gesture_agent.knowledge import KnowledgeBase
from gesture_agent.learning import QuestionParser
from gesture_agent.learning.prompt_builder import build_messages


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
