from gesture_agent.knowledge import KnowledgeBase
from gesture_agent.learning import QuestionParser
from gesture_agent.learning.prompt_builder import build_messages
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
    assert "基础交互机制：强调基础属性" in prompt
