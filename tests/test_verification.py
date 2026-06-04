"""Tests for the verification package (InputVerifier + OutputVerifier)."""

from __future__ import annotations

import pytest

from gesture_agent.core.models import (
    IntentOutputFrames,
    QuestionStructure,
    StructuredKnowledgeItem,
    TermInventory,
)
from gesture_agent.verification import InputVerifier, OutputVerifier


@pytest.fixture
def term_inventory() -> TermInventory:
    return TermInventory(
        by_layer={
            "interaction_mechanism": ["单击", "双击", "长按", "拖拽", "甩动"],
            "control_form": ["按钮", "旋钮", "滚轮", "触控面"],
            "basic_property": ["二元属性", "位置属性", "力属性", "角度属性"],
        },
        by_type={
            "基础交互机制": ["单击", "双击", "长按", "拖拽", "甩动"],
            "控件形态": ["按钮", "旋钮", "滚轮", "触控面"],
            "基础属性": ["二元属性", "位置属性", "力属性", "角度属性"],
        },
        aliases={"点一下": "单击", "按住": "长按", "拖动": "拖拽"},
        structural_terms=["控件形态", "基础属性", "交互机制"],
    )


@pytest.fixture
def structured_items() -> list[StructuredKnowledgeItem]:
    return [
        StructuredKnowledgeItem(
            id="test:1", term="旋钮", term_type="控件形态", layer="control_form",
            mechanisms=["长按", "拖拽"],
        ),
        StructuredKnowledgeItem(
            id="test:2", term="按钮", term_type="控件形态", layer="control_form",
            mechanisms=["单击", "双击", "长按"],
        ),
        StructuredKnowledgeItem(
            id="test:3", term="二元属性", term_type="基础属性", layer="basic_property",
            properties=["二元属性"],
        ),
    ]


@pytest.fixture
def output_frames() -> IntentOutputFrames:
    return IntentOutputFrames(
        frames={
            "basic_interaction_mechanism": ["核心定义", "基础属性", "状态/变化序列", "响应逻辑", "适用与不适用", "关联机制"],
            "control_form": ["控件定义", "可用属性", "可承载的交互机制", "典型案例", "设计注意点"],
        }
    )


def _make_structure(terms: list[str], intent: str = "basic_interaction_mechanism", layers=None) -> QuestionStructure:
    return QuestionStructure(
        raw_query="test query",
        intent=intent,
        layers=layers or ["interaction_mechanism"],
        terms=terms,
        focus=["定义"],
    )


class TestInputVerifier:
    def test_pass_valid_terms(self, term_inventory, structured_items):
        verifier = InputVerifier(term_inventory, structured_items)
        structure = _make_structure(["单击", "按钮"])
        result = verifier.verify(structure)
        assert result.status == "pass"

    def test_detect_wrong_term(self, term_inventory, structured_items):
        verifier = InputVerifier(term_inventory, structured_items)
        structure = _make_structure(["单击", "旋纽"])  # 旋纽 is typo for 旋钮
        result = verifier.verify(structure)
        assert result.status == "corrected"
        assert any(i.original == "旋纽" for i in result.issues)
        assert result.corrected_structure is not None
        assert "旋钮" in result.corrected_structure.terms

    def test_detect_impossible_combination(self, term_inventory, structured_items):
        verifier = InputVerifier(term_inventory, structured_items)
        structure = _make_structure(["旋钮", "双击"], layers=["control_form", "interaction_mechanism"])
        result = verifier.verify(structure)
        assert result.status == "corrected"
        assert any(i.issue_type == "impossible_combination" for i in result.issues)

    def test_detect_property_contradiction(self, term_inventory, structured_items):
        verifier = InputVerifier(term_inventory, structured_items)
        structure = _make_structure(["二元属性", "位置属性"], layers=["basic_property"])
        result = verifier.verify(structure)
        assert result.status == "corrected"
        assert any(i.issue_type == "contradictory" for i in result.issues)

    def test_alias_resolves(self, term_inventory, structured_items):
        verifier = InputVerifier(term_inventory, structured_items)
        structure = _make_structure(["点一下"])  # alias for 单击
        result = verifier.verify(structure)
        assert result.status == "pass"

    def test_fuzzy_scan_corrects_typo_in_raw_query(self, term_inventory, structured_items):
        # 用户把「触控面」写成「触控免」，find_terms 完全漏掉，靠原文模糊扫描补齐。
        verifier = InputVerifier(term_inventory, structured_items)
        structure = QuestionStructure(
            raw_query="触控免怎么用", intent="control_form",
            layers=["control_form"], terms=[], focus=["定义"],
        )
        result = verifier.verify(structure)
        assert result.status == "corrected"
        assert "触控面" in result.corrected_structure.terms
        assert any(c["canonical"] == "触控面" for c in result.corrected_structure.term_corrections)

    def test_fuzzy_scan_no_false_positive_on_known_term(self, term_inventory, structured_items):
        # 「位置属性」是已识别术语，其子串「置属性」不应被误纠成别的属性。
        verifier = InputVerifier(term_inventory, structured_items)
        structure = QuestionStructure(
            raw_query="力属性和位置属性矛盾吗", intent="basic_property",
            layers=["basic_property"], terms=["力属性", "位置属性"], focus=["定义"],
        )
        result = verifier.verify(structure)
        # 仅有矛盾检测（离散 vs 连续）不应被触发，因为两者都是连续属性；
        # 关键是不能出现任何 wrong_term 误纠。
        assert not any(i.issue_type == "wrong_term" for i in result.issues)

    def test_unmappable_word_passes_without_clarification(self, term_inventory, structured_items):
        # 无法映射到任一枚举的输入照常放行，不再返回 needs_clarification。
        verifier = InputVerifier(term_inventory, structured_items)
        structure = QuestionStructure(
            raw_query="今天天气怎么样", intent="background_knowledge",
            layers=["background_knowledge"], terms=[], focus=["定义"],
        )
        result = verifier.verify(structure)
        assert result.status == "pass"

    def test_llm_fallback_maps_synonym(self, term_inventory, structured_items):
        # 编辑距离抓不到的语义近义词（转盘→旋钮）由 LLM 兜底映射。
        class FakeClient:
            def chat(self, messages, **kwargs):
                return '{"issues": [{"original": "转盘", "canonical": "旋钮", "issue_type": "wrong_term", "explanation": "转盘指旋钮"}]}'

        verifier = InputVerifier(
            term_inventory, structured_items, client=FakeClient(), use_llm=True
        )
        structure = QuestionStructure(
            raw_query="这个转盘怎么操作", intent="control_form",
            layers=["control_form"], terms=[], focus=["定义"],
        )
        result = verifier.verify(structure)
        assert result.status == "corrected"
        assert "旋钮" in result.corrected_structure.terms
        assert any(c["original"] == "转盘" for c in result.corrected_structure.term_corrections)


class TestOutputVerifier:
    def test_pass_complete_output(self, term_inventory, output_frames, structured_items):
        verifier = OutputVerifier(term_inventory, output_frames, structured_items)
        output = "## 核心定义\n内容\n## 基础属性\n内容\n## 状态/变化序列\n内容\n## 响应逻辑\n内容\n## 适用与不适用\n内容\n## 关联机制\n内容"
        structure = _make_structure(["单击"])
        result = verifier.verify(output, structure)
        assert result.status == "pass"

    def test_detect_missing_section(self, term_inventory, output_frames, structured_items):
        verifier = OutputVerifier(term_inventory, output_frames, structured_items)
        output = "## 核心定义\n内容\n## 基础属性\n内容"
        structure = _make_structure(["单击"])
        result = verifier.verify(output, structure)
        assert result.status == "issues_found"
        assert result.should_retry
        missing = [i for i in result.issues if i.issue_type == "missing_section"]
        assert len(missing) >= 1

    def test_detect_code_block_wrapping(self, term_inventory, output_frames, structured_items):
        verifier = OutputVerifier(term_inventory, output_frames, structured_items)
        output = "```markdown\n## 核心定义\n内容\n```"
        structure = _make_structure(["单击"])
        result = verifier.verify(output, structure)
        assert any(i.issue_type == "format_error" for i in result.issues)

    def test_correction_hints_built(self, term_inventory, output_frames, structured_items):
        verifier = OutputVerifier(term_inventory, output_frames, structured_items)
        output = "一些没有标题的纯文本回答"
        structure = _make_structure(["单击"])
        result = verifier.verify(output, structure)
        assert result.should_retry
        assert result.correction_hints
