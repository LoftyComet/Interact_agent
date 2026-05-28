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
