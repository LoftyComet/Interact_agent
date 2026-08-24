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
            "基础交互逻辑": ["单击", "双击", "长按", "拖拽", "甩动"],
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

    def test_short_typo_in_terms_passes_without_llm(self, term_inventory, structured_items):
        # 2 字错别字（旋纽→旋钮）：编辑距离硬猜已移除，而 _fuzzy_scan_query 只处理
        # len>=3 的窗口（2 字窗口与真实词只差 1 字太多，误报高）。
        # 故不开 LLM 时此类短错别字放行、不再自动纠正——这是"只保留高置信度校正"的取舍。
        verifier = InputVerifier(term_inventory, structured_items)
        structure = _make_structure(["单击", "旋纽"])  # 旋纽 is typo for 旋钮
        result = verifier.verify(structure)
        assert result.status == "pass"

    def test_detect_impossible_combination(self, term_inventory, structured_items):
        verifier = InputVerifier(term_inventory, structured_items)
        structure = _make_structure(["旋钮", "双击"], layers=["control_form", "interaction_mechanism"])
        result = verifier.verify(structure)
        # 机制不兼容只作信息提示，不改写输入：status 为 pass，但 issues 里保留该检测。
        assert result.status == "pass"
        assert any(i.issue_type == "impossible_combination" for i in result.issues)

    def test_detect_property_contradiction(self, term_inventory, structured_items):
        verifier = InputVerifier(term_inventory, structured_items)
        structure = _make_structure(["二元属性", "位置属性"], layers=["basic_property"])
        result = verifier.verify(structure)
        # 属性矛盾同样只作信息提示，不改写输入。
        assert result.status == "pass"
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

    def test_fuzzy_scan_no_false_positive_across_which_properties(self, term_inventory, structured_items):
        # 「哪些属性」中的「些」是疑问限定词，窗口「些属性」不能被误纠为「光属性」。
        verifier = InputVerifier(term_inventory, structured_items)
        structure = QuestionStructure(
            raw_query="旋钮可以承载哪些属性？", intent="control_form",
            layers=["control_form", "basic_property"], terms=["旋钮"], focus=["属性"],
        )
        result = verifier.verify(structure)
        assert result.status == "pass"
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

    def test_unknown_term_not_force_guessed(self, term_inventory, structured_items):
        # 生造/陌生术语不再用编辑距离硬猜成某个标准词：放行、不改写、不产生 canonical。
        verifier = InputVerifier(term_inventory, structured_items)
        structure = _make_structure(["外星科技按钮"], layers=["control_form"])
        result = verifier.verify(structure)
        assert result.status == "pass"
        assert all(not i.canonical for i in result.issues if i.original == "外星科技按钮")

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
        output = "单击是一种离散触发机制。\n\n## 核心定义\n内容\n## 基础属性\n内容\n## 状态/变化序列\n内容\n## 响应逻辑\n内容\n## 适用与不适用\n内容\n## 关联机制\n内容"
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

    def test_detect_out_of_range_citation(self, term_inventory, output_frames, structured_items):
        verifier = OutputVerifier(term_inventory, output_frames, structured_items)
        output = "单击是离散触发机制。[7]\n\n" + "\n".join(
            f"## {title}\n内容" for title in output_frames.frame_for("basic_interaction_mechanism")
        )
        result = verifier.verify(output, _make_structure(["单击"]), source_count=3)

        assert result.should_retry
        assert any(issue.issue_type == "invalid_citation" for issue in result.issues)

    def test_subtype_uses_its_own_output_frame(self, term_inventory, structured_items):
        frames = IntentOutputFrames(
            frames={"interaction_compare": ["共同基础", "核心差异"]},
            subtype_frames={"control_form_compare": ["形态描述", "选择建议"]},
        )
        verifier = OutputVerifier(term_inventory, frames, structured_items)
        structure = QuestionStructure(
            raw_query="按钮和旋钮怎么选",
            intent="interaction_compare",
            subtype="control_form_compare",
            layers=["control_form"],
            terms=["按钮", "旋钮"],
            focus=["适用边界"],
        )

        result = verifier.verify(
            "结论。\n\n## 形态描述\n内容\n\n## 选择建议\n内容",
            structure,
        )

        assert result.status == "pass"
