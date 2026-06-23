from __future__ import annotations

import json
import re
from dataclasses import replace
from typing import TYPE_CHECKING, Optional

from ..core.models import QuestionStructure, StructuredKnowledgeItem, TermInventory
from .models import InputVerificationResult, TermIssue
from .prompts import INPUT_VERIFICATION_PROMPT, INPUT_VERIFICATION_SYSTEM

if TYPE_CHECKING:
    from ..providers import SiliconFlowClient


# intent 属于"术语相关"类时才值得调用 LLM 兜底做语义映射。
TERM_SENSITIVE_INTENTS = {
    "control_form",
    "basic_property",
    "basic_interaction_mechanism",
    "interaction_compare",
    "case_analysis",
    "design_evaluation",
    "mechanism_identification",
    "control_form_compare",
    "function_interaction_breakdown",
    "mechanism_parameter_compare",
    "control_form_application",
    "interaction_optimization",
}

# 这些虚词/高频字若出现在错别字窗口的差异位，说明窗口跨了词边界，
# 属于假命中（如「和摇杆」≈「手摇杆」），直接丢弃。
_FUNCTION_CHARS = set("的了和与跟还有是怎么吗呢啊吧把被在对从向给为would这那哪什么用要会能可不没")


class InputVerifier:
    def __init__(
        self,
        term_inventory: TermInventory,
        structured_items: list[StructuredKnowledgeItem],
        client: Optional[SiliconFlowClient] = None,
        use_llm: bool = False,
    ) -> None:
        self._inventory = term_inventory
        self._structured = structured_items
        self._client = client
        self._use_llm = use_llm and client is not None
        self._all_terms = set(term_inventory.all_terms())
        self._alias_map = dict(term_inventory.aliases)
        # 错别字模糊匹配只针对"枚举正式名"（不含别名键），且长度≥3。
        # 中文里长度 2 的窗口与大量真实词只差 1 字（区别/级别、天气/力气），
        # 等长+长度≥3+距离=1 才能把误报压到接近 0；更短的错别字交给别名表和 LLM。
        canonical_terms: set[str] = set()
        for layer_terms in term_inventory.by_layer.values():
            canonical_terms.update(layer_terms)
        for type_terms in term_inventory.by_type.values():
            canonical_terms.update(type_terms)
        canonical_terms.update(term_inventory.structural_terms)
        self._fuzzy_terms = [t for t in canonical_terms if len(t) >= 3]
        self._mechanism_by_control: dict[str, set[str]] = {}
        self._property_by_mechanism: dict[str, set[str]] = {}
        self._build_indexes()

    def _build_indexes(self) -> None:
        for item in self._structured:
            if item.layer == "control_form" and item.mechanisms:
                self._mechanism_by_control[item.term] = set(item.mechanisms)
            if item.layer == "interaction_mechanism" and item.properties:
                self._property_by_mechanism[item.term] = set(item.properties)

    def verify(self, structure: QuestionStructure) -> InputVerificationResult:
        # 规则层：已提取术语的存在性 + 原始问题上的错别字模糊扫描。
        issues = self._rule_based_check(structure)
        # LLM 兜底：覆盖编辑距离抓不到的语义近义词（如 转盘→旋钮）。
        if self._use_llm and structure.intent in TERM_SENSITIVE_INTENTS:
            issues.extend(self._llm_check(structure))
        issues = self._dedupe_issues(issues)
        if not issues:
            return InputVerificationResult(status="pass")

        # 只有"错别字/近义词映射"（wrong_term）且给得出确定 canonical 的才改写；
        # 机制不兼容、属性矛盾这类不是用词问题，只作信息提示，不改写、不阻断。
        correctable = [
            i for i in issues
            if i.issue_type == "wrong_term" and i.canonical and i.canonical != i.original
        ]
        if not correctable:
            return InputVerificationResult(status="pass", issues=issues)
        corrected = self._apply_corrections(structure, correctable)
        summary = "; ".join(
            f"「{i.original}」→「{i.canonical}」({i.explanation})" for i in correctable
        )
        return InputVerificationResult(
            status="corrected", issues=issues, corrected_structure=corrected, correction_summary=summary
        )

    def _rule_based_check(self, structure: QuestionStructure) -> list[TermIssue]:
        issues: list[TermIssue] = []
        issues.extend(self._check_term_existence(structure.terms))
        issues.extend(self._fuzzy_scan_query(structure))
        issues.extend(self._check_mechanism_compatibility(structure))
        issues.extend(self._check_property_contradiction(structure))
        return issues

    def _fuzzy_scan_query(self, structure: QuestionStructure) -> list[TermIssue]:
        """在原始问题上滑窗找错别字，映射到最接近的枚举词。

        find_terms 只返回精确命中的术语，写错一个字（如「璇钮」）就完全漏掉。
        这里在掩盖掉已识别术语后，对剩余片段做等长、仅差 1 字的紧匹配补齐。
        """
        query = structure.raw_query or ""
        if not query:
            return []
        already = set(structure.terms)
        masked = self._mask_known_spans(query, already)
        hits: dict[str, str] = {}  # window -> canonical
        for segment in masked.split("\x00"):
            for window in self._candidate_windows(segment):
                if window in self._all_terms or window in self._alias_map:
                    continue  # 已是标准词或别名，交给 find_terms
                match = self._closest_fuzzy(window)
                if match and match not in already:
                    hits.setdefault(window, match)
        issues: list[TermIssue] = []
        for window, canonical in hits.items():
            issues.append(TermIssue(
                original=window,
                canonical=canonical,
                issue_type="wrong_term",
                explanation=f"疑似「{canonical}」的误写",
            ))
        return issues

    def _mask_known_spans(self, query: str, found_terms: set[str]) -> str:
        """把已识别的术语/别名在原文里的出现位置替换成分隔符，

        这样错别字扫描不会落进一个已经看懂的词内部（如「位置属性」里的「置属性」）。
        """
        masked = query
        spans = sorted(
            {t for t in found_terms if t}
            | {a for a in self._alias_map if a}
            | {t for t in self._all_terms if len(t) >= 2},
            key=len,
            reverse=True,
        )
        for term in spans:
            if term in masked:
                masked = masked.replace(term, "\x00")
        return masked

    def _candidate_windows(self, query: str) -> list[str]:
        clean = re.sub(r"[\s，。、？?！!,.;；:：（）()【】\[\]\"'`]", "", query)
        windows: list[str] = []
        n = len(clean)
        # 只取等长窗口（长度≥3），与 self._fuzzy_terms 的长度一致；
        # 长度由 _fuzzy_terms 决定，这里覆盖 3-6。
        for size in (3, 4, 5, 6):
            for start in range(0, n - size + 1):
                windows.append(clean[start : start + size])
        return windows

    def _closest_fuzzy(self, window: str) -> Optional[str]:
        """等长、仅替换、且仅差 1 个字 的紧匹配。

        不允许增删（避免「属性」≈「光属性」这类），且差异字若是虚词
        （和/的/吗/怎…），判为跨词边界的假窗口，丢弃（避免「和摇杆」≈「手摇杆」）。
        """
        for term in self._fuzzy_terms:
            if len(term) != len(window):
                continue
            diff_idx = -1
            diff_count = 0
            for i, (a, b) in enumerate(zip(window, term)):
                if a != b:
                    diff_count += 1
                    diff_idx = i
                    if diff_count > 1:
                        break
            if diff_count == 1 and window[diff_idx] not in _FUNCTION_CHARS:
                return term
        return None

    def _dedupe_issues(self, issues: list[TermIssue]) -> list[TermIssue]:
        seen: set[tuple[str, str]] = set()
        result: list[TermIssue] = []
        for issue in issues:
            key = (issue.original, issue.canonical)
            if key in seen:
                continue
            seen.add(key)
            result.append(issue)
        return result

    def _check_term_existence(self, terms: list[str]) -> list[TermIssue]:
        # 不在枚举、也不是别名的术语：只记录、不硬猜最接近词。
        # 编辑距离猜词（如把生造词强行改成某个标准术语）误判率高，已移除；
        # 真正的错别字交给 _fuzzy_scan_query（等长仅差 1 字）和 LLM 兜底处理。
        issues: list[TermIssue] = []
        for term in terms:
            if term in self._all_terms:
                continue
            if term in self._alias_map:
                continue
            issues.append(TermIssue(
                original=term,
                canonical="",
                issue_type="unknown_term",
                explanation=f"术语「{term}」不在词典枚举中",
            ))
        return issues

    def _check_mechanism_compatibility(self, structure: QuestionStructure) -> list[TermIssue]:
        issues: list[TermIssue] = []
        control_terms = [t for t in structure.terms if t in (self._inventory.by_layer.get("control_form") or [])]
        mechanism_terms = [t for t in structure.terms if t in (self._inventory.by_layer.get("interaction_mechanism") or [])]
        for ctrl in control_terms:
            supported = self._mechanism_by_control.get(ctrl)
            if supported is None:
                continue
            for mech in mechanism_terms:
                if mech not in supported:
                    issues.append(TermIssue(
                        original=f"{ctrl}+{mech}",
                        canonical=ctrl,
                        issue_type="impossible_combination",
                        explanation=f"「{ctrl}」不支持「{mech}」机制",
                    ))
        return issues

    def _check_property_contradiction(self, structure: QuestionStructure) -> list[TermIssue]:
        issues: list[TermIssue] = []
        prop_terms = [t for t in structure.terms if t in (self._inventory.by_layer.get("basic_property") or [])]
        if len(prop_terms) < 2:
            return issues
        discrete_props = {"二元属性"}
        continuous_props = {"位置属性", "角度属性", "力属性", "温度属性"}
        has_discrete = any(p in discrete_props for p in prop_terms)
        has_continuous = any(p in continuous_props for p in prop_terms)
        if has_discrete and has_continuous:
            issues.append(TermIssue(
                original=", ".join(prop_terms),
                canonical=prop_terms[0],
                issue_type="contradictory",
                explanation="同时提到了离散属性和连续属性，可能存在矛盾",
            ))
        return issues

    def _apply_corrections(self, structure: QuestionStructure, issues: list[TermIssue]) -> QuestionStructure:
        correction_map = {i.original: i.canonical for i in issues if i.canonical}
        # 把 canonical 并入已提取术语：原词在 terms 里的直接替换；
        # 仅出现在原始问题里的（如错别字窗口）则追加，让检索与回答都用上标准词。
        new_terms = [correction_map.get(t, t) for t in structure.terms]
        for canonical in correction_map.values():
            if canonical not in new_terms:
                new_terms.append(canonical)
        term_corrections = [
            {"original": i.original, "canonical": i.canonical, "explanation": i.explanation}
            for i in issues
            if i.canonical
        ]
        return replace(structure, terms=new_terms, term_corrections=term_corrections)

    def _llm_check(self, structure: QuestionStructure) -> list[TermIssue]:
        if self._client is None:
            return []
        inventory_excerpt = self._format_inventory_excerpt(structure)
        knowledge_excerpt = self._format_knowledge_excerpt(structure)
        prompt = INPUT_VERIFICATION_PROMPT.format(
            term_inventory_excerpt=inventory_excerpt,
            structured_knowledge_excerpt=knowledge_excerpt,
            terms=", ".join(structure.terms),
            query=structure.raw_query,
        )
        try:
            response = self._client.chat(
                [{"role": "system", "content": INPUT_VERIFICATION_SYSTEM}, {"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=400,
            )
            return self._parse_llm_issues(response)
        except Exception:
            return []

    def _parse_llm_issues(self, response: str) -> list[TermIssue]:
        try:
            match = re.search(r"\{.*\}", response, re.DOTALL)
            if not match:
                return []
            data = json.loads(match.group())
            raw_issues = data.get("issues", [])
            return [
                TermIssue(
                    original=item.get("original", ""),
                    canonical=item.get("canonical", ""),
                    issue_type=item.get("issue_type", "ambiguous"),
                    explanation=item.get("explanation", ""),
                )
                for item in raw_issues
                if item.get("original")
            ]
        except (json.JSONDecodeError, KeyError, TypeError):
            return []

    def _format_inventory_excerpt(self, structure: QuestionStructure) -> str:
        # 用户的非标准词通常没被识别成 layer，所以这里按 intent 给出
        # 最相关的枚举清单，而不是只看 structure.layers。
        layer_order: list = []

        def _add(layer) -> None:
            if layer not in layer_order:
                layer_order.append(layer)

        for layer in structure.layers:
            _add(layer)
        intent_layers = {
            "control_form": ["control_form"],
            "basic_property": ["basic_property"],
            "basic_interaction_mechanism": ["interaction_mechanism"],
            "interaction_compare": ["interaction_mechanism", "control_form"],
            "case_analysis": ["control_form", "basic_property", "interaction_mechanism"],
            "design_evaluation": ["control_form", "basic_property", "interaction_mechanism"],
            "mechanism_identification": ["interaction_mechanism"],
            "control_form_compare": ["control_form", "interaction_mechanism"],
            "function_interaction_breakdown": ["control_form", "basic_property", "interaction_mechanism"],
            "mechanism_parameter_compare": ["interaction_mechanism"],
            "control_form_application": ["control_form", "basic_property", "interaction_mechanism"],
            "interaction_optimization": ["control_form", "basic_property", "interaction_mechanism"],
        }
        for layer in intent_layers.get(structure.intent, ["control_form", "basic_property", "interaction_mechanism"]):
            _add(layer)

        lines: list[str] = []
        for layer in layer_order:
            terms = self._inventory.by_layer.get(layer, [])
            if terms:
                lines.append(f"{layer}: {'、'.join(terms)}")
        return "\n".join(lines) if lines else "（无相关术语）"

    def _format_knowledge_excerpt(self, structure: QuestionStructure) -> str:
        relevant = [item for item in self._structured if item.term in structure.terms][:5]
        lines: list[str] = []
        for item in relevant:
            mechs = ", ".join(item.mechanisms[:8]) if item.mechanisms else "无"
            lines.append(f"- {item.term}（{item.term_type}）: mechanisms=[{mechs}]")
        return "\n".join(lines) if lines else "（无相关知识）"
