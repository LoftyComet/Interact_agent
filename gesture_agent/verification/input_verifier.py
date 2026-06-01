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
        issues = self._rule_based_check(structure)
        if not issues and self._use_llm and structure.intent in ("design_evaluation", "case_analysis"):
            issues = self._llm_check(structure)
        if not issues:
            return InputVerificationResult(status="pass")
        correctable = [i for i in issues if i.canonical]
        if not correctable and issues:
            summary = "; ".join(i.explanation for i in issues)
            return InputVerificationResult(
                status="needs_clarification", issues=issues, correction_summary=summary
            )
        corrected = self._apply_corrections(structure, correctable)
        summary = "; ".join(f"「{i.original}」→「{i.canonical}」({i.explanation})" for i in correctable)
        return InputVerificationResult(
            status="corrected", issues=issues, corrected_structure=corrected, correction_summary=summary
        )

    def _rule_based_check(self, structure: QuestionStructure) -> list[TermIssue]:
        issues: list[TermIssue] = []
        issues.extend(self._check_term_existence(structure.terms))
        issues.extend(self._check_mechanism_compatibility(structure))
        issues.extend(self._check_property_contradiction(structure))
        return issues

    def _check_term_existence(self, terms: list[str]) -> list[TermIssue]:
        issues: list[TermIssue] = []
        for term in terms:
            if term in self._all_terms:
                continue
            if term in self._alias_map:
                continue
            closest = self._find_closest_term(term)
            issues.append(TermIssue(
                original=term,
                canonical=closest or "",
                issue_type="wrong_term",
                explanation=f"术语「{term}」不在词典枚举中" + (f"，最接近的是「{closest}」" if closest else ""),
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

    def _find_closest_term(self, unknown: str) -> Optional[str]:
        best_term: Optional[str] = None
        best_dist = 3
        for term in self._all_terms:
            if len(term) > 10 or len(unknown) > 10:
                continue
            d = _edit_distance(unknown, term)
            if d < best_dist:
                best_dist = d
                best_term = term
        return best_term

    def _apply_corrections(self, structure: QuestionStructure, issues: list[TermIssue]) -> QuestionStructure:
        correction_map = {i.original: i.canonical for i in issues if i.canonical}
        new_terms = [correction_map.get(t, t) for t in structure.terms]
        return replace(structure, terms=new_terms)

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
        lines: list[str] = []
        for layer in structure.layers:
            terms = self._inventory.by_layer.get(layer, [])
            if terms:
                lines.append(f"{layer}: {', '.join(terms[:20])}")
        return "\n".join(lines) if lines else "（无相关术语）"

    def _format_knowledge_excerpt(self, structure: QuestionStructure) -> str:
        relevant = [item for item in self._structured if item.term in structure.terms][:5]
        lines: list[str] = []
        for item in relevant:
            mechs = ", ".join(item.mechanisms[:8]) if item.mechanisms else "无"
            lines.append(f"- {item.term}（{item.term_type}）: mechanisms=[{mechs}]")
        return "\n".join(lines) if lines else "（无相关知识）"


def _edit_distance(a: str, b: str) -> int:
    if len(a) > len(b):
        a, b = b, a
    prev = list(range(len(a) + 1))
    for j in range(1, len(b) + 1):
        curr = [j] + [0] * len(a)
        for i in range(1, len(a) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[i] = min(curr[i - 1] + 1, prev[i] + 1, prev[i - 1] + cost)
        prev = curr
    return prev[len(a)]
