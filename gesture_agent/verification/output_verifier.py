from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Optional

from ..core.models import IntentOutputFrames, QuestionStructure, StructuredKnowledgeItem, TermInventory
from ..knowledge.mechanism_registry import MechanismRegistry
from .answer_schema import parse_answer_markdown, render_answer_markdown
from .answer_blocks import parse_answer_blocks
from .models import OutputIssue, OutputVerificationResult
from .prompts import OUTPUT_VERIFICATION_PROMPT, OUTPUT_VERIFICATION_SYSTEM

if TYPE_CHECKING:
    from ..providers import SiliconFlowClient


class OutputVerifier:
    def __init__(
        self,
        term_inventory: TermInventory,
        output_frames: IntentOutputFrames,
        structured_items: list[StructuredKnowledgeItem],
        client: Optional[SiliconFlowClient] = None,
        use_llm: bool = False,
        mechanism_registry: Optional[MechanismRegistry] = None,
    ) -> None:
        self._inventory = term_inventory
        self._frames = output_frames
        self._structured = structured_items
        self._client = client
        self._use_llm = use_llm and client is not None
        self._all_terms = set(term_inventory.all_terms())
        self._mechanism_registry = mechanism_registry

    def verify(
        self,
        output: str,
        structure: QuestionStructure,
        source_count: int | None = None,
    ) -> OutputVerificationResult:
        issues: list[OutputIssue] = []
        issues.extend(self._check_contract(output, structure, source_count))
        issues.extend(self._check_term_validity(output, structure))
        issues.extend(self._check_mechanism_naming(output, structure))
        if self._use_llm and not issues:
            issues.extend(self._llm_verify(output, structure))
        if not issues:
            return OutputVerificationResult(status="pass")
        errors = [i for i in issues if i.severity == "error"]
        hints = self._build_correction_hints(issues) if errors else ""
        return OutputVerificationResult(
            status="issues_found",
            issues=issues,
            should_retry=bool(errors),
            correction_hints=hints,
        )

    def normalize(self, output: str, structure: QuestionStructure) -> str:
        """Return canonical Markdown when the answer satisfies its contract."""

        # Reordering headings after provenance markers have been added could move
        # a marker away from the text it labels. Marked answers are already
        # machine-readable, so preserve their block boundaries verbatim.
        if parse_answer_blocks(output).explicit_markers:
            return output

        try:
            expected = self._frames.frame_for(structure.intent, structure.subtype)
        except KeyError:
            return output
        result = parse_answer_markdown(output, expected)
        if not result.valid:
            return output
        return render_answer_markdown(result.document, expected)

    def _check_contract(
        self,
        output: str,
        structure: QuestionStructure,
        source_count: int | None,
    ) -> list[OutputIssue]:
        try:
            expected = self._frames.frame_for(structure.intent, structure.subtype)
        except KeyError:
            return []
        result = parse_answer_markdown(output, expected)
        issues = [
            OutputIssue(
                issue_type=violation.code,
                location=violation.location,
                description=violation.message,
                severity="error",
            )
            for violation in result.violations
        ]
        if source_count is not None:
            invalid = [value for value in result.document.citation_ids if value < 1 or value > source_count]
            if invalid:
                refs = "、".join(f"[{value}]" for value in invalid)
                issues.append(OutputIssue(
                    issue_type="invalid_citation",
                    location=refs,
                    description=f"引用编号 {refs} 超出本次提供的 {source_count} 条资料范围",
                    severity="error",
                ))
            elif source_count > 0 and not result.document.citation_ids:
                issues.append(OutputIssue(
                    issue_type="invalid_citation",
                    location="整体",
                    description="回答没有标注任何检索资料引用",
                    severity="warning",
                ))
        return issues

    def _check_term_validity(self, output: str, structure: QuestionStructure) -> list[OutputIssue]:
        issues: list[OutputIssue] = []
        relevant_items = [item for item in self._structured if item.term in structure.terms]
        if not relevant_items:
            return issues
        for item in relevant_items:
            if not item.mechanisms:
                continue
            supported_mechs = set(item.mechanisms)
            mech_layer_terms = self._inventory.by_layer.get("interaction_mechanism", [])
            for mech in mech_layer_terms:
                if mech in supported_mechs:
                    continue
                pattern = re.compile(rf"{re.escape(item.term)}[^。\n]{{0,20}}{re.escape(mech)}")
                if pattern.search(output):
                    issues.append(OutputIssue(
                        issue_type="knowledge_conflict",
                        location=f"{item.term}+{mech}",
                        description=f"输出声称「{item.term}」支持「{mech}」，但知识库中无此记录",
                        severity="warning",
                    ))
                    break
        return issues

    def _check_mechanism_naming(
        self,
        output: str,
        structure: QuestionStructure,
    ) -> list[OutputIssue]:
        if self._mechanism_registry is None:
            return []
        issues: list[OutputIssue] = []
        for issue in self._mechanism_registry.validate_answer(
            output,
            required_labels=structure.terms,
        ):
            found = "、".join(issue.found_codes)
            if issue.issue_type == "missing_mechanism_code":
                description = f"交互机制「{issue.mention}」缺少编号；应写为「{issue.expected}」"
            elif issue.issue_type in {"mechanism_code_mismatch", "mechanism_name_mismatch"}:
                description = f"交互机制名称与编号不匹配（检测到 {found}）；应写为「{issue.expected}」"
            else:
                description = f"交互机制编号「{issue.mention}」不在注册表中"
            issues.append(OutputIssue(
                issue_type=issue.issue_type,
                location=issue.mention,
                description=description,
                severity="error",
            ))
        return issues

    def _llm_verify(self, output: str, structure: QuestionStructure) -> list[OutputIssue]:
        if self._client is None:
            return []
        inventory_excerpt = self._format_inventory_excerpt(structure)
        knowledge_excerpt = self._format_knowledge_excerpt(structure)
        try:
            frame = self._frames.frame_for(structure.intent, structure.subtype)
        except KeyError:
            frame = []
        prompt = OUTPUT_VERIFICATION_PROMPT.format(
            term_inventory_excerpt=inventory_excerpt,
            structured_knowledge_excerpt=knowledge_excerpt,
            output_frame=", ".join(frame),
            output=output[:2000],
        )
        try:
            response = self._client.chat(
                [{"role": "system", "content": OUTPUT_VERIFICATION_SYSTEM}, {"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=500,
            )
            return self._parse_llm_issues(response)
        except Exception:
            return []

    def _parse_llm_issues(self, response: str) -> list[OutputIssue]:
        try:
            match = re.search(r"\{.*\}", response, re.DOTALL)
            if not match:
                return []
            data = json.loads(match.group())
            raw_issues = data.get("issues", [])
            return [
                OutputIssue(
                    issue_type=item.get("issue_type", "format_error"),
                    location=item.get("location", ""),
                    description=item.get("description", ""),
                    severity=item.get("severity", "warning"),
                )
                for item in raw_issues
                if item.get("description")
            ]
        except (json.JSONDecodeError, KeyError, TypeError):
            return []

    def _build_correction_hints(self, issues: list[OutputIssue]) -> str:
        lines: list[str] = []
        for i, issue in enumerate(issues, 1):
            if issue.severity == "error":
                lines.append(f"{i}. [{issue.issue_type}] {issue.description}")
        return "\n".join(lines)

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
