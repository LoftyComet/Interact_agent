from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Optional

from ..core.models import Intent, IntentOutputFrames, QuestionStructure, StructuredKnowledgeItem, TermInventory
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
    ) -> None:
        self._inventory = term_inventory
        self._frames = output_frames
        self._structured = structured_items
        self._client = client
        self._use_llm = use_llm and client is not None
        self._all_terms = set(term_inventory.all_terms())

    def verify(self, output: str, structure: QuestionStructure) -> OutputVerificationResult:
        issues: list[OutputIssue] = []
        issues.extend(self._check_sections(output, structure.intent))
        issues.extend(self._check_format(output))
        issues.extend(self._check_term_validity(output, structure))
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

    def _check_sections(self, output: str, intent: Intent) -> list[OutputIssue]:
        issues: list[OutputIssue] = []
        try:
            expected = self._frames.frame_for(intent)
        except KeyError:
            return issues
        headers = re.findall(r"^##\s+(.+)$", output, re.MULTILINE)
        headers_normalized = [h.strip() for h in headers]
        for section in expected:
            if not any(section in h for h in headers_normalized):
                issues.append(OutputIssue(
                    issue_type="missing_section",
                    location=f"## {section}",
                    description=f"缺少章节「{section}」",
                    severity="error",
                ))
        return issues

    def _check_format(self, output: str) -> list[OutputIssue]:
        issues: list[OutputIssue] = []
        stripped = output.strip()
        if stripped.startswith("```") and stripped.endswith("```"):
            issues.append(OutputIssue(
                issue_type="format_error",
                location="整体",
                description="输出被包裹在代码块中，应直接输出 Markdown",
                severity="error",
            ))
        if not re.search(r"^##\s+", output, re.MULTILINE):
            issues.append(OutputIssue(
                issue_type="format_error",
                location="整体",
                description="输出缺少 ## 标题结构",
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

    def _llm_verify(self, output: str, structure: QuestionStructure) -> list[OutputIssue]:
        if self._client is None:
            return []
        inventory_excerpt = self._format_inventory_excerpt(structure)
        knowledge_excerpt = self._format_knowledge_excerpt(structure)
        try:
            frame = self._frames.frame_for(structure.intent)
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
