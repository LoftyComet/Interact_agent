from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Optional, Protocol

from gesture_agent.core.models import SourceChunk


class ReasoningJudge(Protocol):
    def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> str: ...


@dataclass(frozen=True)
class ReasoningIssue:
    issue_type: Literal[
        "external_fact",
        "corpus_conflict",
        "book_misattribution",
        "unlabeled_assertion",
    ]
    text: str
    reason: str


@dataclass(frozen=True)
class ReasoningReport:
    status: Literal["pass", "issues_found", "unavailable"]
    issues: tuple[ReasoningIssue, ...] = field(default_factory=tuple)
    should_retry: bool = False
    correction_hints: str = ""
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "should_retry": self.should_retry,
            "issues": [asdict(issue) for issue in self.issues],
            "error": self.error,
        }


class ReasoningVerifier:
    """Audit labeled design reasoning without requiring direct corpus support."""

    def __init__(self, judge: Optional[ReasoningJudge]) -> None:
        self._judge = judge

    def verify(self, reasoning: str, sources: list[SourceChunk]) -> ReasoningReport:
        if not reasoning.strip():
            return ReasoningReport(status="pass")
        if self._judge is None:
            return ReasoningReport(
                status="unavailable",
                error="设计推导审计器未配置",
            )
        deterministic_issues = _deterministic_policy_issues(reasoning, sources)
        payload = {
            "design_reasoning": reasoning[:8000],
            "evidence": [
                {
                    "id": index,
                    "title": source.title,
                    "text": source.text[:2400],
                }
                for index, source in enumerate(sources, start=1)
            ],
        }
        try:
            response = self._judge.chat(
                [
                    {"role": "system", "content": _REASONING_AUDIT_PROMPT},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                temperature=0.0,
                max_tokens=1200,
                enable_thinking=False,
            )
            raw = _parse_json_object(response)
            rows = raw.get("issues", [])
            if not isinstance(rows, list):
                raise ValueError("审计模型没有返回 issues 数组")
            issues: list[ReasoningIssue] = list(deterministic_issues)
            valid_types = {
                "external_fact", "corpus_conflict", "book_misattribution", "unlabeled_assertion"
            }
            for row in rows:
                if not isinstance(row, dict):
                    continue
                issue_type = str(row.get("issue_type", ""))
                if issue_type not in valid_types:
                    continue
                candidate = ReasoningIssue(
                    issue_type=issue_type,  # type: ignore[arg-type]
                    text=str(row.get("text", "")).strip()[:500],
                    reason=str(row.get("reason", "")).strip()[:500],
                )
                if candidate not in issues:
                    issues.append(candidate)
        except Exception as exc:
            return ReasoningReport(
                status="unavailable",
                error=f"设计推导审计暂时不可用：{exc}",
            )
        if not issues:
            return ReasoningReport(status="pass")
        hints = "\n".join(
            f"- [{issue.issue_type}] 「{issue.text}」：{issue.reason}"
            for issue in issues
        )
        return ReasoningReport(
            status="issues_found",
            issues=tuple(issues),
            should_retry=True,
            correction_hints=(
                hints
                + "\n请保留合理的设计假设，但删除无独立来源的外部事实/参数，"
                "纠正与语料冲突或冒充书中内容的表述。"
            ),
        )


_REASONING_AUDIT_PROMPT = """你是严格的 IxDL 设计推导审计器。输入包含已明确标注的 design_reasoning 和本轮 evidence。

设计推导可以提出新的方案与待验证假设，不要求被 evidence 直接支持；但必须使用“可以尝试、可能、需要验证”等推导措辞，且不得：
1. 写入 evidence 中没有的外部产品事实、平台行为、研究结论、时间、比例、尺寸、阈值或其他具体参数；
2. 与 evidence 明确冲突；
3. 声称某个推导“来自书中/词典明确指出”；
4. 把推测写成确定事实或唯一/最佳答案。

仅报告实质问题。只输出 JSON：
{"issues":[{"issue_type":"external_fact|corpus_conflict|book_misattribution|unlabeled_assertion","text":"问题原文","reason":"简短理由"}]}
没有问题时输出 {"issues":[]}。"""


def _parse_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise
        parsed = json.loads(match.group())
    if not isinstance(parsed, dict):
        raise ValueError("审计模型返回值不是 JSON 对象")
    return parsed


def _deterministic_policy_issues(
    reasoning: str,
    sources: list[SourceChunk],
) -> tuple[ReasoningIssue, ...]:
    issues: list[ReasoningIssue] = []
    evidence_text = "\n".join(source.text for source in sources).lower()
    cleaned = re.sub(r"(?<!!)\[\d+\](?!\()", "", reasoning)
    cleaned = re.sub(r"(?<![0-9A-Za-z-])\d+-[a-z](?![0-9A-Za-z-])", "", cleaned, flags=re.IGNORECASE)
    parameter_re = re.compile(
        r"(?<![\d.])\d+(?:\.\d+)?\s*(?:ms|px|pt|mm|cm|%|毫秒|秒|分钟|像素|厘米|毫米)(?![A-Za-z\u4e00-\u9fff])",
        re.IGNORECASE,
    )
    for match in parameter_re.finditer(cleaned):
        value = match.group().strip()
        if value.lower() not in evidence_text:
            issues.append(ReasoningIssue(
                issue_type="external_fact",
                text=value,
                reason="推导块出现了语料未提供的具体参数；应省略参数并说明需要测试确定",
            ))

    for sentence in re.split(r"[。！？；;\n]", cleaned):
        sentence = sentence.strip(" -*_：:|（）()")
        if not sentence:
            continue
        absolute_markers = ("最佳", "最优", "必然", "一定", "完全", "显著", "直接消除", "绝不会")
        hedge_markers = ("可以", "可能", "或许", "尝试", "建议", "需要验证", "待验证", "假设")
        if any(marker in sentence for marker in absolute_markers) and not any(
            marker in sentence for marker in hedge_markers
        ):
            issues.append(ReasoningIssue(
                issue_type="unlabeled_assertion",
                text=sentence[:500],
                reason="设计推导使用了确定性或强度过高的表述，应改为待验证假设",
            ))
    return tuple(issues)
