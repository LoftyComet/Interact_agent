from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Literal, Optional, Protocol

from gesture_agent.core.models import SourceChunk


GroundingVerdict = Literal[
    "supported",
    "partially_supported",
    "unsupported",
    "conflicted",
    "unverified",
]

_CITATION_RE = re.compile(r"(?<!!)\[(\d+)\](?!\()")
_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_LIST_PREFIX_RE = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)、]\s*)")
_FENCE_RE = re.compile(r"^```")
_EVIDENCE_LIMIT = 2200
_VALID_VERDICTS = {"supported", "partially_supported", "unsupported", "conflicted"}


class GroundingJudge(Protocol):
    def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> str: ...


@dataclass(frozen=True)
class GroundingClaim:
    id: str
    section: str
    text: str
    citation_ids: tuple[int, ...] = field(default_factory=tuple)
    verdict: GroundingVerdict = "unverified"
    confidence: float = 0.0
    reason: str = ""


@dataclass(frozen=True)
class GroundingReport:
    status: Literal["pass", "issues_found", "unavailable"]
    score: float
    claims: tuple[GroundingClaim, ...] = field(default_factory=tuple)
    should_retry: bool = False
    correction_hints: str = ""
    error: str = ""

    @property
    def issue_descriptions(self) -> list[str]:
        descriptions = [
            f"{claim.section}：{claim.text}（{claim.reason}）"
            for claim in self.claims
            if claim.verdict in {"partially_supported", "unsupported", "conflicted"}
        ]
        if self.error:
            descriptions.append(self.error)
        return descriptions

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "score": self.score,
            "should_retry": self.should_retry,
            "claims": [asdict(claim) for claim in self.claims],
            "error": self.error,
        }


class GroundingVerifier:
    """Verify whether generated claims are supported by their cited chunks.

    The interface intentionally accepts the rendered answer and the exact
    sources supplied to the generator. Claim extraction, evidence binding,
    semantic judging, scoring, and correction hints stay behind this seam.
    """

    def __init__(
        self,
        judge: Optional[GroundingJudge],
        *,
        strict: bool = True,
        minimum_score: float = 0.85,
        max_claims: int = 40,
    ) -> None:
        if not 0 <= minimum_score <= 1:
            raise ValueError("minimum_score must be between 0 and 1")
        if max_claims <= 0:
            raise ValueError("max_claims must be positive")
        self._judge = judge
        self._strict = strict
        self._minimum_score = minimum_score
        self._max_claims = max_claims

    def verify(self, answer: str, sources: list[SourceChunk]) -> GroundingReport:
        claims = _extract_claims(answer)[: self._max_claims]
        if not claims:
            return GroundingReport(
                status="issues_found",
                score=0.0,
                claims=(),
                should_retry=True,
                correction_hints="回答中没有可校验的事实陈述，请基于检索资料重新回答并标注引用。",
            )

        decided: dict[str, GroundingClaim] = {}
        judge_candidates: list[GroundingClaim] = []
        for claim in claims:
            if _is_evidence_limitation(claim.text):
                decided[claim.id] = replace(
                    claim,
                    verdict="supported",
                    confidence=1.0,
                    reason="该句明确声明资料证据不足，未冒充语料事实",
                )
                continue
            if not claim.citation_ids:
                decided[claim.id] = replace(
                    claim,
                    verdict="unsupported",
                    confidence=1.0,
                    reason="事实性陈述没有绑定任何引用",
                )
                continue
            invalid = [citation for citation in claim.citation_ids if citation < 1 or citation > len(sources)]
            if invalid:
                decided[claim.id] = replace(
                    claim,
                    verdict="unsupported",
                    confidence=1.0,
                    reason=f"引用编号超出资料范围：{invalid}",
                )
                continue
            judge_candidates.append(claim)

        if judge_candidates and self._judge is None:
            merged = tuple(decided.get(claim.id, claim) for claim in claims)
            return GroundingReport(
                status="unavailable",
                score=_score(merged),
                claims=merged,
                should_retry=any(claim.verdict == "unsupported" for claim in merged),
                correction_hints=_build_correction_hints(merged),
                error="语义一致性校验器未配置，带引用的陈述尚未完成语义判定",
            )

        if judge_candidates:
            try:
                decisions = self._judge_claims(judge_candidates, sources)
            except Exception as exc:
                merged = tuple(decided.get(claim.id, claim) for claim in claims)
                return GroundingReport(
                    status="unavailable",
                    score=_score(merged),
                    claims=merged,
                    should_retry=any(claim.verdict == "unsupported" for claim in merged),
                    correction_hints=_build_correction_hints(merged),
                    error=f"语义一致性校验暂时不可用：{exc}",
                )
            decided.update(decisions)

        merged = tuple(decided.get(claim.id, claim) for claim in claims)
        score = _score(merged)
        failing_verdicts = {"unsupported", "conflicted"}
        if self._strict:
            failing_verdicts.add("partially_supported")
        should_retry = score < self._minimum_score or any(
            claim.verdict in failing_verdicts for claim in merged
        )
        return GroundingReport(
            status="issues_found" if should_retry else "pass",
            score=score,
            claims=merged,
            should_retry=should_retry,
            correction_hints=_build_correction_hints(merged),
        )

    def _judge_claims(
        self,
        claims: list[GroundingClaim],
        sources: list[SourceChunk],
    ) -> dict[str, GroundingClaim]:
        used_ids = sorted({citation for claim in claims for citation in claim.citation_ids})
        evidence = [
            {
                "id": citation,
                "title": sources[citation - 1].title,
                "source": sources[citation - 1].citation(),
                "text": sources[citation - 1].text[:_EVIDENCE_LIMIT],
            }
            for citation in used_ids
        ]
        payload = {
            "claims": [
                {
                    "id": claim.id,
                    "text": claim.text,
                    "evidence_ids": list(claim.citation_ids),
                }
                for claim in claims
            ],
            "evidence": evidence,
        }
        response = self._judge.chat(
            [
                {"role": "system", "content": _GROUNDING_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            temperature=0.0,
            max_tokens=2400,
            enable_thinking=False,
        )
        raw = _parse_json_object(response)
        rows = raw.get("claims")
        if not isinstance(rows, list):
            raise ValueError("校验模型没有返回 claims 数组")
        by_id = {claim.id: claim for claim in claims}
        decisions: dict[str, GroundingClaim] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            claim_id = str(row.get("id", ""))
            verdict = str(row.get("verdict", ""))
            if claim_id not in by_id or verdict not in _VALID_VERDICTS:
                continue
            confidence = _bounded_float(row.get("confidence", 0.0))
            decisions[claim_id] = replace(
                by_id[claim_id],
                verdict=verdict,
                confidence=confidence,
                reason=str(row.get("reason", "")).strip()[:300],
            )
        missing = [claim.id for claim in claims if claim.id not in decisions]
        if missing:
            raise ValueError(f"校验模型遗漏 Claim：{', '.join(missing)}")
        return decisions


_GROUNDING_SYSTEM_PROMPT = """你是严格的语料一致性审计器。你只能依据输入中的 evidence 判断 claims，不得使用自身知识。

逐条输出以下 verdict 之一：
- supported：引用证据直接支持陈述中的主体、关系、条件、范围和强度。
- partially_supported：核心方向有依据，但陈述扩大了范围、遗漏关键条件、增强了确定性或混入未证实细节。
- unsupported：引用证据没有提供足够依据。
- conflicted：引用证据与陈述明确矛盾。

设计建议只有在证据能够直接推出时才算 supported。证据说“可能/通常/部分场景”，Claim 说“必然/所有场景”时必须判 partially_supported。引用编号存在不等于证据支持。

只输出 JSON，不要 Markdown，不要解释 JSON 以外的内容：
{"claims":[{"id":"c1","verdict":"supported|partially_supported|unsupported|conflicted","confidence":0.0,"reason":"简短依据"}]}"""


def _extract_claims(answer: str) -> list[GroundingClaim]:
    section = "直接回答"
    claims: list[GroundingClaim] = []
    in_fence = False
    for raw_line in answer.splitlines():
        line = raw_line.strip()
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence or not line:
            continue
        if line.startswith("## "):
            section = line[3:].strip()
            continue
        if line.startswith("#") or line.startswith("![") or _is_table_separator(line):
            continue
        line = _LIST_PREFIX_RE.sub("", line)
        for sentence in _split_sentences(line):
            citation_ids = tuple(dict.fromkeys(int(value) for value in _CITATION_RE.findall(sentence)))
            text = _clean_claim_text(sentence)
            if len(text) < 8:
                continue
            claims.append(GroundingClaim(
                id=f"c{len(claims) + 1}",
                section=section,
                text=text,
                citation_ids=citation_ids,
            ))
    return claims


def _split_sentences(line: str) -> list[str]:
    parts: list[str] = []
    start = 0
    index = 0
    while index < len(line):
        if line[index] not in "。！？!?；;":
            index += 1
            continue
        end = index + 1
        while True:
            match = re.match(r"\s*\[\d+\]", line[end:])
            if not match:
                break
            end += match.end()
        segment = line[start:end].strip()
        if segment:
            parts.append(segment)
        start = end
        index = end
    remainder = line[start:].strip()
    if remainder:
        parts.append(remainder)
    return parts


def _clean_claim_text(text: str) -> str:
    text = _CITATION_RE.sub("", text)
    text = _MARKDOWN_LINK_RE.sub(r"\1", text)
    text = text.replace("**", "").replace("__", "").replace("`", "")
    return re.sub(r"\s+", " ", text).strip(" |")


def _is_table_separator(line: str) -> bool:
    compact = line.replace("|", "").replace(":", "").replace("-", "").strip()
    return not compact and "-" in line


def _is_evidence_limitation(text: str) -> bool:
    return any(marker in text for marker in (
        "当前资料没有直接证据",
        "当前语料没有直接证据",
        "现有资料无法确认",
        "现有语料无法确认",
        "资料不足以判断",
        "语料不足以判断",
    ))


def _score(claims: tuple[GroundingClaim, ...]) -> float:
    if not claims:
        return 0.0
    weights = {
        "supported": 1.0,
        "partially_supported": 0.5,
        "unsupported": 0.0,
        "conflicted": 0.0,
        "unverified": 0.0,
    }
    return round(sum(weights[claim.verdict] for claim in claims) / len(claims), 4)


def _build_correction_hints(claims: tuple[GroundingClaim, ...]) -> str:
    lines = []
    for claim in claims:
        if claim.verdict not in {"partially_supported", "unsupported", "conflicted"}:
            continue
        lines.append(
            f"- [{claim.verdict}] 「{claim.text}」：{claim.reason}。"
            "请删除、缩小表述范围，或改成‘当前资料没有直接证据’，并绑定真正支持它的引用。"
        )
    return "\n".join(lines)


def _parse_json_object(response: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", response, re.DOTALL)
    if not match:
        raise ValueError("校验模型没有返回 JSON 对象")
    payload = json.loads(match.group())
    if not isinstance(payload, dict):
        raise ValueError("校验模型返回的 JSON 不是对象")
    return payload


def _bounded_float(value: Any) -> float:
    try:
        return round(min(max(float(value), 0.0), 1.0), 4)
    except (TypeError, ValueError):
        return 0.0
