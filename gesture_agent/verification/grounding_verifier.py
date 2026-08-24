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
_EVIDENCE_LIMIT = 4000
_JUDGE_BATCH_SIZE = 10
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

    def sanitize(self, answer: str, report: GroundingReport) -> tuple[str, GroundingReport]:
        """Remove claims already judged unsafe and derive a report for the safe subset.

        This is the deterministic last resort after generation retries. It never
        upgrades an undecided claim: every retained claim must match one that the
        judge already marked supported.
        """
        rejected = {
            claim.text
            for claim in report.claims
            if claim.verdict in {"partially_supported", "unsupported", "conflicted"}
        }
        if not rejected:
            return answer, report
        sanitized = _remove_claim_texts(answer, rejected)
        if sanitized == answer:
            return answer, report

        supported = {
            (claim.section, claim.text, claim.citation_ids): claim
            for claim in report.claims
            if claim.verdict == "supported"
        }
        retained: list[GroundingClaim] = []
        for _ in range(3):
            retained = []
            unknown: set[str] = set()
            for claim in _extract_claims(sanitized)[: self._max_claims]:
                previous = supported.get((claim.section, claim.text, claim.citation_ids))
                if previous is None:
                    unknown.add(claim.text)
                    continue
                retained.append(replace(
                    claim,
                    verdict="supported",
                    confidence=previous.confidence,
                    reason=previous.reason,
                ))
            if not unknown:
                break
            narrowed = _remove_claim_texts(sanitized, unknown)
            if narrowed == sanitized:
                return answer, report
            sanitized = narrowed
        else:
            return answer, report
        if not retained:
            return sanitized, GroundingReport(
                status="pass",
                score=1.0,
                claims=(),
                should_retry=False,
            )
        safe_claims = tuple(retained)
        return sanitized, GroundingReport(
            status="pass",
            score=_score(safe_claims),
            claims=safe_claims,
            should_retry=False,
        )

    def _judge_claims(
        self,
        claims: list[GroundingClaim],
        sources: list[SourceChunk],
    ) -> dict[str, GroundingClaim]:
        decisions: dict[str, GroundingClaim] = {}
        for start in range(0, len(claims), _JUDGE_BATCH_SIZE):
            batch = claims[start : start + _JUDGE_BATCH_SIZE]
            decisions.update(self._judge_claim_batch(batch, sources))
        return decisions

    def _judge_claim_batch(
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

特别严格区分“资料事实”和“设计推导”：
- 证据只描述控件/机制的属性，Claim 把该属性进一步转换成某个新场景的选型建议、优先级或具体方案时，除非证据明确提到该场景与建议，否则判 partially_supported。
- 证据只说可以空间分区，Claim 自行指定左上角、热区宽度、替代手势、平台行为等新增设计细节时，判 partially_supported 或 unsupported。
- Claim 在末尾附上“当前资料没有直接证据”不能让前面的无依据内容变成 supported。

只输出 JSON，不要 Markdown，不要解释 JSON 以外的内容：
{"claims":[{"id":"c1","verdict":"supported|partially_supported|unsupported|conflicted","confidence":0.0,"reason":"简短依据"}]}"""


def _extract_claims(answer: str) -> list[GroundingClaim]:
    section = "直接回答"
    claims: list[GroundingClaim] = []
    in_fence = False
    raw_lines = answer.splitlines()
    for line_index, raw_line in enumerate(raw_lines):
        line = raw_line.strip()
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence or not line:
            continue
        if line.startswith("## "):
            section = line[3:].strip()
            continue
        next_line = raw_lines[line_index + 1].strip() if line_index + 1 < len(raw_lines) else ""
        if (
            line.startswith("#")
            or line.startswith("![")
            or _is_table_separator(line)
            or _is_table_separator(next_line)
            or line.endswith(("：", ":"))
        ):
            continue
        line = _LIST_PREFIX_RE.sub("", line)
        line_citations = tuple(dict.fromkeys(int(value) for value in _CITATION_RE.findall(line)))
        for sentence in _split_sentences(line):
            citation_ids = tuple(dict.fromkeys(int(value) for value in _CITATION_RE.findall(sentence)))
            if not citation_ids:
                citation_ids = line_citations
            text = _clean_claim_text(sentence)
            if len(text) < 5 or _is_clarification_question(text, section):
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
    normalized = text.strip(" -*_：:|（）()")
    return any(normalized.startswith(marker) for marker in (
        "当前资料没有直接证据",
        "当前语料没有直接证据",
        "现有资料无法确认",
        "现有语料无法确认",
        "资料不足以判断",
        "语料不足以判断",
    ))


def _is_clarification_question(text: str, section: str = "") -> bool:
    normalized = text.strip(" -*_：:|（）()")
    if not normalized.endswith(("？", "?")):
        return False
    if "澄清" in section or "补充的信息" in section:
        return True
    return bool(re.search(
        r"(?:^|[：:])(是否|能否|可否|有没有|有无|谁|什么|哪|几|多少|为何|为什么|怎么|如何)|"
        r"还是|需不需要|要不要|请确认|请问",
        normalized,
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


def _remove_claim_texts(answer: str, rejected: set[str]) -> str:
    lines: list[str] = []
    for raw_line in answer.splitlines():
        parts = _split_sentences(raw_line)
        kept = [
            part for part in parts
            if _clean_claim_text(_LIST_PREFIX_RE.sub("", part.strip())) not in rejected
        ]
        if len(kept) == len(parts):
            lines.append(raw_line)
        elif kept:
            lines.append(" ".join(part.strip() for part in kept))
        else:
            lines.append("")
    sanitized = "\n".join(lines).strip()
    return re.sub(r"\n{3,}", "\n\n", sanitized)


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
