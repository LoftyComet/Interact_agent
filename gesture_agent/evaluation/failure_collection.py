from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional


_WHITESPACE_RE = re.compile(r"\s+")
_REDACTIONS = (
    (re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b"), "[REDACTED_API_KEY]"),
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "[REDACTED_EMAIL]"),
    (re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"), "[REDACTED_PHONE]"),
    (re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)"), "[REDACTED_ID]"),
)


@dataclass(frozen=True)
class CollectionResult:
    response_id: str
    candidate_id: Optional[str] = None


def detect_failure_signals(
    *,
    provider_error: str = "",
    output_issues: Iterable[str] = (),
    grounding_status: str = "",
    reasoning_status: str = "",
    safety_fallback_applied: bool = False,
    retry_count: int = 0,
    collect_retries: bool = True,
) -> list[str]:
    """Map runtime diagnostics to stable failure-candidate trigger labels."""

    signals: list[str] = []
    if provider_error:
        signals.append("provider_error")
    if list(output_issues):
        signals.append("output_quality_failure")
    if grounding_status and grounding_status != "pass":
        signals.append(f"grounding_{grounding_status}")
    if reasoning_status and reasoning_status != "pass":
        signals.append(f"reasoning_{reasoning_status}")
    if safety_fallback_applied:
        signals.append("safety_fallback")
    if collect_retries and retry_count > 0:
        signals.append("generation_retry")
    return list(dict.fromkeys(signals))


class FailureCollector:
    """Keep real failure candidates outside the corpus and retrieval index.

    Only triggered responses are persisted. Successful responses stay in a
    bounded in-memory cache long enough for explicit user feedback; a thumbs
    down promotes that response into the candidate database.
    """

    def __init__(
        self,
        database_path: str | Path,
        *,
        enabled: bool = False,
        store_raw_query: bool = True,
        redact_sensitive_data: bool = True,
        max_recent_responses: int = 256,
    ) -> None:
        if max_recent_responses <= 0:
            raise ValueError("max_recent_responses must be positive")
        self.path = Path(database_path)
        self.enabled = enabled
        self.store_raw_query = store_raw_query
        self.redact_sensitive_data = redact_sensitive_data
        self.max_recent_responses = max_recent_responses
        self._lock = threading.Lock()
        self._recent: OrderedDict[str, dict[str, Any]] = OrderedDict()
        if enabled:
            self._initialize()

    def observe(
        self,
        record: dict[str, Any],
        *,
        triggers: Iterable[str] = (),
    ) -> CollectionResult:
        if not self.enabled:
            return CollectionResult(response_id="")
        response_id = str(record.get("response_id") or f"resp_{uuid.uuid4().hex}")
        prepared = self._prepare_record({**record, "response_id": response_id})
        trigger_list = sorted({str(value) for value in triggers if str(value)})
        with self._lock:
            self._remember(response_id, prepared)
            candidate_id = self._capture_locked(prepared, trigger_list) if trigger_list else None
        return CollectionResult(response_id=response_id, candidate_id=candidate_id)

    def record_feedback(
        self,
        response_id: str,
        *,
        rating: str,
        note: str = "",
    ) -> Optional[str]:
        if not self.enabled:
            raise RuntimeError("failure collection is disabled")
        normalized_rating = rating.strip().lower()
        if normalized_rating not in {"up", "down"}:
            raise ValueError("rating must be 'up' or 'down'")
        with self._lock:
            record = self._recent.get(response_id)
            if record is None:
                return None
            note = self._redact_text(note.strip()[:1000])
            if normalized_rating == "down":
                candidate_id = self._candidate_for_response_locked(response_id)
                if candidate_id:
                    self._add_trigger_locked(candidate_id, "user_negative")
                else:
                    candidate_id = self._capture_locked(
                        {
                            **record,
                            "user_feedback": {"rating": "down", "note": note},
                        },
                        ["user_negative"],
                    )
            else:
                candidate_id = self._candidate_for_response_locked(response_id)
            self._write_feedback_locked(response_id, normalized_rating, note, candidate_id)
            return candidate_id

    def _initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS candidates (
                    candidate_id TEXT PRIMARY KEY,
                    fingerprint TEXT NOT NULL UNIQUE,
                    first_seen TEXT NOT NULL,
                    last_seen TEXT NOT NULL,
                    occurrence_count INTEGER NOT NULL DEFAULT 1,
                    status TEXT NOT NULL DEFAULT 'pending',
                    response_id TEXT NOT NULL,
                    triggers_json TEXT NOT NULL,
                    query_text TEXT NOT NULL,
                    query_sha256 TEXT NOT NULL,
                    intent TEXT NOT NULL,
                    subtype TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    answer_text TEXT NOT NULL,
                    answer_blocks_json TEXT NOT NULL,
                    chunk_refs_json TEXT NOT NULL,
                    diagnostics_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    user_rating TEXT NOT NULL DEFAULT '',
                    user_note TEXT NOT NULL DEFAULT ''
                );
                CREATE INDEX IF NOT EXISTS idx_candidates_status_last_seen
                    ON candidates(status, last_seen DESC);
                CREATE TABLE IF NOT EXISTS feedback_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    response_id TEXT NOT NULL,
                    candidate_id TEXT,
                    rating TEXT NOT NULL,
                    note TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS response_candidates (
                    response_id TEXT PRIMARY KEY,
                    candidate_id TEXT NOT NULL,
                    FOREIGN KEY(candidate_id) REFERENCES candidates(candidate_id)
                );
                """
            )

    def _prepare_record(self, record: dict[str, Any]) -> dict[str, Any]:
        prepared = _redact_value(record, self._redact_text) if self.redact_sensitive_data else record
        query = str(prepared.get("query") or "")
        resolved_query = str(prepared.get("resolved_query") or query)
        query_hash = hashlib.sha256(query.encode("utf-8")).hexdigest()
        if not self.store_raw_query:
            prepared = {**prepared, "query": "", "resolved_query": ""}
            resolved_query = ""
        prepared["query_sha256"] = query_hash
        prepared["metadata"] = {
            **(prepared.get("metadata") or {}),
            "resolved_query": resolved_query,
        }
        # Source excerpts are intentionally never persisted here. The immutable
        # chunk ids and locators are enough to reproduce evidence during review.
        prepared["chunks"] = [
            {
                "id": str(chunk.get("id") or ""),
                "title": str(chunk.get("title") or ""),
                "citation": str(chunk.get("citation") or ""),
                "source": str(chunk.get("source") or ""),
                "start_line": chunk.get("start_line"),
                "end_line": chunk.get("end_line"),
                "score": chunk.get("score"),
            }
            for chunk in prepared.get("chunks") or []
            if isinstance(chunk, dict)
        ]
        return prepared

    def _remember(self, response_id: str, record: dict[str, Any]) -> None:
        self._recent[response_id] = record
        self._recent.move_to_end(response_id)
        while len(self._recent) > self.max_recent_responses:
            self._recent.popitem(last=False)

    def _capture_locked(self, record: dict[str, Any], triggers: list[str]) -> str:
        now = datetime.now().astimezone().isoformat()
        fingerprint = self._fingerprint(record)
        candidate_id = f"fail_{fingerprint[:16]}"
        structure = record.get("structure") or {}
        diagnostics = record.get("diagnostics") or {}
        metadata = record.get("metadata") or {}
        with sqlite3.connect(self.path) as connection:
            current = connection.execute(
                "SELECT triggers_json, occurrence_count FROM candidates WHERE fingerprint = ?",
                (fingerprint,),
            ).fetchone()
            if current:
                merged_triggers = sorted(set(json.loads(current[0])) | set(triggers))
                connection.execute(
                    """
                    UPDATE candidates
                    SET last_seen = ?, occurrence_count = ?, response_id = ?,
                        triggers_json = ?, answer_text = ?, answer_blocks_json = ?,
                        chunk_refs_json = ?, diagnostics_json = ?, metadata_json = ?,
                        user_rating = ?, user_note = ?
                    WHERE fingerprint = ?
                    """,
                    (
                        now,
                        int(current[1]) + 1,
                        record["response_id"],
                        _json(merged_triggers),
                        str(record.get("answer") or ""),
                        _json(record.get("answer_blocks") or []),
                        _json(record.get("chunks") or []),
                        _json(diagnostics),
                        _json(metadata),
                        str((record.get("user_feedback") or {}).get("rating") or ""),
                        str((record.get("user_feedback") or {}).get("note") or ""),
                        fingerprint,
                    ),
                )
            else:
                connection.execute(
                    """
                    INSERT INTO candidates (
                        candidate_id, fingerprint, first_seen, last_seen,
                        occurrence_count, status, response_id, triggers_json,
                        query_text, query_sha256, intent, subtype, provider, model,
                        answer_text, answer_blocks_json, chunk_refs_json,
                        diagnostics_json, metadata_json, user_rating, user_note
                    ) VALUES (?, ?, ?, ?, 1, 'pending', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        candidate_id,
                        fingerprint,
                        now,
                        now,
                        record["response_id"],
                        _json(triggers),
                        str(record.get("query") or ""),
                        str(record.get("query_sha256") or ""),
                        str(structure.get("intent") or ""),
                        str(structure.get("subtype") or ""),
                        str(record.get("provider") or ""),
                        str(record.get("model") or ""),
                        str(record.get("answer") or ""),
                        _json(record.get("answer_blocks") or []),
                        _json(record.get("chunks") or []),
                        _json(diagnostics),
                        _json(metadata),
                        str((record.get("user_feedback") or {}).get("rating") or ""),
                        str((record.get("user_feedback") or {}).get("note") or ""),
                    ),
                )
            connection.execute(
                "INSERT OR REPLACE INTO response_candidates(response_id, candidate_id) VALUES (?, ?)",
                (record["response_id"], candidate_id),
            )
        return candidate_id

    def _fingerprint(self, record: dict[str, Any]) -> str:
        structure = record.get("structure") or {}
        signature = {
            "query": _WHITESPACE_RE.sub("", str(record.get("query") or "")).lower(),
            "query_sha256": record.get("query_sha256"),
            "resolved_query": _WHITESPACE_RE.sub(
                "", str(record.get("resolved_query") or "")
            ).lower(),
            "intent": structure.get("intent"),
            "subtype": structure.get("subtype"),
        }
        return hashlib.sha256(_json(signature).encode("utf-8")).hexdigest()

    def _candidate_for_response_locked(self, response_id: str) -> Optional[str]:
        with sqlite3.connect(self.path) as connection:
            row = connection.execute(
                "SELECT candidate_id FROM response_candidates WHERE response_id = ?",
                (response_id,),
            ).fetchone()
        return str(row[0]) if row else None

    def _add_trigger_locked(self, candidate_id: str, trigger: str) -> None:
        with sqlite3.connect(self.path) as connection:
            row = connection.execute(
                "SELECT triggers_json FROM candidates WHERE candidate_id = ?",
                (candidate_id,),
            ).fetchone()
            if not row:
                return
            triggers = sorted(set(json.loads(row[0])) | {trigger})
            connection.execute(
                "UPDATE candidates SET triggers_json = ? WHERE candidate_id = ?",
                (_json(triggers), candidate_id),
            )

    def _write_feedback_locked(
        self,
        response_id: str,
        rating: str,
        note: str,
        candidate_id: Optional[str],
    ) -> None:
        now = datetime.now().astimezone().isoformat()
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                "INSERT INTO feedback_events(timestamp, response_id, candidate_id, rating, note) VALUES (?, ?, ?, ?, ?)",
                (now, response_id, candidate_id, rating, note),
            )
            if candidate_id:
                connection.execute(
                    "UPDATE candidates SET user_rating = ?, user_note = ? WHERE candidate_id = ?",
                    (rating, note, candidate_id),
                )

    def _redact_text(self, text: str) -> str:
        redacted = text
        for pattern, replacement in _REDACTIONS:
            redacted = pattern.sub(replacement, redacted)
        return redacted


def _redact_value(value: Any, redact) -> Any:
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, list):
        return [_redact_value(item, redact) for item in value]
    if isinstance(value, tuple):
        return [_redact_value(item, redact) for item in value]
    if isinstance(value, dict):
        return {str(key): _redact_value(item, redact) for key, item in value.items()}
    return value


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
