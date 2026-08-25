from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union


DEFAULT_AGENT_CONFIG_PATH = "agent_config.json"


@dataclass
class VerificationConfig:
    verify_input: bool = True
    verify_input_llm: bool = False
    verify_output: bool = True
    verify_output_llm: bool = False
    output_max_retries: int = 2
    verify_grounding: bool = True
    grounding_provider: str = "deepseek"
    grounding_model: Optional[str] = None
    grounding_strict: bool = True
    grounding_minimum_score: float = 0.85


@dataclass
class FailureCollectionConfig:
    enabled: bool = False
    database_path: str = "runtime/evaluation_candidates.sqlite"
    store_raw_query: bool = True
    collect_retries: bool = True
    redact_sensitive_data: bool = True
    max_recent_responses: int = 256


@dataclass
class PromptConfig:
    system_prompt: Optional[str] = None
    extra_system_prompt: str = ""
    response_instructions: Optional[list[str]] = None
    extra_response_instructions: list[str] = field(default_factory=list)


@dataclass
class AgentConfig:
    source: str = "defaults"
    data_dir: str = "data"
    term_inventory: Optional[str] = None
    output_frames: Optional[str] = None
    structured_knowledge: Optional[str] = None
    knowledge_index: Optional[str] = None
    top_k: int = 6
    images: list[str] = field(default_factory=list)
    model: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.2
    max_tokens: int = 1600
    timeout: Optional[int] = None
    enable_thinking: Optional[bool] = None
    stream: bool = False
    check_api: bool = False
    show_structure: bool = False
    show_context: bool = False
    dry_run: bool = False
    default_interactive: bool = True
    llm_intent: bool = False
    llm_clarify: bool = True
    llm_dynamic_clarify_question: bool = True
    intent_candidate_options: bool = True
    llm_output_frame: str = "auto"
    llm_output_frame_confidence_threshold: float = 0.80
    prompt: PromptConfig = field(default_factory=PromptConfig)
    verification: VerificationConfig = field(default_factory=VerificationConfig)
    failure_collection: FailureCollectionConfig = field(default_factory=FailureCollectionConfig)


def load_agent_config(config_path: Optional[Union[str, Path]] = None) -> AgentConfig:
    path = Path(config_path) if config_path else Path(DEFAULT_AGENT_CONFIG_PATH)
    if not path.exists():
        if config_path:
            raise FileNotFoundError(f"Agent config does not exist: {path}")
        return AgentConfig()

    raw = json.loads(_strip_json_comments(path.read_text(encoding="utf-8")))
    if not isinstance(raw, dict):
        raise ValueError(f"Agent config must be a JSON object: {path}")
    return agent_config_from_dict(raw, source=str(path))


def agent_config_from_dict(raw: dict[str, Any], *, source: str = "config") -> AgentConfig:
    data = _object_section(raw, "data")
    retrieval = _object_section(raw, "retrieval")
    model = _object_section(raw, "model")
    runtime = _object_section(raw, "runtime")
    intent = _object_section(raw, "intent")
    media = _object_section(raw, "media")
    prompt = _object_section(raw, "prompt")
    verification = _object_section(raw, "verification")
    failure_collection = _object_section(raw, "failure_collection")

    return AgentConfig(
        source=source,
        data_dir=str(data.get("data_dir", "data")),
        term_inventory=_optional_str(data.get("term_inventory")),
        output_frames=_optional_str(data.get("output_frames")),
        structured_knowledge=_optional_str(data.get("structured_knowledge")),
        knowledge_index=_optional_str(data.get("knowledge_index")),
        top_k=_int_value(retrieval.get("top_k", 6), "retrieval.top_k"),
        images=_string_list(media.get("images", []), "media.images"),
        model=_optional_str(model.get("model")),
        base_url=_optional_str(model.get("base_url")),
        temperature=float(model.get("temperature", 0.2)),
        max_tokens=_int_value(model.get("max_tokens", 1600), "model.max_tokens"),
        timeout=_optional_int(model.get("timeout"), "model.timeout"),
        enable_thinking=_optional_bool(model.get("enable_thinking"), "model.enable_thinking"),
        stream=bool(runtime.get("stream", False)),
        check_api=bool(runtime.get("check_api", False)),
        show_structure=bool(runtime.get("show_structure", False)),
        show_context=bool(runtime.get("show_context", False)),
        dry_run=bool(runtime.get("dry_run", False)),
        default_interactive=bool(runtime.get("default_interactive", True)),
        llm_intent=bool(intent.get("llm_intent", False)),
        llm_clarify=bool(intent.get("llm_clarify", True)),
        llm_dynamic_clarify_question=bool(intent.get("llm_dynamic_clarify_question", True)),
        intent_candidate_options=bool(intent.get("intent_candidate_options", True)),
        llm_output_frame=_llm_output_frame_value(intent.get("llm_output_frame", "auto")),
        llm_output_frame_confidence_threshold=float(intent.get("llm_output_frame_confidence_threshold", 0.80)),
        prompt=PromptConfig(
            system_prompt=_optional_text(prompt.get("system_prompt")),
            extra_system_prompt=_text_value(prompt.get("extra_system_prompt", "")),
            response_instructions=_optional_string_list(prompt.get("response_instructions"), "prompt.response_instructions"),
            extra_response_instructions=_string_list(prompt.get("extra_response_instructions", []), "prompt.extra_response_instructions"),
        ),
        verification=VerificationConfig(
            verify_input=bool(verification.get("verify_input", True)),
            verify_input_llm=bool(verification.get("verify_input_llm", False)),
            verify_output=bool(verification.get("verify_output", True)),
            verify_output_llm=bool(verification.get("verify_output_llm", False)),
            output_max_retries=_int_value(verification.get("output_max_retries", 2), "verification.output_max_retries"),
            verify_grounding=bool(verification.get("verify_grounding", True)),
            grounding_provider=str(verification.get("grounding_provider", "deepseek")),
            grounding_model=_optional_str(verification.get("grounding_model")),
            grounding_strict=bool(verification.get("grounding_strict", True)),
            grounding_minimum_score=_bounded_float(
                verification.get("grounding_minimum_score", 0.85),
                "verification.grounding_minimum_score",
            ),
        ),
        failure_collection=FailureCollectionConfig(
            enabled=bool(failure_collection.get("enabled", False)),
            database_path=str(
                failure_collection.get(
                    "database_path", "runtime/evaluation_candidates.sqlite"
                )
            ),
            store_raw_query=bool(failure_collection.get("store_raw_query", True)),
            collect_retries=bool(failure_collection.get("collect_retries", True)),
            redact_sensitive_data=bool(
                failure_collection.get("redact_sensitive_data", True)
            ),
            max_recent_responses=_int_value(
                failure_collection.get("max_recent_responses", 256),
                "failure_collection.max_recent_responses",
            ),
        ),
    )


def _object_section(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Agent config `{key}` must be an object.")
    return value


def _optional_str(value: Any) -> Optional[str]:
    if value is None or value == "":
        return None
    return str(value)


def _optional_text(value: Any) -> Optional[str]:
    text = _text_value(value)
    return text or None


def _text_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "\n".join(str(item) for item in value)
    return str(value)


def _optional_int(value: Any, field_name: str) -> Optional[int]:
    if value is None or value == "":
        return None
    return _int_value(value, field_name)


def _int_value(value: Any, field_name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Agent config `{field_name}` must be an integer.") from exc


def _bounded_float(value: Any, field_name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Agent config `{field_name}` must be a number.") from exc
    if not 0 <= parsed <= 1:
        raise ValueError(f"Agent config `{field_name}` must be between 0 and 1.")
    return parsed


def _optional_bool(value: Any, field_name: str) -> Optional[bool]:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n"}:
            return False
    raise ValueError(f"Agent config `{field_name}` must be true, false, or null.")


def _optional_string_list(value: Any, field_name: str) -> Optional[list[str]]:
    if value is None:
        return None
    return _string_list(value, field_name)


def _string_list(value: Any, field_name: str) -> list[str]:
    if value is None or value == "":
        return []
    if not isinstance(value, list):
        raise ValueError(f"Agent config `{field_name}` must be an array.")
    return [str(item).strip() for item in value if str(item).strip()]


_LLM_OUTPUT_FRAME_MODES = {"false", "auto", "always"}


def _llm_output_frame_value(value: Any) -> str:
    v = str(value).strip().lower() if value is not None else "auto"
    if v not in _LLM_OUTPUT_FRAME_MODES:
        raise ValueError(f"Agent config `intent.llm_output_frame` must be one of: {sorted(_LLM_OUTPUT_FRAME_MODES)}")
    return v


def _strip_json_comments(text: str) -> str:
    result: list[str] = []
    idx = 0
    in_string = False
    escape = False
    while idx < len(text):
        char = text[idx]
        next_char = text[idx + 1] if idx + 1 < len(text) else ""

        if in_string:
            result.append(char)
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            idx += 1
            continue

        if char == '"':
            in_string = True
            result.append(char)
            idx += 1
            continue

        if char == "/" and next_char == "/":
            idx += 2
            while idx < len(text) and text[idx] not in "\r\n":
                idx += 1
            continue

        if char == "/" and next_char == "*":
            idx += 2
            while idx + 1 < len(text) and not (text[idx] == "*" and text[idx + 1] == "/"):
                result.append("\n" if text[idx] in "\r\n" else " ")
                idx += 1
            idx += 2
            continue

        result.append(char)
        idx += 1

    return "".join(result)
