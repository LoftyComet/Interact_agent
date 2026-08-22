"""Flask backend for the gesture agent web UI.

Stays in `web/backend/` so it doesn't mix with the core `gesture_agent` package.
Reuses the existing pipeline: KnowledgeBase → QuestionParser → ConversationSession →
PromptBuilder → SiliconFlowClient.
"""

from __future__ import annotations

import json
import os
import re
import sys
import threading
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote

# Make the project root importable so `gesture_agent` resolves regardless of cwd.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flask import Flask, Response, jsonify, request, send_from_directory, stream_with_context
from flask_cors import CORS

from gesture_agent.core.models import QuestionStructure, SourceChunk
from gesture_agent.knowledge import KnowledgeBase
from gesture_agent.knowledge.image_index import ImageIndex
from gesture_agent.learning import ClarificationIntentResolver, ConversationSession, LLMOutputFrameResolver, LLMTurnRelationResolver, QuestionParser, TurnClassifier
from gesture_agent.learning.output_frames import load_output_frames
from gesture_agent.learning.prompt_builder import build_messages
from gesture_agent.media import image_path_to_data_url
from gesture_agent.providers import (
    ProviderError,
    SiliconFlowClient,
    SiliconFlowError,
    build_chat_client,
    list_providers,
    resolve_provider_id,
)
from gesture_agent.settings.app_config import load_agent_config
from gesture_agent.verification import GroundingReport, GroundingVerifier, InputVerifier, OutputVerifier
from gesture_agent.verification.prompts import OUTPUT_CORRECTION_PROMPT


FRONTEND_DIR = (Path(__file__).resolve().parent.parent / "frontend").resolve()
EXTRACTED_IMAGES_DIR = (PROJECT_ROOT / "data" / "pictures" / "extracted").resolve()

IMAGE_REF_RE = re.compile(r"!\[([^\]]*)\]\(image:([^)]+)\)")

QUESTION_LOG_PATH = (PROJECT_ROOT / "data" / "logs" / "web_questions.jsonl").resolve()
_question_log_lock = threading.Lock()


@dataclass(frozen=True)
class AnswerQualityResult:
    should_retry: bool
    correction_hints: str
    issue_descriptions: list[str]
    grounding: Optional[GroundingReport]


def log_user_question(
    session_id: str,
    question: str,
    *,
    endpoint: str,
    image_count: int = 0,
    style: str = "",
    intent: str = "",
    background: bool = False,
) -> None:
    """Append one user question per line to a JSONL log (thread-safe)."""
    record = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "session_id": session_id,
        "endpoint": endpoint,
        "question": question,
        "image_count": image_count,
        "style": style,
        "intent": intent,
        "background": background,
    }
    try:
        with _question_log_lock:
            QUESTION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with QUESTION_LOG_PATH.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        # Logging must never break a request.
        pass


def build_retrieval_answer(
    chunks: list[SourceChunk],
    resolved_query: str,
    available_images: Optional[list] = None,
) -> str:
    """直接返回检索资料，用于 retrieval_instruction 意图。
    优先返回图片（有图就不给文字），没有图片时才返回文本 chunks。"""
    # 有图片时只给图片
    if available_images:
        img_lines = []
        for img in available_images:
            img_lines.append(f"![{img.annotation}](image:{img.id})")
        return "\n\n".join(img_lines)

    # 没图片时给文本
    if chunks:
        return "\n\n---\n\n".join(chunk.text for chunk in chunks)

    return "词典中未找到对应的表达式、图示或案例。"


def resolve_image_refs(text: str, image_index: Optional[ImageIndex], base_url: str = "/api/images") -> str:
    if image_index is None:
        return text

    def _replacer(match):
        alt_text = match.group(1)
        image_id = match.group(2)
        filename = image_index.get_filename(image_id)
        if filename:
            # 文件名含空格/中文/冒号，必须 URL 编码，否则浏览器会在空格处截断 URL。
            return f"![{alt_text}]({base_url}/{quote(filename)})"
        return match.group(0)

    return IMAGE_REF_RE.sub(_replacer, text)


class AgentRuntime:
    """Holds the singletons the request handlers need."""

    def __init__(self, config_path: Optional[str] = None) -> None:
        self.config = load_agent_config(config_path)
        index_dir = self.config.knowledge_index
        if index_dir is None and Path("knowledge_index/manifest.json").exists():
            index_dir = "knowledge_index"
        index_embedder = self._maybe_build_index_embedder(index_dir)
        self.kb = KnowledgeBase.load(
            self.config.data_dir,
            term_inventory_path=self.config.term_inventory,
            structured_knowledge_path=self.config.structured_knowledge,
            index_dir=index_dir,
            embedder=index_embedder,
        )
        self.output_frames = load_output_frames(
            self.config.data_dir, output_frames_path=self.config.output_frames
        )
        self.parser = QuestionParser(self.kb, output_frames=self.output_frames)
        self._sessions: dict[str, ConversationSession] = {}
        self._sessions_lock = threading.Lock()
        # LLM 语义兜底需要 client；开启对应开关时才构造，失败则回退到纯规则。
        input_verifier_client = self._maybe_build_verifier_client(
            self.config.verification.verify_input_llm
        )
        output_verifier_client = self._maybe_build_verifier_client(
            self.config.verification.verify_output_llm
        )
        self.input_verifier = InputVerifier(
            term_inventory=self.kb.term_inventory,
            structured_items=self.kb.structured_items,
            client=input_verifier_client,
            use_llm=self.config.verification.verify_input_llm,
        )
        self.output_verifier = OutputVerifier(
            term_inventory=self.kb.term_inventory,
            output_frames=self.output_frames,
            structured_items=self.kb.structured_items,
            client=output_verifier_client,
            use_llm=self.config.verification.verify_output_llm,
        )
        grounding_client = self._maybe_build_grounding_client()
        self.grounding_verifier = GroundingVerifier(
            grounding_client,
            strict=self.config.verification.grounding_strict,
            minimum_score=self.config.verification.grounding_minimum_score,
        )
        index_path = Path(self.config.data_dir) / "pictures" / "extracted" / "image_index.json"
        self.image_index = ImageIndex.load(index_path)

    @staticmethod
    def _maybe_build_index_embedder(index_dir: Optional[str]) -> Optional[SiliconFlowClient]:
        if not index_dir:
            return None
        try:
            manifest = json.loads((Path(index_dir) / "manifest.json").read_text(encoding="utf-8"))
            if manifest.get("embedding", {}).get("status") != "ready":
                return None
            return SiliconFlowClient.from_env()
        except (OSError, ValueError, json.JSONDecodeError, SiliconFlowError):
            return None

    def get_session(self, session_id: str) -> ConversationSession:
        with self._sessions_lock:
            session = self._sessions.get(session_id)
            if session is None:
                session = ConversationSession(
                    self.parser,
                    intent_resolver=self._build_intent_resolver(),
                    output_frame_resolver=self._build_output_frame_resolver(),
                    turn_classifier=self._build_turn_classifier(),
                    dynamic_clarify_question=self.config.llm_dynamic_clarify_question,
                    intent_candidate_options=self.config.intent_candidate_options,
                )
                self._sessions[session_id] = session
            return session

    def reset_session(self, session_id: str) -> None:
        with self._sessions_lock:
            if session_id in self._sessions:
                self._sessions[session_id].reset()

    def drop_session(self, session_id: str) -> None:
        with self._sessions_lock:
            self._sessions.pop(session_id, None)

    def _maybe_build_verifier_client(self, enabled: bool) -> Optional[SiliconFlowClient]:
        """Build a client for LLM-backed verification; None if disabled or unavailable."""
        if not enabled:
            return None
        try:
            return SiliconFlowClient.from_env(
                model=self.config.model,
                base_url=self.config.base_url,
                timeout=self.config.timeout,
            )
        except SiliconFlowError:
            return None

    def _maybe_build_grounding_client(self) -> Any:
        if not self.config.verification.verify_grounding:
            return None
        try:
            return build_chat_client(
                self.config.verification.grounding_provider,
                model=self.config.verification.grounding_model,
                timeout=self.config.timeout,
            )
        except ProviderError:
            return None

    def _build_intent_resolver(self):
        if not self.config.llm_clarify and not self.config.llm_intent:
            return None
        try:
            client = SiliconFlowClient.from_env(
                model=self.config.model,
                base_url=self.config.base_url,
                timeout=self.config.timeout,
            )
        except SiliconFlowError:
            return None
        return ClarificationIntentResolver(self.parser, client)

    def _build_turn_classifier(self) -> TurnClassifier:
        if not self.config.llm_clarify and not self.config.llm_intent:
            return TurnClassifier(self.parser)
        try:
            client = SiliconFlowClient.from_env(
                model=self.config.model,
                base_url=self.config.base_url,
                timeout=self.config.timeout,
            )
        except SiliconFlowError:
            return TurnClassifier(self.parser)
        return TurnClassifier(self.parser, relation_resolver=LLMTurnRelationResolver(client))

    def _build_output_frame_resolver(self) -> Optional[LLMOutputFrameResolver]:
        mode = self.config.llm_output_frame
        if mode == "false":
            return None
        try:
            client = SiliconFlowClient.from_env(
                model=self.config.model,
                base_url=self.config.base_url,
                timeout=self.config.timeout,
            )
        except SiliconFlowError:
            return None
        return LLMOutputFrameResolver(
            client,
            self.output_frames,
            mode=mode,
            confidence_threshold=self.config.llm_output_frame_confidence_threshold,
        )

    def make_chat_client(self, use_vision: bool, provider_id: Optional[str] = None) -> Any:
        return build_chat_client(
            provider_id,
            model=self.config.model,
            base_url=self.config.base_url,
            timeout=self.config.timeout,
            use_vision_model=use_vision,
        )


def verify_answer_quality(
    runtime: AgentRuntime,
    answer: str,
    structure: QuestionStructure,
    chunks: list[SourceChunk],
) -> AnswerQualityResult:
    """Run format/terminology and evidence-grounding checks behind one call."""

    should_retry = False
    hints: list[str] = []
    descriptions: list[str] = []
    if runtime.config.verification.verify_output:
        output_result = runtime.output_verifier.verify(answer, structure, source_count=len(chunks))
        should_retry = should_retry or output_result.should_retry
        if output_result.correction_hints:
            hints.append(output_result.correction_hints)
        descriptions.extend(issue.description for issue in output_result.issues)

    grounding = None
    if runtime.config.verification.verify_grounding:
        grounding = runtime.grounding_verifier.verify(answer, chunks)
        should_retry = should_retry or grounding.should_retry
        if grounding.correction_hints:
            hints.append(grounding.correction_hints)
        descriptions.extend(grounding.issue_descriptions)

    return AnswerQualityResult(
        should_retry=should_retry,
        correction_hints="\n".join(hints),
        issue_descriptions=list(dict.fromkeys(descriptions)),
        grounding=grounding,
    )


def serialize_chunk(chunk: SourceChunk) -> dict[str, Any]:
    return {
        "id": chunk.id,
        "title": chunk.title,
        "source": chunk.source,
        "citation": chunk.citation(),
        "layer": chunk.layer,
        "score": chunk.score,
        "start_line": chunk.start_line,
        "end_line": chunk.end_line,
        "terms": list(chunk.terms),
        "text": chunk.text,
    }


def serialize_structure(structure: Optional[QuestionStructure]) -> Optional[dict[str, Any]]:
    if structure is None:
        return None
    return structure.to_dict()


def create_app(config_path: Optional[str] = None) -> Flask:
    runtime = AgentRuntime(config_path=config_path)
    app = Flask(
        __name__,
        static_folder=str(FRONTEND_DIR),
        static_url_path="",
    )
    CORS(app)

    app.config["AGENT_RUNTIME"] = runtime

    # 论文 RAG 子系统(独立 Blueprint,挂在 /api/papers 下)。
    # 延迟到运行时按需加载,语料/索引缺失不影响词典功能。
    # 用绝对导入以兼容不同启动方式(python web/backend/app.py、gunicorn
    # web.backend.app:app 等);裸模块名仅在前者下可解析,故先确保
    # web/backend/ 在 sys.path 中。
    try:
        backend_dir = str(Path(__file__).resolve().parent)
        if backend_dir not in sys.path:
            sys.path.insert(0, backend_dir)
        from papers_api import create_paper_blueprint

        app.register_blueprint(create_paper_blueprint(config_path=config_path))
    except Exception:  # noqa: BLE001 - 论文子系统不可用时不应阻断词典服务
        import traceback

        print("[papers] 论文 RAG 子系统未加载:", file=sys.stderr)
        traceback.print_exc()

    @app.get("/")
    def index() -> Response:
        index_path = FRONTEND_DIR / "index.html"
        if not index_path.exists():
            return Response("frontend/index.html not found", status=404)
        return send_from_directory(str(FRONTEND_DIR), "index.html")

    @app.get("/api/images/<path:filename>")
    def serve_image(filename: str) -> Response:
        if not EXTRACTED_IMAGES_DIR.exists():
            return Response("Image directory not found", status=404)
        return send_from_directory(str(EXTRACTED_IMAGES_DIR), filename)

    @app.get("/api/health")
    def health() -> Response:
        rt: AgentRuntime = app.config["AGENT_RUNTIME"]
        info = {
            "status": "ok",
            "data_dir": rt.config.data_dir,
            "model": rt.config.model,
            "base_url": rt.config.base_url,
            "top_k": rt.config.top_k,
            "term_count": len(rt.kb.term_inventory.all_terms()),
            "doc_count": len(rt.kb.chunks),
            "index_chunk_count": rt.kb.index_chunk_count,
            "retrieval_mode": rt.kb.retriever.mode if rt.kb.retriever else "legacy",
        }
        return jsonify(info)

    @app.get("/api/intents")
    def intents() -> Response:
        rt: AgentRuntime = app.config["AGENT_RUNTIME"]
        return jsonify({"frames": {k: list(v) for k, v in rt.output_frames.frames.items()}})

    @app.get("/api/intents_list")
    def intents_list() -> Response:
        from gesture_agent.learning.question_parser import INTENT_LABELS
        items = [{"value": k, "label": v} for k, v in INTENT_LABELS.items()]
        return jsonify({"intents": items})

    @app.get("/api/providers")
    def providers() -> Response:
        return jsonify({"providers": list_providers()})

    @app.post("/api/reset")
    def reset() -> Response:
        payload = request.get_json(silent=True) or {}
        session_id = payload.get("session_id") or ""
        if not session_id:
            return jsonify({"error": "missing session_id"}), 400
        rt: AgentRuntime = app.config["AGENT_RUNTIME"]
        rt.reset_session(session_id)
        return jsonify({"status": "ok", "session_id": session_id})

    @app.post("/api/drop")
    def drop() -> Response:
        payload = request.get_json(silent=True) or {}
        session_id = payload.get("session_id") or ""
        if not session_id:
            return jsonify({"error": "missing session_id"}), 400
        rt: AgentRuntime = app.config["AGENT_RUNTIME"]
        rt.drop_session(session_id)
        return jsonify({"status": "ok", "session_id": session_id})

    @app.post("/api/session")
    def new_session() -> Response:
        return jsonify({"session_id": uuid.uuid4().hex})

    @app.post("/api/ask")
    def ask() -> Response:
        rt: AgentRuntime = app.config["AGENT_RUNTIME"]
        payload = request.get_json(silent=True) or {}
        question = (payload.get("question") or "").strip()
        session_id = payload.get("session_id") or uuid.uuid4().hex
        image_paths = payload.get("images") or []
        style = payload.get("style") or "concise"
        provider_id = resolve_provider_id(payload.get("provider"))
        intent_raw = payload.get("intent") or ""
        intent = intent_raw.strip() if intent_raw.strip() and intent_raw.strip() != "auto" else None
        background = (payload.get("background") or "").strip() or None
        if not question:
            return jsonify({"error": "missing question"}), 400

        log_user_question(
            session_id,
            question,
            endpoint="/api/ask",
            image_count=len(image_paths),
            style=style,
            intent=intent_raw,
            background=bool(background),
        )

        session = rt.get_session(session_id)
        session_result = session.receive(question, image_paths=image_paths, forced_intent=intent)

        if session_result.status == "clarify":
            return jsonify(
                {
                    "session_id": session_id,
                    "status": "clarify",
                    "message": session_result.message,
                    "options": session_result.options,
                }
            )

        structure = session_result.structure

        # --- 输入验证 ---
        input_corrections = ""
        if structure and rt.config.verification.verify_input:
            iv_result = rt.input_verifier.verify(structure)
            if iv_result.status == "corrected" and iv_result.corrected_structure:
                input_corrections = iv_result.correction_summary
                structure = iv_result.corrected_structure

        chunks = rt.kb.search(
            session_result.resolved_query,
            top_k=rt.config.top_k,
            prefer_terms=structure.terms if structure else None,
        )

        available_images = []
        if rt.image_index:
            available_images = rt.image_index.search(
                structure.terms if structure else [],
                chunks,
                top_k=3,
            )

        # retrieval_instruction 快速路径：跳过 LLM，直接返回检索内容
        if structure and structure.intent == "retrieval_instruction":
            answer = build_retrieval_answer(chunks, session_result.resolved_query, available_images)
            if structure is not None:
                session.record_turn(
                    user_query=session_result.user_query,
                    resolved_query=session_result.resolved_query,
                    structure=structure,
                )
            answer = resolve_image_refs(answer, rt.image_index)
            return jsonify(
                {
                    "session_id": session_id,
                    "status": "ready",
                    "answer": answer,
                    "structure": serialize_structure(structure),
                    "chunks": [serialize_chunk(c) for c in chunks],
                    "memory_context": session_result.memory_context,
                }
            )
        # --- 检索指令快速路径结束 ---

        answer = ""
        error_text: Optional[str] = None
        try:
            image_urls = [image_path_to_data_url(p) for p in image_paths]
            messages = build_messages(
                structure,
                chunks,
                image_urls=image_urls,
                memory_context=session_result.memory_context,
                term_inventory=rt.kb.term_inventory,
                prompt_config=rt.config.prompt,
                available_images=available_images,
                style=style,
                background=background,
            )
            client = rt.make_chat_client(use_vision=bool(image_paths), provider_id=provider_id)
            answer = client.chat(
                messages,
                temperature=rt.config.temperature,
                max_tokens=rt.config.max_tokens,
                enable_thinking=rt.config.enable_thinking,
            )
        except ProviderError as exc:
            error_text = str(exc)

        # --- 输出验证 ---
        output_issues: list[str] = []
        grounding_report: Optional[GroundingReport] = None
        verification_enabled = (
            rt.config.verification.verify_output or rt.config.verification.verify_grounding
        )
        if answer and not error_text and verification_enabled and structure:
            quality_result = verify_answer_quality(rt, answer, structure, chunks)
            max_retries = rt.config.verification.output_max_retries
            for _ in range(max_retries):
                if not quality_result.should_retry:
                    break
                correction_prompt = OUTPUT_CORRECTION_PROMPT.format(
                    correction_hints=quality_result.correction_hints
                )
                retry_messages = messages + [
                    {"role": "assistant", "content": answer},
                    {"role": "user", "content": correction_prompt},
                ]
                try:
                    answer = client.chat(
                        retry_messages,
                        temperature=rt.config.temperature,
                        max_tokens=rt.config.max_tokens,
                        enable_thinking=rt.config.enable_thinking,
                    )
                except ProviderError:
                    break
                quality_result = verify_answer_quality(rt, answer, structure, chunks)
            output_issues = quality_result.issue_descriptions
            grounding_report = quality_result.grounding
            if rt.config.verification.verify_output:
                answer = rt.output_verifier.normalize(answer, structure)

        if structure is not None and not error_text:
            session.record_turn(
                user_query=session_result.user_query,
                resolved_query=session_result.resolved_query,
                structure=structure,
            )

        answer = resolve_image_refs(answer, rt.image_index)

        body = {
            "session_id": session_id,
            "status": "ready",
            "answer": answer,
            "error": error_text,
            "input_corrections": input_corrections,
            "output_issues": output_issues,
            "grounding": grounding_report.to_dict() if grounding_report else None,
            "structure": serialize_structure(structure),
            "chunks": [serialize_chunk(c) for c in chunks],
            "memory_context": session_result.memory_context,
        }
        status_code = 200 if not error_text else 502
        return jsonify(body), status_code

    @app.post("/api/ask_stream")
    def ask_stream() -> Response:
        rt: AgentRuntime = app.config["AGENT_RUNTIME"]
        payload = request.get_json(silent=True) or {}
        question = (payload.get("question") or "").strip()
        session_id = payload.get("session_id") or uuid.uuid4().hex
        image_paths = payload.get("images") or []
        style = payload.get("style") or "concise"
        provider_id = resolve_provider_id(payload.get("provider"))
        intent_raw = payload.get("intent") or ""
        intent = intent_raw.strip() if intent_raw.strip() and intent_raw.strip() != "auto" else None
        background = (payload.get("background") or "").strip() or None
        if not question:
            return jsonify({"error": "missing question"}), 400

        log_user_question(
            session_id,
            question,
            endpoint="/api/ask_stream",
            image_count=len(image_paths),
            style=style,
            intent=intent_raw,
            background=bool(background),
        )

        session = rt.get_session(session_id)
        session_result = session.receive(question, image_paths=image_paths, forced_intent=intent)

        def sse(event: str, data: dict[str, Any]) -> str:
            return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

        if session_result.status == "clarify":
            def clarify_only():
                yield sse("clarify", {"message": session_result.message, "options": session_result.options, "session_id": session_id})
                yield sse("done", {"session_id": session_id})
            return Response(stream_with_context(clarify_only()), mimetype="text/event-stream")

        structure = session_result.structure

        # --- 输入验证 ---
        input_corrections = ""
        if structure and rt.config.verification.verify_input:
            iv_result = rt.input_verifier.verify(structure)
            if iv_result.status == "corrected" and iv_result.corrected_structure:
                input_corrections = iv_result.correction_summary
                structure = iv_result.corrected_structure

        chunks = rt.kb.search(
            session_result.resolved_query,
            top_k=rt.config.top_k,
            prefer_terms=structure.terms if structure else None,
        )

        available_images = []
        if rt.image_index:
            available_images = rt.image_index.search(
                structure.terms if structure else [],
                chunks,
                top_k=3,
            )

        # retrieval_instruction 快速路径：跳过 LLM，直接返回检索内容
        if structure and structure.intent == "retrieval_instruction":
            answer = build_retrieval_answer(chunks, session_result.resolved_query, available_images)

            def retrieval_stream():
                yield sse(
                    "meta",
                    {
                        "session_id": session_id,
                        "structure": serialize_structure(structure),
                        "chunks": [serialize_chunk(c) for c in chunks],
                        "memory_context": session_result.memory_context,
                        "input_corrections": input_corrections,
                    },
                )
                # 将答案按小块流式输出，模拟打字效果
                chunk_size = 60
                for i in range(0, len(answer), chunk_size):
                    yield sse("delta", {"text": answer[i : i + chunk_size]})
                if structure is not None:
                    session.record_turn(
                        user_query=session_result.user_query,
                        resolved_query=session_result.resolved_query,
                        structure=structure,
                    )
                full = resolve_image_refs(answer, rt.image_index)
                yield sse("done", {"session_id": session_id, "answer": full, "output_issues": []})

            return Response(stream_with_context(retrieval_stream()), mimetype="text/event-stream")
        # --- 检索指令快速路径结束 ---

        def generate():
            yield sse(
                "meta",
                {
                    "session_id": session_id,
                    "structure": serialize_structure(structure),
                    "chunks": [serialize_chunk(c) for c in chunks],
                    "memory_context": session_result.memory_context,
                    "input_corrections": input_corrections,
                },
            )
            answer_parts: list[str] = []
            try:
                image_urls = [image_path_to_data_url(p) for p in image_paths]
                messages = build_messages(
                    structure,
                    chunks,
                    image_urls=image_urls,
                    memory_context=session_result.memory_context,
                    term_inventory=rt.kb.term_inventory,
                    prompt_config=rt.config.prompt,
                    available_images=available_images,
                    style=style,
                    background=background,
                )
                client = rt.make_chat_client(use_vision=bool(image_paths), provider_id=provider_id)
                for delta in client.chat_stream(
                    messages,
                    temperature=rt.config.temperature,
                    max_tokens=rt.config.max_tokens,
                    enable_thinking=rt.config.enable_thinking,
                ):
                    answer_parts.append(delta)
                    yield sse("delta", {"text": delta})
            except ProviderError as exc:
                yield sse("error", {"message": str(exc)})
                yield sse("done", {"session_id": session_id})
                return
            except Exception as exc:  # pragma: no cover - defensive
                yield sse("error", {"message": f"unexpected error: {exc}"})
                yield sse("done", {"session_id": session_id})
                return

            full_answer = "".join(answer_parts)

            # --- 输出验证 ---
            output_issues: list[str] = []
            grounding_report: Optional[GroundingReport] = None
            verification_enabled = (
                rt.config.verification.verify_output or rt.config.verification.verify_grounding
            )
            if full_answer and verification_enabled and structure:
                quality_result = verify_answer_quality(rt, full_answer, structure, chunks)
                max_retries = rt.config.verification.output_max_retries
                for _ in range(max_retries):
                    if not quality_result.should_retry:
                        break
                    correction_prompt = OUTPUT_CORRECTION_PROMPT.format(
                        correction_hints=quality_result.correction_hints
                    )
                    retry_messages = messages + [
                        {"role": "assistant", "content": full_answer},
                        {"role": "user", "content": correction_prompt},
                    ]
                    try:
                        yield sse("retry", {"reason": quality_result.correction_hints})
                        full_answer = client.chat(
                            retry_messages,
                            temperature=rt.config.temperature,
                            max_tokens=rt.config.max_tokens,
                            enable_thinking=rt.config.enable_thinking,
                        )
                        yield sse("replace", {"text": resolve_image_refs(full_answer, rt.image_index)})
                    except ProviderError:
                        break
                    quality_result = verify_answer_quality(rt, full_answer, structure, chunks)
                output_issues = quality_result.issue_descriptions
                grounding_report = quality_result.grounding
                if rt.config.verification.verify_output:
                    normalized_answer = rt.output_verifier.normalize(full_answer, structure)
                    if normalized_answer != full_answer:
                        full_answer = normalized_answer
                        yield sse("replace", {"text": resolve_image_refs(full_answer, rt.image_index)})

            if structure is not None:
                session.record_turn(
                    user_query=session_result.user_query,
                    resolved_query=session_result.resolved_query,
                    structure=structure,
                )
            full_answer = resolve_image_refs(full_answer, rt.image_index)
            yield sse("done", {
                "session_id": session_id,
                "answer": full_answer,
                "output_issues": output_issues,
                "grounding": grounding_report.to_dict() if grounding_report else None,
            })

        return Response(stream_with_context(generate()), mimetype="text/event-stream")

    return app


def main() -> None:
    config_path = os.environ.get("AGENT_CONFIG_PATH")
    host = os.environ.get("WEB_HOST", "127.0.0.1")
    port = int(os.environ.get("WEB_PORT", "5000"))
    debug = os.environ.get("WEB_DEBUG", "0") in {"1", "true", "True"}
    app = create_app(config_path=config_path)
    app.run(host=host, port=port, debug=debug, threaded=True)


if __name__ == "__main__":
    main()
