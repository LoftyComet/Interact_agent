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
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

# Make the project root importable so `gesture_agent` resolves regardless of cwd.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flask import Flask, Response, jsonify, request, send_from_directory, stream_with_context
from flask_cors import CORS

from gesture_agent.core.models import QuestionStructure, SourceChunk
from gesture_agent.knowledge import KnowledgeBase
from gesture_agent.knowledge.image_index import ImageIndex
from gesture_agent.learning import ClarificationIntentResolver, ConversationSession, LLMOutputFrameResolver, QuestionParser
from gesture_agent.learning.output_frames import load_output_frames
from gesture_agent.learning.prompt_builder import build_messages
from gesture_agent.media import image_path_to_data_url
from gesture_agent.providers import SiliconFlowClient, SiliconFlowError
from gesture_agent.settings.app_config import load_agent_config
from gesture_agent.verification import InputVerifier, OutputVerifier
from gesture_agent.verification.prompts import OUTPUT_CORRECTION_PROMPT


FRONTEND_DIR = (Path(__file__).resolve().parent.parent / "frontend").resolve()
EXTRACTED_IMAGES_DIR = (PROJECT_ROOT / "data" / "pictures" / "extracted").resolve()

IMAGE_REF_RE = re.compile(r"!\[([^\]]*)\]\(image:([^)]+)\)")


def resolve_image_refs(text: str, image_index: Optional[ImageIndex], base_url: str = "/api/images") -> str:
    if image_index is None:
        return text

    def _replacer(match):
        alt_text = match.group(1)
        image_id = match.group(2)
        filename = image_index.get_filename(image_id)
        if filename:
            return f"![{alt_text}]({base_url}/{filename})"
        return match.group(0)

    return IMAGE_REF_RE.sub(_replacer, text)


class AgentRuntime:
    """Holds the singletons the request handlers need."""

    def __init__(self, config_path: Optional[str] = None) -> None:
        self.config = load_agent_config(config_path)
        self.kb = KnowledgeBase.load(
            self.config.data_dir,
            term_inventory_path=self.config.term_inventory,
            structured_knowledge_path=self.config.structured_knowledge,
        )
        self.output_frames = load_output_frames(
            self.config.data_dir, output_frames_path=self.config.output_frames
        )
        self.parser = QuestionParser(self.kb, output_frames=self.output_frames)
        self._sessions: dict[str, ConversationSession] = {}
        self._sessions_lock = threading.Lock()
        self.input_verifier = InputVerifier(
            term_inventory=self.kb.term_inventory,
            structured_items=self.kb.structured_items,
            use_llm=self.config.verification.verify_input_llm,
        )
        self.output_verifier = OutputVerifier(
            term_inventory=self.kb.term_inventory,
            output_frames=self.output_frames,
            structured_items=self.kb.structured_items,
            use_llm=self.config.verification.verify_output_llm,
        )
        index_path = Path(self.config.data_dir) / "pictures" / "extracted" / "image_index.json"
        self.image_index = ImageIndex.load(index_path)

    def get_session(self, session_id: str) -> ConversationSession:
        with self._sessions_lock:
            session = self._sessions.get(session_id)
            if session is None:
                session = ConversationSession(
                    self.parser,
                    intent_resolver=self._build_intent_resolver(),
                    output_frame_resolver=self._build_output_frame_resolver(),
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

    def make_chat_client(self, use_vision: bool) -> SiliconFlowClient:
        return SiliconFlowClient.from_env(
            model=self.config.model,
            base_url=self.config.base_url,
            timeout=self.config.timeout,
            use_vision_model=use_vision,
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
        }
        return jsonify(info)

    @app.get("/api/intents")
    def intents() -> Response:
        rt: AgentRuntime = app.config["AGENT_RUNTIME"]
        return jsonify({"frames": {k: list(v) for k, v in rt.output_frames.frames.items()}})

    @app.post("/api/reset")
    def reset() -> Response:
        payload = request.get_json(silent=True) or {}
        session_id = payload.get("session_id") or ""
        if not session_id:
            return jsonify({"error": "missing session_id"}), 400
        rt: AgentRuntime = app.config["AGENT_RUNTIME"]
        rt.reset_session(session_id)
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
        if not question:
            return jsonify({"error": "missing question"}), 400

        session = rt.get_session(session_id)
        session_result = session.receive(question, image_paths=image_paths)

        if session_result.status == "clarify":
            return jsonify(
                {
                    "session_id": session_id,
                    "status": "clarify",
                    "message": session_result.message,
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
            )
            client = rt.make_chat_client(use_vision=bool(image_paths))
            answer = client.chat(
                messages,
                temperature=rt.config.temperature,
                max_tokens=rt.config.max_tokens,
                enable_thinking=rt.config.enable_thinking,
            )
        except SiliconFlowError as exc:
            error_text = str(exc)

        # --- 输出验证 ---
        output_issues: list[str] = []
        if answer and not error_text and rt.config.verification.verify_output and structure:
            ov_result = rt.output_verifier.verify(answer, structure)
            if ov_result.should_retry:
                correction_prompt = OUTPUT_CORRECTION_PROMPT.format(correction_hints=ov_result.correction_hints)
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
                except SiliconFlowError:
                    pass
            output_issues = [i.description for i in ov_result.issues]

        if structure is not None and not error_text:
            session.record_turn(
                user_query=session_result.user_query,
                resolved_query=session_result.resolved_query,
                structure=structure,
                answer=answer,
            )

        answer = resolve_image_refs(answer, rt.image_index)

        body = {
            "session_id": session_id,
            "status": "ready",
            "answer": answer,
            "error": error_text,
            "input_corrections": input_corrections,
            "output_issues": output_issues,
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
        if not question:
            return jsonify({"error": "missing question"}), 400

        session = rt.get_session(session_id)
        session_result = session.receive(question, image_paths=image_paths)

        def sse(event: str, data: dict[str, Any]) -> str:
            return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

        if session_result.status == "clarify":
            def clarify_only():
                yield sse("clarify", {"message": session_result.message, "session_id": session_id})
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
                )
                client = rt.make_chat_client(use_vision=bool(image_paths))
                for delta in client.chat_stream(
                    messages,
                    temperature=rt.config.temperature,
                    max_tokens=rt.config.max_tokens,
                    enable_thinking=rt.config.enable_thinking,
                ):
                    answer_parts.append(delta)
                    yield sse("delta", {"text": delta})
            except SiliconFlowError as exc:
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
            if full_answer and rt.config.verification.verify_output and structure:
                ov_result = rt.output_verifier.verify(full_answer, structure)
                if ov_result.should_retry:
                    correction_prompt = OUTPUT_CORRECTION_PROMPT.format(correction_hints=ov_result.correction_hints)
                    retry_messages = messages + [
                        {"role": "assistant", "content": full_answer},
                        {"role": "user", "content": correction_prompt},
                    ]
                    try:
                        yield sse("retry", {"reason": ov_result.correction_hints})
                        full_answer = client.chat(
                            retry_messages,
                            temperature=rt.config.temperature,
                            max_tokens=rt.config.max_tokens,
                            enable_thinking=rt.config.enable_thinking,
                        )
                        full_answer = resolve_image_refs(full_answer, rt.image_index)
                        yield sse("replace", {"text": full_answer})
                    except SiliconFlowError:
                        pass
                output_issues = [i.description for i in ov_result.issues]

            if structure is not None:
                session.record_turn(
                    user_query=session_result.user_query,
                    resolved_query=session_result.resolved_query,
                    structure=structure,
                    answer=full_answer,
                )
            full_answer = resolve_image_refs(full_answer, rt.image_index)
            yield sse("done", {"session_id": session_id, "answer": full_answer, "output_issues": output_issues})

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
