"""论文 RAG 的 Web API(Flask Blueprint)。

挂载到主 app(web/backend/app.py),与手势词典逻辑解耦。提供:
- GET  /api/papers/status        语料 + 索引状态(公开)
- POST /api/papers/ask           非流式问答(公开)
- POST /api/papers/ask_stream    流式问答(SSE,公开)
- POST /api/papers/fetch         后台抓取语料(需管理员令牌)
- POST /api/papers/build         后台构建向量索引(需管理员令牌)

PaperRAG 懒加载;抓取/构建在后台线程执行,通过 /status 轮询进度。

权限:fetch / build 是写操作,需在 PAPERS_ADMIN_TOKEN 环境变量配置令牌后,
请求带 `X-Admin-Token` 头(或 Authorization: Bearer <token>)校验通过才能调用。
未配置令牌时这两个接口默认关闭(返回 503),避免裸奔。
"""

from __future__ import annotations

import hmac
import json
import threading
from functools import wraps
from typing import Any, Optional

from flask import Blueprint, Response, jsonify, request, stream_with_context

from gesture_agent.providers import ProviderError
from gesture_agent.settings.env import get_env
from paper_agent.config import load_paper_config
from paper_agent.prepare import build_index, corpus_status, fetch_corpus
from paper_agent.rag import PaperRAG
from paper_agent.models import RetrievedPaper


def _extract_token() -> str:
    """从 X-Admin-Token 头或 Authorization: Bearer 取令牌。"""
    token = request.headers.get("X-Admin-Token", "")
    if token:
        return token
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth[len("Bearer "):].strip()
    return ""


def require_admin(view):
    """装饰器:校验管理员令牌。未配置令牌时拒绝(503),令牌不符时 401。"""

    @wraps(view)
    def wrapper(*args, **kwargs):
        expected = get_env("PAPERS_ADMIN_TOKEN") or ""
        if not expected:
            return jsonify({"error": "管理接口未启用:未配置 PAPERS_ADMIN_TOKEN"}), 503
        provided = _extract_token()
        # 用 hmac.compare_digest 做常数时间比较,避免时序侧信道。
        if not provided or not hmac.compare_digest(provided, expected):
            return jsonify({"error": "需要管理员令牌"}), 401
        return view(*args, **kwargs)

    return wrapper



class PaperRuntime:
    """持有论文 RAG 的单例与后台任务状态。"""

    def __init__(self, config_path: Optional[str] = None) -> None:
        self.config = load_paper_config(config_path)
        self._rag: Optional[PaperRAG] = None
        self._rag_lock = threading.Lock()
        self._job_lock = threading.Lock()
        self._job: dict[str, Any] = {"running": False, "kind": "", "logs": [], "error": ""}

    def get_rag(self, *, reload: bool = False) -> PaperRAG:
        """懒加载 PaperRAG。reload=True 时强制重建(数据更新后调用)。"""
        with self._rag_lock:
            if self._rag is None or reload:
                self._rag = PaperRAG.load(self.config.data_dir, timeout=self.config.timeout)
            return self._rag

    def invalidate_rag(self) -> None:
        with self._rag_lock:
            self._rag = None

    # --- 后台任务 ---
    def job_snapshot(self) -> dict[str, Any]:
        with self._job_lock:
            return dict(self._job)

    def _log(self, message: str) -> None:
        with self._job_lock:
            self._job["logs"].append(message)

    def start_job(self, kind: str, target) -> bool:
        """启动后台任务。已有任务在跑则返回 False。"""
        with self._job_lock:
            if self._job["running"]:
                return False
            self._job = {"running": True, "kind": kind, "logs": [], "error": ""}

        def runner():
            try:
                target(self._log)
            except Exception as exc:  # noqa: BLE001 - 后台任务需吞掉异常并上报
                with self._job_lock:
                    self._job["error"] = str(exc)
                self._log(f"任务失败:{exc}")
            finally:
                with self._job_lock:
                    self._job["running"] = False
                self.invalidate_rag()

        threading.Thread(target=runner, daemon=True).start()
        return True


def _serialize_paper(item: RetrievedPaper) -> dict[str, Any]:
    p = item.paper
    return {
        "id": p.id,
        "title": p.title,
        "authors": p.authors,
        "year": p.year,
        "venue": p.venue,
        "abstract": p.abstract,
        "doi": p.doi,
        "url": p.url,
        "citations": p.citations,
        "score": round(item.score, 4),
        "citation": p.citation(),
    }


def create_paper_blueprint(config_path: Optional[str] = None) -> Blueprint:
    runtime = PaperRuntime(config_path=config_path)
    bp = Blueprint("papers", __name__, url_prefix="/api/papers")

    @bp.get("/status")
    def status() -> Response:
        body = corpus_status(runtime.config)
        body["job"] = runtime.job_snapshot()
        # 告知前端管理功能是否启用(配置了令牌才显示抓取/构建控件)。
        body["admin_enabled"] = bool(get_env("PAPERS_ADMIN_TOKEN") or "")
        return jsonify(body)

    @bp.post("/fetch")
    @require_admin
    def fetch() -> Response:
        payload = request.get_json(silent=True) or {}
        # 允许临时覆盖 venues / 年份范围
        if payload.get("venues"):
            runtime.config.venues = list(payload["venues"])
        if payload.get("from_year"):
            runtime.config.from_year = int(payload["from_year"])
        if payload.get("to_year"):
            runtime.config.to_year = int(payload["to_year"])
        started = runtime.start_job(
            "fetch", lambda log: fetch_corpus(runtime.config, progress=log)
        )
        if not started:
            return jsonify({"error": "已有任务在运行"}), 409
        return jsonify({"status": "started", "kind": "fetch"})

    @bp.post("/build")
    @require_admin
    def build() -> Response:
        payload = request.get_json(silent=True) or {}
        rebuild = bool(payload.get("rebuild", False))
        started = runtime.start_job(
            "build", lambda log: build_index(runtime.config, rebuild=rebuild, progress=log)
        )
        if not started:
            return jsonify({"error": "已有任务在运行"}), 409
        return jsonify({"status": "started", "kind": "build"})

    @bp.post("/ask")
    def ask() -> Response:
        payload = request.get_json(silent=True) or {}
        question = (payload.get("question") or "").strip()
        top_k = int(payload.get("top_k") or runtime.config.top_k)
        if not question:
            return jsonify({"error": "missing question"}), 400
        try:
            rag = runtime.get_rag()
        except FileNotFoundError as exc:
            return jsonify({"error": f"向量索引未就绪:{exc}"}), 409
        try:
            answer, retrieved = rag.answer(
                question,
                k=top_k,
                temperature=runtime.config.temperature,
                max_tokens=runtime.config.max_tokens,
            )
        except ProviderError as exc:
            return jsonify({"error": str(exc)}), 502
        return jsonify(
            {
                "status": "ready",
                "answer": answer,
                "papers": [_serialize_paper(r) for r in retrieved],
            }
        )

    @bp.post("/ask_stream")
    def ask_stream() -> Response:
        payload = request.get_json(silent=True) or {}
        question = (payload.get("question") or "").strip()
        top_k = int(payload.get("top_k") or runtime.config.top_k)

        def sse(event: str, data: dict[str, Any]) -> str:
            return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

        if not question:
            return jsonify({"error": "missing question"}), 400

        try:
            rag = runtime.get_rag()
        except FileNotFoundError as exc:
            return jsonify({"error": f"向量索引未就绪:{exc}"}), 409

        def generate():
            try:
                stream, retrieved = rag.answer_stream(
                    question,
                    k=top_k,
                    temperature=runtime.config.temperature,
                    max_tokens=runtime.config.max_tokens,
                )
            except ProviderError as exc:
                yield sse("error", {"message": str(exc)})
                yield sse("done", {})
                return
            yield sse("meta", {"papers": [_serialize_paper(r) for r in retrieved]})
            try:
                for delta in stream:
                    yield sse("delta", {"text": delta})
            except ProviderError as exc:
                yield sse("error", {"message": str(exc)})
                yield sse("done", {})
                return
            yield sse("done", {"references": PaperRAG.references(retrieved)})

        return Response(stream_with_context(generate()), mimetype="text/event-stream")

    return bp
