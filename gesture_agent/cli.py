from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from typing import Optional

from .core.models import QuestionStructure
from .knowledge import KnowledgeBase
from .learning import ClarificationIntentResolver, ConversationSession, LLMIntentResolver, QuestionParser
from .learning.prompt_builder import build_messages
from .media import image_path_to_data_url
from .providers import SiliconFlowClient, SiliconFlowError


@dataclass
class RunResult:
    exit_code: int
    answer: str = ""
    structure: Optional[QuestionStructure] = None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gesture dictionary learning agent.")
    parser.add_argument("question", nargs="*", help="用户问题。留空并使用 --interactive 可进入连续提问。")
    parser.add_argument("--data-dir", default="data", help="Markdown 资料目录，默认 data。")
    parser.add_argument("--top-k", type=int, default=6, help="检索资料片段数量。")
    parser.add_argument("--image", action="append", default=[], help="图片案例路径，可重复传入。")
    parser.add_argument(
        "--model",
        default=None,
        help="SiliconFlow 模型名。默认读取 SILICONFLOW_MODEL；传入 --image 时优先读取 SILICONFLOW_VISION_MODEL。",
    )
    parser.add_argument("--base-url", default=None, help="SiliconFlow base URL。")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=1600, help="模型最大输出 token 数；回答半句停止通常需要调大它。")
    parser.add_argument("--timeout", type=int, default=None, help="API 请求超时时间，单位秒。默认读取 SILICONFLOW_TIMEOUT 或 60。")
    parser.add_argument("--stream", action="store_true", help="使用流式输出，避免长回答结束前终端无反馈。")
    parser.add_argument("--check-api", action="store_true", help="只检查 SiliconFlow API 连通性和模型列表，不读取词典、不提问。")
    parser.add_argument("--show-structure", action="store_true", help="输出问题结构。")
    parser.add_argument("--show-context", action="store_true", help="输出检索到的资料标题。")
    parser.add_argument("--dry-run", action="store_true", help="只做拆解和检索，不调用 API。")
    parser.add_argument("--interactive", action="store_true", help="连续问答模式。")
    parser.add_argument("--llm-intent", action="store_true", help="每轮都使用硅基流动大模型辅助判断 intent；失败时回退到本地规则。")
    parser.add_argument("--no-llm-clarify", action="store_true", help="关闭“本地规则信息不足时用大模型二次判断”的默认行为。")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.check_api:
        return check_api(args)

    kb = KnowledgeBase.load(args.data_dir)
    parser = QuestionParser(kb)

    if args.interactive:
        return interactive_loop(args, kb, parser)

    question = " ".join(args.question).strip()
    if not question:
        print("请提供问题，或使用 --interactive。", file=sys.stderr)
        return 2
    result = run_once(args, kb, parser, question)
    return result.exit_code


def interactive_loop(args: argparse.Namespace, kb: KnowledgeBase, parser: QuestionParser) -> int:
    session = ConversationSession(parser, intent_resolver=build_intent_resolver(args, parser))
    print("Gesture Agent interactive mode. 输入 exit 退出，输入 reset 清空当前澄清会话。")
    while True:
        try:
            question = input("> ").strip()
        except EOFError:
            break
        if question.lower() in {"exit", "quit", "q"}:
            break
        if question.lower() in {"reset", "clear"}:
            session.reset()
            print("已清空当前会话。")
            continue
        if not question:
            continue

        session_result = session.receive(question, image_paths=args.image)
        if session_result.status == "clarify":
            print(f"需要澄清：{session_result.message}")
            continue

        run_result = run_once(
            args,
            kb,
            parser,
            session_result.resolved_query,
            structure=session_result.structure,
            memory_context=session_result.memory_context,
        )
        if run_result.exit_code != 0:
            return run_result.exit_code
        if session_result.structure is not None:
            session.record_turn(
                user_query=session_result.user_query,
                resolved_query=session_result.resolved_query,
                structure=session_result.structure,
                answer=run_result.answer,
            )
    return 0


def run_once(
    args: argparse.Namespace,
    kb: KnowledgeBase,
    parser: QuestionParser,
    question: str,
    structure: Optional[QuestionStructure] = None,
    memory_context: str = "",
) -> RunResult:
    structure = structure or resolve_structure(args, parser, question)
    chunks = kb.search(question, top_k=args.top_k, prefer_terms=structure.terms)

    if args.show_structure or args.dry_run:
        print("问题结构：")
        print(json.dumps(structure.to_dict(), ensure_ascii=False, indent=2))

    if args.show_context or args.dry_run:
        print("\n检索资料：")
        for idx, chunk in enumerate(chunks, start=1):
            print(f"{idx}. {chunk.title} ({chunk.citation()}, score={chunk.score})")

    if args.dry_run:
        return RunResult(exit_code=0, structure=structure)

    try:
        image_urls = [image_path_to_data_url(path) for path in args.image]
        messages = build_messages(structure, chunks, image_urls=image_urls, memory_context=memory_context)
        client = SiliconFlowClient.from_env(
            model=args.model,
            base_url=args.base_url,
            timeout=args.timeout,
            use_vision_model=bool(args.image),
        )
        print(
            f"已检索到 {len(chunks)} 条资料，正在调用 SiliconFlow API，model={client.model}，timeout={client.timeout}s...",
            file=sys.stderr,
            flush=True,
        )
        answer = client.chat(
            messages,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            enable_thinking=False,
        ) if not args.stream else None
        if args.stream:
            answer_parts: list[str] = []
            for delta in client.chat_stream(
                messages,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                enable_thinking=False,
            ):
                print(delta, end="", flush=True)
                answer_parts.append(delta)
            print()
            answer = "".join(answer_parts)
    except SiliconFlowError as exc:
        if "Missing SILICONFLOW_API_KEY" in str(exc):
            print(
                "未设置 SILICONFLOW_API_KEY，已完成本地拆解与检索。把 key 填入 `.env` 或设置 shell 环境变量后可调用硅基流动 API。",
                file=sys.stderr,
            )
            if not (args.show_structure or args.show_context):
                print(json.dumps(structure.to_dict(), ensure_ascii=False, indent=2))
            return RunResult(exit_code=1, structure=structure)
        print(str(exc), file=sys.stderr)
        return RunResult(exit_code=1, structure=structure)

    if answer is not None:
        print(answer)
    return RunResult(exit_code=0, answer=answer or "", structure=structure)


def resolve_structure(args: argparse.Namespace, parser: QuestionParser, question: str) -> QuestionStructure:
    if not args.llm_intent or args.dry_run:
        return parser.parse(question, image_paths=args.image)

    resolver = build_intent_resolver(args, parser)
    if resolver is None:
        return parser.parse(question, image_paths=args.image)

    resolution = resolver.resolve(question, image_paths=args.image)
    if resolution.needs_clarification:
        print(f"需要澄清：{resolution.clarification_question}", file=sys.stderr)
    return parser.parse(question, image_paths=args.image, forced_intent=resolution.intent)


def build_intent_resolver(args: argparse.Namespace, parser: QuestionParser) -> Optional[LLMIntentResolver]:
    if args.dry_run or (args.no_llm_clarify and not args.llm_intent):
        return None
    try:
        client = SiliconFlowClient.from_env(
            model=args.model,
            base_url=args.base_url,
            timeout=args.timeout,
            use_vision_model=bool(args.image),
        )
    except SiliconFlowError as exc:
        print(f"LLM intent 判断不可用，已回退到本地规则：{exc}", file=sys.stderr)
        return None
    if args.llm_intent:
        return LLMIntentResolver(parser, client)
    return ClarificationIntentResolver(parser, client)


def check_api(args: argparse.Namespace) -> int:
    try:
        client = SiliconFlowClient.from_env(model=args.model, base_url=args.base_url, timeout=args.timeout)
        print(f"正在检查 SiliconFlow API，base_url={client.base_url}，timeout={client.timeout}s...", file=sys.stderr)
        models = client.list_models(limit=30)
    except SiliconFlowError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"API 连通。当前配置模型：{client.model}")
    if models:
        print("可见模型示例：")
        for model in models:
            marker = " *当前配置" if model == client.model else ""
            print(f"- {model}{marker}")
    else:
        print("API 返回成功，但没有解析到模型列表。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
