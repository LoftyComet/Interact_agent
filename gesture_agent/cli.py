from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from .core.models import QuestionStructure
from .knowledge import KnowledgeBase
from .learning import ClarificationIntentResolver, ConversationSession, LLMIntentResolver, QuestionParser
from .learning.output_frames import load_output_frames
from .learning.prompt_builder import build_messages
from .media import image_path_to_data_url
from .providers import SiliconFlowClient, SiliconFlowError
from .settings.app_config import DEFAULT_AGENT_CONFIG_PATH, load_agent_config


@dataclass
class RunResult:
    exit_code: int
    answer: str = ""
    structure: Optional[QuestionStructure] = None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gesture dictionary learning agent.")
    parser.add_argument("question", nargs="*", help="用户问题。留空时默认进入连续问答。")
    parser.add_argument("--config", default=None, help=f"Agent 运行配置 JSON；默认自动读取 {DEFAULT_AGENT_CONFIG_PATH}。")
    parser.add_argument("--data-dir", default=None, help="Markdown 资料目录，默认 data。")
    parser.add_argument("--term-inventory", default=None, help="术语枚举配置 JSON；默认尝试读取 data/term_inventory.json。")
    parser.add_argument("--show-term-inventory", action="store_true", default=None, help="输出当前生效的术语枚举并退出。")
    parser.add_argument("--export-term-inventory", default=None, help="把当前生效的术语枚举导出到指定 JSON 文件并退出。")
    parser.add_argument("--output-frames", default=None, help="Intent 输出框架配置 JSON；默认尝试读取 data/output_frames.json。")
    parser.add_argument("--show-output-frames", action="store_true", default=None, help="输出当前生效的 Intent 输出框架并退出。")
    parser.add_argument("--export-output-frames", default=None, help="把当前生效的 Intent 输出框架导出到指定 JSON 文件并退出。")
    parser.add_argument("--top-k", type=int, default=None, help="检索资料片段数量。")
    parser.add_argument("--image", action="append", default=None, help="图片案例路径，可重复传入。")
    parser.add_argument(
        "--model",
        default=None,
        help="SiliconFlow 模型名。默认读取 SILICONFLOW_MODEL；传入 --image 时优先读取 SILICONFLOW_VISION_MODEL。",
    )
    parser.add_argument("--base-url", default=None, help="SiliconFlow base URL。")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=None, help="模型最大输出 token 数；回答半句停止通常需要调大它。")
    parser.add_argument("--timeout", type=int, default=None, help="API 请求超时时间，单位秒。默认读取 SILICONFLOW_TIMEOUT 或 60。")
    parser.add_argument("--stream", action="store_true", default=None, help="使用流式输出，避免长回答结束前终端无反馈。")
    parser.add_argument("--check-api", action="store_true", default=None, help="只检查 SiliconFlow API 连通性和模型列表，不读取词典、不提问。")
    parser.add_argument("--show-structure", action="store_true", default=None, help="输出问题结构。")
    parser.add_argument("--show-context", action="store_true", default=None, help="输出检索到的资料标题。")
    parser.add_argument("--dry-run", action="store_true", default=None, help="只做拆解和检索，不调用 API。")
    parser.add_argument("--interactive", action="store_true", default=None, help="强制进入连续问答模式。")
    parser.add_argument("--llm-intent", action="store_true", default=None, help="每轮都使用硅基流动大模型辅助判断 intent；失败时回退到本地规则。")
    parser.add_argument("--no-llm-clarify", action="store_true", default=None, help="关闭“本地规则信息不足时用大模型二次判断”的默认行为。")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    raw_args = build_arg_parser().parse_args(argv)
    try:
        args = apply_agent_config(raw_args)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if args.check_api:
        return check_api(args)

    kb = KnowledgeBase.load(args.data_dir, term_inventory_path=args.term_inventory)
    if args.show_term_inventory:
        print(json.dumps(asdict(kb.term_inventory), ensure_ascii=False, indent=2))
        return 0
    if args.export_term_inventory:
        output_path = Path(args.export_term_inventory)
        output_path.write_text(json.dumps(asdict(kb.term_inventory), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"已导出术语枚举：{output_path}")
        return 0

    output_frames = load_output_frames(args.data_dir, output_frames_path=args.output_frames)
    if args.show_output_frames:
        print(json.dumps(asdict(output_frames), ensure_ascii=False, indent=2))
        return 0
    if args.export_output_frames:
        output_path = Path(args.export_output_frames)
        output_path.write_text(json.dumps(asdict(output_frames), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"已导出 Intent 输出框架：{output_path}")
        return 0

    parser = QuestionParser(kb, output_frames=output_frames)

    question = " ".join(args.question).strip()
    if args.interactive or (not question and args.default_interactive):
        return interactive_loop(args, kb, parser)

    if not question:
        print("请提供问题，或使用 --interactive。", file=sys.stderr)
        return 2
    result = run_once(args, kb, parser, question)
    return result.exit_code


def apply_agent_config(args: argparse.Namespace) -> argparse.Namespace:
    config = load_agent_config(args.config)
    args.config_source = config.source
    args.data_dir = _pick(args.data_dir, config.data_dir)
    args.term_inventory = _pick(args.term_inventory, config.term_inventory)
    args.output_frames = _pick(args.output_frames, config.output_frames)
    args.top_k = _pick(args.top_k, config.top_k)
    args.image = args.image if args.image is not None else list(config.images)
    args.model = _pick(args.model, config.model)
    args.base_url = _pick(args.base_url, config.base_url)
    args.temperature = _pick(args.temperature, config.temperature)
    args.max_tokens = _pick(args.max_tokens, config.max_tokens)
    args.timeout = _pick(args.timeout, config.timeout)
    args.enable_thinking = config.enable_thinking
    args.stream = _pick_bool(args.stream, config.stream)
    args.check_api = _pick_bool(args.check_api, config.check_api)
    args.show_structure = _pick_bool(args.show_structure, config.show_structure)
    args.show_context = _pick_bool(args.show_context, config.show_context)
    args.dry_run = _pick_bool(args.dry_run, config.dry_run)
    args.interactive = _pick_bool(args.interactive, False)
    args.default_interactive = config.default_interactive
    args.llm_intent = _pick_bool(args.llm_intent, config.llm_intent)
    args.no_llm_clarify = _pick_bool(args.no_llm_clarify, not config.llm_clarify)
    args.show_term_inventory = _pick_bool(args.show_term_inventory, False)
    args.show_output_frames = _pick_bool(args.show_output_frames, False)
    args.prompt_config = config.prompt
    return args


def _pick(value, fallback):
    return fallback if value is None else value


def _pick_bool(value: Optional[bool], fallback: bool) -> bool:
    return bool(fallback if value is None else value)


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
        messages = build_messages(
            structure,
            chunks,
            image_urls=image_urls,
            memory_context=memory_context,
            term_inventory=kb.term_inventory,
            prompt_config=args.prompt_config,
        )
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
            enable_thinking=args.enable_thinking,
        ) if not args.stream else None
        if args.stream:
            answer_parts: list[str] = []
            for delta in client.chat_stream(
                messages,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                enable_thinking=args.enable_thinking,
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
