from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional, Union

from gesture_agent.core.models import Intent

logger = logging.getLogger(__name__)

DEFAULT_INTENT_EXAMPLES_FILENAME = "intent_examples.json"

# 近似命中阈值：超过则把样例 intent 作为一个强规则候选。
NEAR_MATCH_THRESHOLD = 0.86
NEAR_MATCH_SCORE = 0.95
# 近乎完全一致（标注样例本身就是这条问题）时给更高分，
# 让人工标注的权威标签能压过纯 regex 的对比/案例信号（如 0.98 的 interaction_compare）。
EXACT_MATCH_THRESHOLD = 0.97
EXACT_MATCH_SCORE = 0.995
# 进入大模型提示词的参考示例数量上限。
MAX_PROMPT_EXAMPLES = 6
# 进入提示词的最低相关度，过滤掉完全不相干的样例。
PROMPT_RELEVANCE_FLOOR = 0.18


@dataclass
class IntentExample:
    question: str
    intent: Intent
    note: str = ""


@dataclass
class IntentExampleMatch:
    example: IntentExample
    similarity: float


class IntentExampleBank:
    """已标注优质问题样例库，用于辅助意图识别。"""

    def __init__(self, examples: list[IntentExample]) -> None:
        self.examples = examples

    @property
    def is_empty(self) -> bool:
        return not self.examples

    def best_match(self, query: str) -> Optional[IntentExampleMatch]:
        """返回与 query 文本相似度最高的样例。"""
        best: Optional[IntentExampleMatch] = None
        for example in self.examples:
            similarity = _text_similarity(query, example.question)
            if best is None or similarity > best.similarity:
                best = IntentExampleMatch(example=example, similarity=similarity)
        return best

    def relevant(
        self,
        query: str,
        *,
        limit: int = MAX_PROMPT_EXAMPLES,
        floor: float = PROMPT_RELEVANCE_FLOOR,
    ) -> list[IntentExampleMatch]:
        """返回与 query 最相关的若干样例（按相似度降序），用于提示词参考。"""
        scored = [
            IntentExampleMatch(example=example, similarity=_text_similarity(query, example.question))
            for example in self.examples
        ]
        scored = [match for match in scored if match.similarity >= floor]
        scored.sort(key=lambda match: match.similarity, reverse=True)
        return scored[:limit]


def _text_similarity(a: str, b: str) -> float:
    a = (a or "").strip()
    b = (b or "").strip()
    if not a or not b:
        return 0.0
    ratio = SequenceMatcher(None, a, b).ratio()
    bigram = _bigram_overlap(a, b)
    return max(ratio, bigram)


def _bigram_overlap(a: str, b: str) -> float:
    grams_a = _char_bigrams(a)
    grams_b = _char_bigrams(b)
    if not grams_a or not grams_b:
        return 0.0
    intersection = len(grams_a & grams_b)
    return 2 * intersection / (len(grams_a) + len(grams_b))


def _char_bigrams(text: str) -> set[str]:
    cleaned = "".join(ch for ch in text if not ch.isspace())
    if len(cleaned) < 2:
        return {cleaned} if cleaned else set()
    return {cleaned[i : i + 2] for i in range(len(cleaned) - 1)}


def load_intent_examples(
    data_dir: Union[str, Path] = "data",
    examples_path: Optional[Union[str, Path]] = None,
) -> IntentExampleBank:
    path = Path(examples_path) if examples_path else Path(data_dir) / DEFAULT_INTENT_EXAMPLES_FILENAME
    if not path.exists():
        return IntentExampleBank(examples=[])

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to read intent examples from %s: %s", path, exc)
        return IntentExampleBank(examples=[])

    valid_intents = set(Intent.__args__)  # type: ignore[attr-defined]
    items = raw.get("examples", []) if isinstance(raw, dict) else raw
    if not isinstance(items, list):
        logger.warning("Intent examples `examples` must be a list in %s", path)
        return IntentExampleBank(examples=[])

    examples: list[IntentExample] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        question = str(item.get("question", "")).strip()
        intent = item.get("intent")
        if not question or intent not in valid_intents:
            continue
        examples.append(
            IntentExample(question=question, intent=intent, note=str(item.get("note", "")).strip())
        )

    return IntentExampleBank(examples=examples)
