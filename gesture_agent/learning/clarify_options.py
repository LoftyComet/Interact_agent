"""意图子类型的选项式追问（通用机制）。

某些意图在正式作答前，需要用户先选定一个子类型——例如「语音交互」要先确认
是「含义识别类」还是「非语言声学控制类」。这类追问不适合只在回答正文里用文字提一句，
而应触发前端交互（可点击的选项按钮）。

本模块用一张配置表描述「意图 → 子类型选项 + 触发判定」：
- 当 intent 命中配置、且用户在问题里还没点明子类型时，返回需要追问的选项；
- 当用户已经点明（命中某子类型的关键词），返回 None，照常进入作答。

选项里的 ``value`` 是用户点击后回填、作为下一轮输入的文本——它会再次进入
``ConversationSession.receive``，这次能被关键词判定为「已指明类别」，从而直接作答。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from gesture_agent.core.models import Intent


@dataclass(frozen=True)
class SubTypeOption:
    """一个子类型选项。"""

    label: str
    value: str
    desc: str = ""
    # 命中其中任一关键词，即认为用户已经点明这个子类型，无需再追问。
    keywords: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, str]:
        return {"label": self.label, "value": self.value, "desc": self.desc}


@dataclass(frozen=True)
class ClarifyOptionSpec:
    """某个意图的选项式追问配置。"""

    message: str
    options: tuple[SubTypeOption, ...] = field(default_factory=tuple)

    def matched_subtype(self, text: str) -> Optional[SubTypeOption]:
        """若文本已点明某个子类型，返回该选项；否则 None。"""
        for option in self.options:
            for kw in option.keywords:
                if re.search(re.escape(kw), text):
                    return option
        return None


# —— 配置表：意图 -> 选项式追问 ——
# voice_interaction 是首个使用者；未来其他意图按同样结构追加即可。
CLARIFY_OPTION_SPECS: dict[Intent, ClarifyOptionSpec] = {
    "voice_interaction": ClarifyOptionSpec(
        message="声音交互可分为两类，你想了解哪一类？",
        options=(
            SubTypeOption(
                label="含义识别类",
                value="含义识别类语音交互：把语音内容识别为含义或指令的会话式间接控制",
                desc="把语音内容识别为含义/指令的会话式间接控制（对应词典“含义识别”机制）",
                keywords=("含义", "语义", "指令", "口令", "对话", "识别内容", "说什么"),
            ),
            SubTypeOption(
                label="非语言声学控制类",
                value="非语言声学控制类语音交互：用音高、音量、持续、舌音等非语言声学特征做直接即时控制",
                desc=(
                    "用音高/音量/持续/舌音等非语言声学特征做直接即时控制"
                    "（文章参考：Igarashi & Hughes, “Voice as Sound”, UIST 2001）"
                ),
                keywords=("音高", "音量", "持续", "发声", "舌音", "声学", "音调", "直接控制", "连续发声"),
            ),
        ),
    ),
}


def get_clarify_spec(intent: Optional[Intent]) -> Optional[ClarifyOptionSpec]:
    """返回该意图的选项式追问配置；没有则 None。"""
    if intent is None:
        return None
    return CLARIFY_OPTION_SPECS.get(intent)


def needs_subtype_clarification(intent: Optional[Intent], text: str) -> Optional[ClarifyOptionSpec]:
    """若该意图需要先选子类型、且用户尚未点明，返回追问配置；否则 None。"""
    spec = get_clarify_spec(intent)
    if spec is None:
        return None
    if spec.matched_subtype(text) is not None:
        return None
    return spec
