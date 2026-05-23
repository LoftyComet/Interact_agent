from __future__ import annotations

import re
from typing import Optional

from gesture_agent.core.models import Intent, IntentCandidate, IntentOutputFrames, IntentResolution, Layer, QuestionStructure
from gesture_agent.evaluation import parse_design_evaluation
from gesture_agent.knowledge.base import KnowledgeBase
from gesture_agent.learning.output_frames import load_output_frames


DESIGN_EVALUATION_RE = re.compile(r"(评估|评价|评审|设计方案|这个方案|方案合理|合理吗|有什么问题|哪里有问题|改进建议|优化建议|怎么优化|帮我看看.*设计|设计.*建议)")
COMPARE_RE = re.compile(r"(对比|比较|区别|差异|不同|vs|VS|相比|哪个更|如何选择)")
CASE_RE = re.compile(r"(案例|例子|图片|图中|截图|这个交互|这个设计|分析|拆解|应用)")
PODCAST_RE = re.compile(r"(播客|podcast|口播|脚本|讲稿|音频节目|节目稿|访谈)")
VOICE_RE = re.compile(r"(语音交互|语音|声控|口令|唤醒词|对话式|说话|语速|声纹)")
MULTIMODAL_RE = re.compile(r"(多模态|multimodal|跨模态|模态|视觉.*语音|语音.*手势|图像.*语音|触觉.*视觉)")
BACKGROUND_RE = re.compile(r"(背景|交互的本质|操控力|虚拟操控力|IxDL|声明式|AI时代|适用人群|为什么)")
PROPERTY_RE = re.compile(r"(基础属性|属性|二元|多级|位置|角度|力属性|声音属性|光属性|温度|形变|时间属性|生理信号|阶次控制)")
CONTROL_FORM_RE = re.compile(r"(控件形态|控件|按钮|拨钮|滚轮|摇杆|轨迹球|指点杆|触控面|旋钮|手柄|踏板|眼睛|嘴巴|手势)")
ADVANCED_MECHANISM_RE = re.compile(
    r"(高级|组合|拓展|多维协同|冲突|调和|限位|长按拖拽|双按拖拽|轻扫|速率式|域控|异位|向量菜单|动势|快击|缓冲|解耦|互斥|轻拨)"
)
MECHANISM_RE = re.compile(r"(交互机制|交互方式|点击|单击|双击|长按|按下|开关|拖拽|甩动|滑动|翻动|捏合|旋转)")

INTENT_LABELS: dict[Intent, str] = {
    "basic_interaction_mechanism": "基础交互机制",
    "advanced_interaction_mechanism": "高级交互机制",
    "control_form": "控件形态",
    "basic_property": "基础属性",
    "multimodal_interaction": "多模态交互",
    "voice_interaction": "语音交互",
    "podcast_content": "播客内容",
    "interaction_compare": "交互机制对比",
    "background_knowledge": "背景知识",
    "case_analysis": "理解交互案例",
    "design_evaluation": "设计方案评估",
}

# Intent classification thresholds — shared with llm_intent.py
CONFIDENCE_THRESHOLD = 0.65
CLOSE_CALL_MARGIN = 0.15
HIGH_CONFIDENCE_THRESHOLD = 0.9


class QuestionParser:
    def __init__(self, kb: KnowledgeBase, output_frames: Optional[IntentOutputFrames] = None) -> None:
        self.kb = kb
        self.output_frames = output_frames or load_output_frames(kb.data_dir)

    def parse(
        self,
        query: str,
        image_paths: Optional[list[str]] = None,
        forced_intent: Optional[Intent] = None,
    ) -> QuestionStructure:
        image_paths = image_paths or []
        terms = self.kb.find_terms(query)
        intent = forced_intent or self._detect_intent(query, image_paths, terms)
        layers = self._detect_layers(query, terms, intent)
        focus = self._detect_focus(query)
        compare_targets = self._detect_compare_targets(query, terms) if intent == "interaction_compare" else []
        case_modality = self._detect_case_modality(query, image_paths)
        missing_info = self._missing_info(intent, query, image_paths)
        output_frame = self._output_frame(intent)
        design_evaluation = (
            parse_design_evaluation(query, image_paths=image_paths, terms=terms)
            if intent == "design_evaluation"
            else None
        )

        return QuestionStructure(
            raw_query=query,
            intent=intent,
            layers=layers,
            terms=terms,
            focus=focus,
            compare_targets=compare_targets,
            case_modality=case_modality,
            missing_info=missing_info,
            output_frame=output_frame,
            design_evaluation=design_evaluation,
        )

    def resolve_intent(self, query: str, image_paths: Optional[list[str]] = None) -> IntentResolution:
        image_paths = image_paths or []
        terms = self.kb.find_terms(query)
        candidates = self._intent_candidates(query, image_paths, terms)
        if not candidates:
            return IntentResolution(
                intent=None,
                confidence=0.0,
                candidates=[],
                needs_clarification=True,
                clarification_question=self._build_clarification_question(query, []),
            )

        top = candidates[0]
        second = candidates[1] if len(candidates) > 1 else None
        missing_info = self._blocking_missing_info(top.intent, query, image_paths)
        is_low_confidence = top.score < CONFIDENCE_THRESHOLD
        is_close_call = bool(second and top.score < HIGH_CONFIDENCE_THRESHOLD and top.score - second.score < CLOSE_CALL_MARGIN)
        needs_clarification = is_low_confidence or is_close_call or bool(missing_info)

        return IntentResolution(
            intent=top.intent,
            confidence=top.score,
            candidates=candidates[:4],
            needs_clarification=needs_clarification,
            clarification_question=self._build_clarification_question(query, candidates[:4], missing_info),
            missing_info=missing_info,
        )

    def _detect_intent(self, query: str, image_paths: list[str], terms: list[str]) -> Intent:
        candidates = self._intent_candidates(query, image_paths, terms)
        return candidates[0].intent if candidates else "background_knowledge"

    def _intent_candidates(self, query: str, image_paths: list[str], terms: list[str]) -> list[IntentCandidate]:
        candidates: dict[Intent, IntentCandidate] = {}

        def add(intent: Intent, score: float, reason: str) -> None:
            current = candidates.get(intent)
            if current is None or score > current.score:
                candidates[intent] = IntentCandidate(intent=intent, score=score, reason=reason)

        if DESIGN_EVALUATION_RE.search(query):
            add("design_evaluation", 0.99, "问题包含评估/评价/优化设计方案等设计评审信号。")
        if image_paths:
            add("case_analysis", 0.94, "用户提供了图片，需要理解交互案例。")
        if CASE_RE.search(query):
            add("case_analysis", 0.92, "问题包含案例/图片/分析/拆解等案例理解信号。")
        if COMPARE_RE.search(query):
            add("interaction_compare", 0.98, "问题包含对比/区别/差异等比较信号。")
        if PODCAST_RE.search(query):
            add("podcast_content", 0.95, "问题包含播客/脚本/口播等内容生成信号。")
        if MULTIMODAL_RE.search(query):
            add("multimodal_interaction", 0.95, "问题包含多模态或跨模态分工信号。")
        if CONTROL_FORM_RE.search(query):
            add("control_form", 0.88, "问题包含控件形态或具体控件名称。")
        if PROPERTY_RE.search(query):
            add("basic_property", 0.84, "问题包含基础属性或具体属性名称。")
        if VOICE_RE.search(query):
            add("voice_interaction", 0.86, "问题包含语音交互、声控或口令等信号。")
        if BACKGROUND_RE.search(query):
            add("background_knowledge", 0.86, "问题包含背景知识、操控力、IxDL 或声明式等信号。")
        if ADVANCED_MECHANISM_RE.search(query):
            add("advanced_interaction_mechanism", 0.9, "问题包含高级机制、组合或冲突调和信号。")
        if MECHANISM_RE.search(query) or self._has_interaction_mechanism_match(query, terms, query_text=query):
            add(self._detect_mechanism_intent(query, terms), 0.86, "问题命中交互机制术语或机制章节。")

        for chunk in self.kb.search(query, top_k=5, prefer_terms=terms):
            if chunk.score < 3.0:
                continue
            if chunk.layer == "basic_property":
                add("basic_property", 0.74, f"检索命中基础属性章节：{chunk.title}")
            elif chunk.layer == "control_form":
                add("control_form", 0.74, f"检索命中控件形态章节：{chunk.title}")
            elif chunk.layer == "multimodal_interaction":
                add("multimodal_interaction", 0.74, f"检索命中多模态章节：{chunk.title}")
            elif chunk.layer == "voice_interaction":
                add("voice_interaction", 0.74, f"检索命中语音交互章节：{chunk.title}")
            elif chunk.layer == "interaction_mechanism":
                add(self._detect_mechanism_intent(query, terms), 0.74, f"检索命中交互机制章节：{chunk.title}")

        return sorted(candidates.values(), key=lambda item: item.score, reverse=True)

    def _detect_layers(self, query: str, terms: list[str], intent: Intent) -> list[Layer]:
        layers: list[Layer] = []
        for chunk in self.kb.search(query, top_k=8, prefer_terms=terms):
            if chunk.layer != "unknown" and chunk.layer not in layers:
                layers.append(chunk.layer)

        intent_layer_map: dict[Intent, Layer] = {
            "basic_property": "basic_property",
            "basic_interaction_mechanism": "interaction_mechanism",
            "advanced_interaction_mechanism": "interaction_mechanism",
            "control_form": "control_form",
            "case_analysis": "interaction_case",
            "design_evaluation": "design_evaluation",
            "multimodal_interaction": "multimodal_interaction",
            "voice_interaction": "voice_interaction",
            "podcast_content": "podcast_content",
            "background_knowledge": "background_knowledge",
            "interaction_compare": "interaction_mechanism",
        }
        mapped_layer = intent_layer_map[intent]
        if mapped_layer not in layers:
            layers.append(mapped_layer)

        if "属性" in query and "basic_property" not in layers:
            layers.append("basic_property")
        if any(word in query for word in ["交互机制", "交互方式", "手势", "点击", "拖拽", "滑动"]):
            if "interaction_mechanism" not in layers:
                layers.append("interaction_mechanism")
        if any(word in query for word in ["控件", "形态", "按钮", "旋钮", "摇杆", "触控"]):
            if "control_form" not in layers:
                layers.append("control_form")
        return layers or ["unknown"]

    def _detect_focus(self, query: str) -> list[str]:
        mapping = [
            ("定义", ["是什么", "定义", "概念", "含义"]),
            ("属性", ["属性", "用了哪几个属性", "输入维度"]),
            ("逻辑关系", ["逻辑", "关系", "怎么工作", "结构", "映射"]),
            ("交互特性", ["特性", "响应速度", "操作难度", "容错", "安全"]),
            ("适用边界", ["适用", "不适用", "场景", "边界", "什么时候"]),
            ("案例理解", ["案例", "例子", "图片", "图中", "截图"]),
            ("设计评估", ["评估", "评价", "评审", "设计方案", "合理吗", "有什么问题", "优化", "改进建议"]),
            ("学习引导", ["引导", "追问", "测验", "学习"]),
        ]
        focus = [name for name, keys in mapping if any(key in query for key in keys)]
        return focus or ["定义", "逻辑关系", "适用边界"]

    def _detect_compare_targets(self, query: str, terms: list[str]) -> list[str]:
        if len(terms) >= 2:
            return terms[:4]
        candidates: list[str] = []
        for sep in ["和", "与", "vs", "VS", "、", ",", "，"]:
            if sep in query:
                parts = [part.strip(" ？?。") for part in query.split(sep)]
                candidates.extend(part for part in parts if 1 < len(part) <= 16)
        for term in terms:
            if term not in candidates:
                candidates.append(term)
        return candidates[:4]

    def _detect_case_modality(self, query: str, image_paths: list[str]) -> list[str]:
        modality: list[str] = []
        if image_paths or any(word in query for word in ["图片", "图中", "截图"]):
            modality.append("image")
        if query.strip():
            modality.append("text")
        return modality or ["text"]

    def _missing_info(self, intent: Intent, query: str, image_paths: list[str]) -> list[str]:
        if intent == "design_evaluation":
            terms = self.kb.find_terms(query)
            return parse_design_evaluation(query, image_paths=image_paths, terms=terms).missing_info
        return self._blocking_missing_info(intent, query, image_paths)

    def _blocking_missing_info(self, intent: Intent, query: str, image_paths: list[str]) -> list[str]:
        missing: list[str] = []
        if intent == "case_analysis" and not image_paths and len(query) < 30:
            missing.append("案例描述较短，可能需要补充界面/控件/用户动作/系统反馈。")
        if intent == "interaction_compare" and len(self.kb.find_terms(query)) < 2:
            missing.append("对比对象不够明确，建议至少给出两个交互方式或控件形态。")
        if intent == "podcast_content" and len(query) < 20:
            missing.append("播客主题较短，建议补充听众对象、时长、栏目风格或希望讲解的知识点。")
        if intent == "design_evaluation":
            terms = self.kb.find_terms(query)
            evaluation = parse_design_evaluation(query, image_paths=image_paths, terms=terms)
            critical_missing = [
                item
                for item in evaluation.missing_info
                if item.startswith("设计方案描述较短")
                or item.startswith("缺少用户目标")
                or item.startswith("缺少明确的交互机制")
            ]
            if not image_paths and not evaluation.control_forms and not evaluation.mechanisms:
                critical_missing.append("缺少可拆解的控件形态和交互机制。")
            missing.extend(critical_missing)
        return missing

    def _build_clarification_question(
        self,
        query: str,
        candidates: list[IntentCandidate],
        missing_info: Optional[list[str]] = None,
    ) -> str:
        missing_info = missing_info or []
        if missing_info:
            if candidates and candidates[0].intent == "interaction_compare":
                return "你想对比哪两个或哪几个对象？请补充具体交互机制、控件形态或属性名称。"
            if candidates and candidates[0].intent == "case_analysis":
                return "这个案例信息还不够。请补充：界面/产品是什么、用户做了什么动作、系统有什么反馈；或者直接提供图片。"
            if candidates and candidates[0].intent == "podcast_content":
                return "播客方向可以确定，但还缺少制作参数。请补充听众对象、预计时长、风格，以及想重点讲哪个知识点。"
            if candidates and candidates[0].intent == "design_evaluation":
                return "要评估设计方案，还需要补充：产品/界面场景、用户目标、用户动作、控件形态、系统反馈或状态变化。"

        if not candidates:
            return (
                "这个问题目前无法确定要按哪类任务处理。你是想问：基础交互机制、控件形态、基础属性、"
                "交互机制对比、背景知识、案例分析、设计方案评估，还是播客/语音/多模态内容？请补充一个具体对象或场景。"
            )

        options = "、".join(f"{INTENT_LABELS[item.intent]}({item.reason})" for item in candidates[:3])
        return f"这个问题可能有多种理解：{options}。你希望我按哪一种来回答？也可以补充具体对象、场景或输出形式。"

    def _output_frame(self, intent: Intent) -> list[str]:
        return self.output_frames.frame_for(intent)

    def _detect_mechanism_intent(self, query: str, terms: list[str]) -> Intent:
        if ADVANCED_MECHANISM_RE.search(query):
            return "advanced_interaction_mechanism"

        for chunk in self.kb.search(query, top_k=5, prefer_terms=terms):
            if chunk.layer != "interaction_mechanism":
                continue
            if re.match(r"^[34]-[a-z]", chunk.title, flags=re.IGNORECASE):
                return "advanced_interaction_mechanism"
            if re.match(r"^[12]-[a-z]", chunk.title, flags=re.IGNORECASE):
                return "basic_interaction_mechanism"
        return "basic_interaction_mechanism"

    def _has_interaction_mechanism_match(self, query: str, terms: list[str], *, query_text: Optional[str] = None) -> bool:
        if not terms:
            return False
        mechanism_terms = set(self.kb.term_inventory.by_layer.get("interaction_mechanism", []))
        mechanism_terms.update(self.kb.term_inventory.by_type.get("基础交互机制", []))
        mechanism_terms.update(self.kb.term_inventory.by_type.get("高级交互机制", []))
        if not any(term in mechanism_terms for term in terms):
            return False
        source_query = query_text or query
        return any(
            chunk.layer == "interaction_mechanism" and chunk.score >= 3.0
            for chunk in self.kb.search(source_query, top_k=3, prefer_terms=[term for term in terms if term in mechanism_terms])
        )
