from __future__ import annotations

import re
from typing import Optional

from gesture_agent.core.models import Intent, IntentCandidate, IntentOutputFrames, IntentResolution, Layer, QuestionStructure
from gesture_agent.evaluation import parse_design_evaluation
from gesture_agent.knowledge.base import KnowledgeBase
from gesture_agent.learning.intent_examples import (
    EXACT_MATCH_SCORE,
    EXACT_MATCH_THRESHOLD,
    NEAR_MATCH_SCORE,
    NEAR_MATCH_THRESHOLD,
    IntentExampleBank,
    IntentExampleMatch,
    load_intent_examples,
)
from gesture_agent.learning.output_frames import load_output_frames


DESIGN_EVALUATION_RE = re.compile(r"(评估|评价|评审|设计方案|这个方案|方案合理|合理吗|有什么问题|哪里有问题|改进建议|优化建议|怎么优化|帮我看看.*设计|设计.*建议)")
COMPARE_RE = re.compile(r"(对比|比较|区别|差异|不同|vs|VS|相比|哪个更|如何选择)")
CASE_RE = re.compile(r"(案例|例子|图片|图中|截图|这个交互|这个设计|分析|拆解|应用)")
VOICE_RE = re.compile(r"(语音交互|语音|声控|口令|唤醒词|对话式|说话|语速|声纹)")
MULTIMODAL_RE = re.compile(r"(多模态|multimodal|跨模态|模态|视觉.*语音|语音.*手势|图像.*语音|触觉.*视觉)")
BACKGROUND_RE = re.compile(r"(背景|交互的本质|操控力|虚拟操控力|IxDL|声明式|AI时代|适用人群|为什么)")
PROPERTY_RE = re.compile(r"(基础属性|属性|二元|多级|位置|角度|力属性|声音属性|光属性|温度|形变|时间属性|生理信号|阶次控制)")
CONTROL_FORM_RE = re.compile(r"(控件形态|控件|按钮|拨钮|滚轮|摇杆|轨迹球|指点杆|触控面|旋钮|手柄|踏板|眼睛|嘴巴|手势)")
MECHANISM_RE = re.compile(r"(交互机制|交互方式|点击|单击|双击|长按|按下|开关|拖拽|甩动|滑动|翻动|捏合|旋转|高级|组合|拓展|多维协同|冲突|调和|限位|长按拖拽|双按拖拽|轻扫|速率式|域控|异位|向量菜单|动势|快击|缓冲|解耦|互斥|轻拨)")
# 反推信号：给一段操作描述，问"属于什么机制/这是什么交互/能不能生成表达式"。
MECHANISM_IDENTIFY_RE = re.compile(r"(属于什么交互机制|属于什么机制|是什么交互机制|这是什么交互|算什么交互|属于哪一?类|属于哪种|是哪种机制|对应.*交互机制|对应.*机制|生成.*交互表达式|生成.*表达式|表达式图)")
# 枚举信号：拆解"一类功能"在多种情况下的交互，而非单个案例。
BREAKDOWN_RE = re.compile(r"(涉及哪些|有哪些手势|有哪些交互|各种情况|每种情况|哪些情况|不同情况|涉及.*哪些|都涉及了哪些|拆解.*各种|各种.*交互逻辑|交互逻辑系统)")
# 同机制不同参数对比：长/短距离、不同力度/速度/时长等参数维度。
PARAM_RE = re.compile(r"(长距离|短距离|远距离|近距离|不同参数|参数不同|不同力度|不同速度|不同距离|不同时长|大幅.*小幅)")
# 控件形态延伸应用：控件与位置/人群/场景的关系（书中通常没有直接内容）。
CONTROL_APPLICATION_RE = re.compile(r"(适合放置在什么位置|放置在什么位置|放在什么位置|放置在哪|安装在哪|适合老年人|适合儿童|适合.*人群|更适合.*(用户|人)|什么样的控件.*适合|什么控件.*适合|哪种控件.*适合|作为交互体|交互体.*应用场景|控件.*应用场景|控件.*场景)")
# 评估方法论：如何评估交互、好坏标准、评估维度（不是评估某个具体方案）。
EVAL_METHODOLOGY_RE = re.compile(r"(如何评估|怎么评估|怎样评估|如何去评估|该如何评估|评估维度|评估的维度|评估方法|评估标准|什么是好的交互|怎么衡量|如何衡量)")
# 交互优化：现有交互存在具体问题、想优化提升（区别于评估完整方案）。
OPTIMIZATION_RE = re.compile(r"(优化|提升.*体验|提升.*交互|改善.*交互|改善.*体验|误触|容易误|不顺手|卡顿)")

INTENT_LABELS: dict[Intent, str] = {
    "basic_interaction_mechanism": "交互机制",
    "control_form": "控件形态",
    "basic_property": "基础属性",
    "multimodal_interaction": "含义与多模态",
    "voice_interaction": "语音交互",
    "interaction_compare": "交互机制对比",
    "background_knowledge": "背景知识",
    "case_analysis": "理解交互案例",
    "design_evaluation": "设计方案评估",
    "open_ended": "开放问题",
    "mechanism_identification": "机制识别",
    "control_form_compare": "控件形态对比",
    "function_interaction_breakdown": "功能交互拆解",
    "mechanism_parameter_compare": "同机制参数对比",
    "control_form_application": "控件形态延伸应用",
    "interaction_optimization": "交互优化",
    "evaluation_methodology": "评估方法论",
}

# Intent classification thresholds — shared with llm_intent.py
CONFIDENCE_THRESHOLD = 0.65
CLOSE_CALL_MARGIN = 0.15
HIGH_CONFIDENCE_THRESHOLD = 0.9


class QuestionParser:
    def __init__(
        self,
        kb: KnowledgeBase,
        output_frames: Optional[IntentOutputFrames] = None,
        example_bank: Optional[IntentExampleBank] = None,
    ) -> None:
        self.kb = kb
        self.output_frames = output_frames or load_output_frames(kb.data_dir)
        self.example_bank = example_bank or load_intent_examples(kb.data_dir)

    def parse(
        self,
        query: str,
        image_paths: Optional[list[str]] = None,
        forced_intent: Optional[Intent] = None,
        output_frame_override: Optional[list[str]] = None,
    ) -> QuestionStructure:
        image_paths = image_paths or []
        terms = self.kb.find_terms(query)
        intent = forced_intent or self._detect_intent(query, image_paths, terms)
        layers = self._detect_layers(query, terms, intent)
        focus = self._detect_focus(query)
        compare_targets = self._detect_compare_targets(query, terms) if intent == "interaction_compare" else []
        case_modality = self._detect_case_modality(query, image_paths)
        missing_info = self._missing_info(intent, query, image_paths)
        output_frame = output_frame_override or self._output_frame(intent)
        if output_frame_override:
            output_frame_source = "cot"
        elif intent == "open_ended":
            output_frame_source = "cot"
        else:
            output_frame_source = "static"
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
            output_frame_source=output_frame_source,
            design_evaluation=design_evaluation,
        )

    def resolve_intent(
        self,
        query: str,
        image_paths: Optional[list[str]] = None,
        clarification_history: Optional[list[str]] = None,
    ) -> IntentResolution:
        image_paths = image_paths or []
        terms = self.kb.find_terms(query)
        candidates = self._intent_candidates(query, image_paths, terms)
        if not candidates:
            return IntentResolution(
                intent=None,
                confidence=0.0,
                candidates=[],
                needs_clarification=True,
                clarification_question=self._build_clarification_question(
                    query, [], clarification_history=clarification_history
                ),
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
            clarification_question=self._build_clarification_question(
                query, candidates[:4], missing_info, clarification_history=clarification_history
            ),
            missing_info=missing_info,
        )

    def relevant_examples(self, query: str) -> list[IntentExampleMatch]:
        """返回与 query 相关的已标注样例，供大模型意图判定参考。"""
        if self.example_bank.is_empty:
            return []
        return self.example_bank.relevant(query)

    def _detect_intent(self, query: str, image_paths: list[str], terms: list[str]) -> Intent:
        candidates = self._intent_candidates(query, image_paths, terms)
        return candidates[0].intent if candidates else "background_knowledge"

    def _intent_candidates(self, query: str, image_paths: list[str], terms: list[str]) -> list[IntentCandidate]:
        candidates: dict[Intent, IntentCandidate] = {}

        def add(intent: Intent, score: float, reason: str) -> None:
            current = candidates.get(intent)
            if current is None or score > current.score:
                candidates[intent] = IntentCandidate(intent=intent, score=score, reason=reason)

        # 评估方法论：问"如何评估/评估维度/好坏标准"，是方法论而非评估某个具体方案。
        # 须放在 design_evaluation 之前并给更高分（"评估"也会命中 DESIGN_EVALUATION_RE）。
        if EVAL_METHODOLOGY_RE.search(query):
            add("evaluation_methodology", 0.995, "问题问的是如何评估交互/评估维度（方法论），而非评估某个具体方案。")
        # 交互优化：现有交互有具体问题、想优化提升（如误触），先诊断再追问。
        # 须放在 design_evaluation 之前（"优化"也会命中 DESIGN_EVALUATION_RE）。
        if OPTIMIZATION_RE.search(query):
            add("interaction_optimization", 0.995, "问题在优化现有交互的具体问题（先诊断、必要时追问）。")
        if DESIGN_EVALUATION_RE.search(query):
            add("design_evaluation", 0.99, "问题包含评估/评价/优化设计方案等设计评审信号。")
        if image_paths:
            add("case_analysis", 0.94, "用户提供了图片，需要理解交互案例。")
        if CASE_RE.search(query):
            add("case_analysis", 0.92, "问题包含案例/图片/分析/拆解等案例理解信号。")
        # 机制识别：反推操作描述对应哪些机制（与 basic_interaction_mechanism 方向相反）。
        # 0.93 压过 case_analysis(0.92, 如"这个交互…生成表达式")与 basic_interaction_mechanism(0.86)。
        if MECHANISM_IDENTIFY_RE.search(query):
            add("mechanism_identification", 0.93, "问题在反推操作描述对应哪些交互机制（先给候选、暂不附表达式）。")
        # 功能交互拆解：拆解一类功能在多种情况下的交互枚举（与单案例 case_analysis 区分）。
        if BREAKDOWN_RE.search(query):
            add("function_interaction_breakdown", 0.93, "拆解一类功能在多种情况下的交互枚举，而非单个案例。")
        # 控件形态对比：硬件/控件载体之间的对比，须放在 interaction_compare 之前并给 0.99。
        # 命中条件更严格（同时命中控件且不命中机制），不会误伤纯机制对比。
        if COMPARE_RE.search(query) and CONTROL_FORM_RE.search(query) and not MECHANISM_RE.search(query):
            add("control_form_compare", 0.99, "对比对象是控件/硬件载体而非交互机制。")
        # 同机制参数对比：同一机制不同参数（如拖拽长/短距离）的取舍，也放在 interaction_compare 之前。
        if COMPARE_RE.search(query) and PARAM_RE.search(query) and MECHANISM_RE.search(query):
            add("mechanism_parameter_compare", 0.99, "对比同一交互机制在不同参数下的取舍（书中可能无系统研究）。")
        if COMPARE_RE.search(query):
            add("interaction_compare", 0.98, "问题包含对比/区别/差异等比较信号。")
        if MULTIMODAL_RE.search(query):
            add("multimodal_interaction", 0.95, "问题包含多模态或跨模态分工信号。")
        # 控件形态延伸应用：控件与位置/人群/场景的关系（书中通常没有直接内容），
        # 给 0.9 压过普通 control_form(0.88)。
        if CONTROL_APPLICATION_RE.search(query):
            add("control_form_application", 0.9, "问题问的是控件与位置/人群/场景的关系（书中通常没有直接内容）。")
        if CONTROL_FORM_RE.search(query):
            add("control_form", 0.88, "问题包含控件形态或具体控件名称。")
        if PROPERTY_RE.search(query):
            add("basic_property", 0.84, "问题包含基础属性或具体属性名称。")
        if VOICE_RE.search(query):
            add("voice_interaction", 0.86, "问题包含语音交互、声控或口令等信号。")
        if BACKGROUND_RE.search(query):
            add("background_knowledge", 0.86, "问题包含背景知识、操控力、IxDL 或声明式等信号。")
        if MECHANISM_RE.search(query) or self._has_interaction_mechanism_match(query, terms, query_text=query):
            add("basic_interaction_mechanism", 0.86, "问题命中交互机制术语或机制章节。")

        if not self.example_bank.is_empty:
            best = self.example_bank.best_match(query)
            if best and best.similarity >= NEAR_MATCH_THRESHOLD:
                score = EXACT_MATCH_SCORE if best.similarity >= EXACT_MATCH_THRESHOLD else NEAR_MATCH_SCORE
                add(
                    best.example.intent,
                    score,
                    f"与已标注样例高度相似（{best.similarity:.0%}）：{best.example.question}",
                )

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
                add("basic_interaction_mechanism", 0.74, f"检索命中交互机制章节：{chunk.title}")

        return sorted(candidates.values(), key=lambda item: item.score, reverse=True)

    def _detect_layers(self, query: str, terms: list[str], intent: Intent) -> list[Layer]:
        layers: list[Layer] = []
        for chunk in self.kb.search(query, top_k=8, prefer_terms=terms):
            if chunk.layer != "unknown" and chunk.layer not in layers:
                layers.append(chunk.layer)

        intent_layer_map: dict[Intent, Layer] = {
            "basic_property": "basic_property",
            "basic_interaction_mechanism": "interaction_mechanism",
            "control_form": "control_form",
            "case_analysis": "interaction_case",
            "design_evaluation": "design_evaluation",
            "multimodal_interaction": "multimodal_interaction",
            "voice_interaction": "voice_interaction",
            "background_knowledge": "background_knowledge",
            "interaction_compare": "interaction_mechanism",
            "open_ended": "unknown",
            "mechanism_identification": "interaction_mechanism",
            "control_form_compare": "control_form",
            "function_interaction_breakdown": "interaction_case",
            "mechanism_parameter_compare": "interaction_mechanism",
            "control_form_application": "control_form",
            "interaction_optimization": "design_evaluation",
            "evaluation_methodology": "design_evaluation",
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
        if intent == "open_ended":
            return []
        if intent == "design_evaluation":
            terms = self.kb.find_terms(query)
            return parse_design_evaluation(query, image_paths=image_paths, terms=terms).missing_info
        return self._blocking_missing_info(intent, query, image_paths)

    def _blocking_missing_info(self, intent: Intent, query: str, image_paths: list[str]) -> list[str]:
        if intent == "open_ended":
            return []
        missing: list[str] = []
        if intent == "case_analysis" and not image_paths and len(query) < 30:
            missing.append("案例描述较短，可能需要补充界面/控件/用户动作/系统反馈。")
        if intent == "interaction_compare" and len(self.kb.find_terms(query)) < 2:
            missing.append("对比对象不够明确，建议至少给出两个交互方式或控件形态。")
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
        clarification_history: Optional[list[str]] = None,
    ) -> str:
        missing_info = missing_info or []
        repeat_note = ""
        if clarification_history and len(clarification_history) >= 2:
            repeat_note = "（我换个方式问：）"

        if missing_info:
            if candidates and candidates[0].intent == "interaction_compare":
                return f"{repeat_note}你想对比哪两个或哪几个对象？请补充具体交互机制、控件形态或属性名称。"
            if candidates and candidates[0].intent == "case_analysis":
                return f"{repeat_note}这个案例信息还不够。请补充：界面/产品是什么、用户做了什么动作、系统有什么反馈；或者直接提供图片。"
            if candidates and candidates[0].intent == "design_evaluation":
                return f"{repeat_note}要评估设计方案，还需要补充：产品/界面场景、用户目标、用户动作、控件形态、系统反馈或状态变化。"

        if not candidates:
            return (
                f"{repeat_note}这个问题目前无法确定要按哪类任务处理。你是想问：交互机制、控件形态、基础属性、"
                "交互机制对比、背景知识、案例分析、设计方案评估，还是语音/多模态内容？请补充一个具体对象或场景。"
            )

        options = "、".join(f"{INTENT_LABELS[item.intent]}({item.reason})" for item in candidates[:3])
        return f"{repeat_note}这个问题可能有多种理解：{options}。你希望我按哪一种来回答？也可以补充具体对象、场景或输出形式。"

    def _is_ambiguous_call(self, candidates: list[IntentCandidate]) -> bool:
        """Return True when the top candidates are close enough to present as options."""
        if len(candidates) < 2:
            return False
        top = candidates[0]
        second = candidates[1]
        return (
            top.score < HIGH_CONFIDENCE_THRESHOLD
            and top.score - second.score < CLOSE_CALL_MARGIN
            and second.score >= CONFIDENCE_THRESHOLD - 0.15
        )

    def _build_intent_candidate_options(
        self,
        query: str,
        candidates: list[IntentCandidate],
    ) -> list[dict[str, str]]:
        """Build clickable options from top intent candidates for ambiguous queries."""
        options: list[dict[str, str]] = []
        for item in candidates[:4]:
            label = INTENT_LABELS.get(item.intent, item.intent)
            value = f"【intent:{item.intent}】{query}"
            desc = item.reason
            options.append({"label": label, "value": value, "desc": desc})
        return options

    def _output_frame(self, intent: Intent) -> list[str]:
        if intent == "open_ended":
            return []
        return self.output_frames.frame_for(intent)

    def _has_interaction_mechanism_match(self, query: str, terms: list[str], *, query_text: Optional[str] = None) -> bool:
        if not terms:
            return False
        mechanism_terms = set(self.kb.term_inventory.by_layer.get("interaction_mechanism", []))
        mechanism_terms.update(self.kb.term_inventory.by_type.get("基础交互逻辑", []))
        mechanism_terms.update(self.kb.term_inventory.by_type.get("交互逻辑的组合与扩展", []))
        if not any(term in mechanism_terms for term in terms):
            return False
        source_query = query_text or query
        return any(
            chunk.layer == "interaction_mechanism" and chunk.score >= 3.0
            for chunk in self.kb.search(source_query, top_k=3, prefer_terms=[term for term in terms if term in mechanism_terms])
        )
