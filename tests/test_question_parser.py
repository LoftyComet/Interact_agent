from gesture_agent.knowledge import KnowledgeBase
from gesture_agent.learning import QuestionParser


def test_compare_question_detects_targets() -> None:
    kb = KnowledgeBase.load("data")
    parser = QuestionParser(kb)

    structure = parser.parse("单击和长按有什么区别？")

    assert structure.intent == "interaction_compare"
    assert "单击" in structure.terms
    assert "长按" in structure.terms
    assert "interaction_mechanism" in structure.layers


def test_basic_interaction_mechanism_intent() -> None:
    kb = KnowledgeBase.load("data")
    parser = QuestionParser(kb)

    structure = parser.parse("什么是单击？")

    assert structure.intent == "basic_interaction_mechanism"
    assert structure.output_frame == ["核心定义", "基础属性", "响应逻辑", "收益与代价", "典型案例", "适用与不适用", "关联机制"]


def test_advanced_mechanism_folds_into_basic_intent() -> None:
    kb = KnowledgeBase.load("data")
    parser = QuestionParser(kb)

    structure = parser.parse("快击如何解决单击和长按的冲突？")

    assert structure.intent == "basic_interaction_mechanism"
    assert "核心定义" in structure.output_frame


def test_control_form_intent() -> None:
    kb = KnowledgeBase.load("data")
    parser = QuestionParser(kb)

    structure = parser.parse("旋钮这种控件形态可以承载哪些属性？")

    assert structure.intent == "control_form"
    assert "可用属性" in structure.output_frame


def test_clear_control_form_property_question_does_not_require_clarification() -> None:
    parser = QuestionParser(KnowledgeBase.load("data"))

    resolution = parser.resolve_intent("旋钮可以承载哪些属性？")

    assert resolution.intent == "control_form"
    assert resolution.needs_clarification is False


def test_basic_property_intent() -> None:
    kb = KnowledgeBase.load("data")
    parser = QuestionParser(kb)

    structure = parser.parse("二元属性的关键性质是什么？")

    assert structure.intent == "basic_property"
    assert "基本性质" in structure.output_frame


def test_multimodal_intent() -> None:
    kb = KnowledgeBase.load("data")
    parser = QuestionParser(kb)

    structure = parser.parse("多模态交互中语音和手势如何分工？")

    assert structure.intent == "multimodal_interaction"
    assert "模态组成" in structure.output_frame


def test_voice_interaction_intent() -> None:
    kb = KnowledgeBase.load("data")
    parser = QuestionParser(kb)

    structure = parser.parse("语音交互需要怎样设计反馈闭环？")

    assert structure.intent == "voice_interaction"
    assert "识别/触发逻辑" in structure.output_frame


def test_podcast_content_intent() -> None:
    kb = KnowledgeBase.load("data")
    parser = QuestionParser(kb)

    structure = parser.parse("帮我写一期关于长按的播客脚本")

    # podcast intent 已移除，机制类问题统一归入 basic_interaction_mechanism；
    # 「播客」不再触发独立 intent。
    assert structure.intent in {"basic_interaction_mechanism", "open_ended", "background_knowledge"}


def test_background_knowledge_intent() -> None:
    kb = KnowledgeBase.load("data")
    parser = QuestionParser(kb)

    structure = parser.parse("交互的本质和操控力视角是什么？")

    assert structure.intent == "background_knowledge"
    assert "背景回答" in structure.output_frame


def test_case_question_detects_case_intent() -> None:
    kb = KnowledgeBase.load("data")
    parser = QuestionParser(kb)

    structure = parser.parse("分析一下手机图标长按进入编辑模式这个案例")

    assert structure.intent == "case_analysis"
    assert "text" in structure.case_modality
    assert "案例理解" in structure.focus
    assert "控件形态" in structure.output_frame


def test_design_evaluation_intent_extracts_structure() -> None:
    kb = KnowledgeBase.load("data")
    parser = QuestionParser(kb)

    structure = parser.parse(
        "请评估这个设计方案：在音乐播放器界面，用户长按音量旋钮后拖动来调节音量，松手后系统高亮确认。"
    )

    assert structure.intent == "design_evaluation"
    assert "设计评估" in structure.focus
    assert structure.design_evaluation is not None
    assert "旋钮" in structure.design_evaluation.control_forms
    assert "长按" in structure.design_evaluation.mechanisms
    assert "拖动" in structure.design_evaluation.mechanisms
    assert structure.design_evaluation.user_goal == "调节音量"
    assert "高亮" in structure.design_evaluation.system_feedback
    assert "问题诊断" in structure.output_frame


def test_design_evaluation_with_image_prefers_evaluation_over_case_analysis() -> None:
    kb = KnowledgeBase.load("data")
    parser = QuestionParser(kb)

    structure = parser.parse("请评估这张界面图里的交互方案是否合理", image_paths=["case.png"])

    assert structure.intent == "design_evaluation"
    assert structure.design_evaluation is not None
    assert "image" in structure.design_evaluation.modality


def test_design_evaluation_requires_clarification_when_too_vague() -> None:
    kb = KnowledgeBase.load("data")
    parser = QuestionParser(kb)

    resolution = parser.resolve_intent("帮我评估这个方案")

    assert resolution.intent == "design_evaluation"
    assert resolution.needs_clarification is True
    assert "评估设计方案" in resolution.clarification_question


def test_resolve_intent_requires_clarification_for_vague_query() -> None:
    kb = KnowledgeBase.load("data")
    parser = QuestionParser(kb)

    resolution = parser.resolve_intent("这个怎么用？")

    assert resolution.needs_clarification is True
    assert resolution.intent is None or resolution.confidence < 0.65


def test_resolve_intent_is_ready_for_clear_query() -> None:
    kb = KnowledgeBase.load("data")
    parser = QuestionParser(kb)

    resolution = parser.resolve_intent("什么是单击？")

    assert resolution.needs_clarification is False
    assert resolution.intent == "basic_interaction_mechanism"


def test_design_evaluation_wins_over_dictionary_phrasing() -> None:
    kb = KnowledgeBase.load("data")
    parser = QuestionParser(kb)

    # 含“评估”的方案评审，即便提到“这本词典”也应判为 design_evaluation
    structure = parser.parse("帮我评估这本词典里旋钮+长按的方案")

    assert structure.intent == "design_evaluation"


def test_mechanism_identification_intent() -> None:
    kb = KnowledgeBase.load("data")
    parser = QuestionParser(kb)

    structure = parser.parse("按住按钮释放火柱、移动方向杆改变方向、松开结束，这属于什么交互机制？")

    assert structure.intent == "mechanism_identification"
    assert "候选交互机制" in structure.output_frame
    # 关键约束：模版不引导直接给出表达式
    assert "表达式" not in "".join(structure.output_frame)


def test_function_interaction_breakdown_intent() -> None:
    kb = KnowledgeBase.load("data")
    parser = QuestionParser(kb)

    structure = parser.parse("英雄联盟中涉及英雄控制的交互都涉及了哪些手势？")

    assert structure.intent == "function_interaction_breakdown"
    assert "情况与对应交互" in structure.output_frame


def test_mechanism_parameter_compare_intent() -> None:
    kb = KnowledgeBase.load("data")
    parser = QuestionParser(kb)

    structure = parser.parse("拖拽短距离和拖拽长距离有什么不同的应用场景")

    assert structure.intent == "mechanism_parameter_compare"
    assert "资料依据" in structure.output_frame


def test_plain_mechanism_compare_stays_interaction_compare() -> None:
    kb = KnowledgeBase.load("data")
    parser = QuestionParser(kb)

    # 纯机制对比（无控件载体、无参数维度）仍应判 interaction_compare
    structure = parser.parse("拖拽和滑动有什么区别")

    assert structure.intent == "interaction_compare"


def test_control_form_compare_has_subtype_frame() -> None:
    parser = QuestionParser(KnowledgeBase.load("data"))

    structure = parser.parse("在车上设计交互，是装旋钮好还是用触控面好？")

    assert structure.intent == "interaction_compare"
    assert structure.subtype == "control_form_compare"
    assert structure.output_frame == [
        "对比对象", "形态描述", "包含属性", "常用交互机制", "适用场景", "选择建议"
    ]


def test_control_form_application_has_subtype_frame() -> None:
    parser = QuestionParser(KnowledgeBase.load("data"))

    structure = parser.parse("做适老化界面时应该用按钮还是手势？")

    assert structure.intent == "design_suggestion"
    assert structure.subtype == "control_form_application"
    assert structure.output_frame == [
        "资料边界说明", "需要澄清的信息", "可参考的属性与机制", "初步建议"
    ]


def test_design_suggestion_intent() -> None:
    kb = KnowledgeBase.load("data")
    parser = QuestionParser(kb)

    structure = parser.parse("我想设计一个VR中的键盘交互，有什么建议")

    assert structure.intent == "design_suggestion"
    assert "设计建议（仅供参考）" in structure.output_frame


def test_retrieval_instruction_intent() -> None:
    kb = KnowledgeBase.load("data")
    parser = QuestionParser(kb)

    structure = parser.parse("给我向量菜单的IxDL表达式")

    assert structure.intent == "retrieval_instruction"
    assert "检索结果" in structure.output_frame


def test_interaction_optimization_intent() -> None:
    kb = KnowledgeBase.load("data")
    parser = QuestionParser(kb)

    structure = parser.parse("怎么优化交互提升原神切换角色释放技能呢？现在容易误触其他角色")

    assert structure.intent == "interaction_optimization"
    assert "问题诊断" in structure.output_frame


def test_evaluation_methodology_intent() -> None:
    kb = KnowledgeBase.load("data")
    parser = QuestionParser(kb)

    structure = parser.parse("在设计中我该如何评估我的交互，评估的维度都有哪些")

    assert structure.intent == "evaluation_methodology"
    assert "评估单元" in structure.output_frame


def test_concrete_design_evaluation_not_optimization() -> None:
    kb = KnowledgeBase.load("data")
    parser = QuestionParser(kb)

    # 评审一个完整方案（不含"优化/误触"信号）仍判 design_evaluation
    structure = parser.parse("帮我评估这本词典里旋钮+长按的方案")

    assert structure.intent == "design_evaluation"


def test_real_world_intent_routing_regressions() -> None:
    parser = QuestionParser(KnowledgeBase.load("data"))
    cases = {
        "书里那个缓冲式越界切换看了三遍才勉强懂，有没有更直观的讲法？最好用个生活中的例子。": "basic_interaction_mechanism",
        "眼动追踪+手势结合时有什么要注意的吗？比如某个产品案例。": "multimodal_interaction",
        "单击和按下什么时候该用单击，什么时候该用按下？": "interaction_compare",
        "iPhone滑动解锁属于你们书里的哪个机制？": "mechanism_identification",
        "拆一个播放器看看：播放是点击、进度条是拖拽，还有别的吗？": "function_interaction_breakdown",
        "长按时间设多少合适？200ms和500ms各有什么取舍？": "mechanism_parameter_compare",
        "侧滑返回和侧边栏菜单冲突，这个怎么解决？": "interaction_optimization",
        "适老化设计时，用按钮还是手势好？": "design_suggestion",
        "一个交互设计好不好，有没有客观的评价标准？": "evaluation_methodology",
    }

    for query, expected in cases.items():
        assert parser.parse(query).intent == expected, query
