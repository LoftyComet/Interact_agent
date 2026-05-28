from __future__ import annotations

INPUT_VERIFICATION_SYSTEM = "你是手势交互词典的术语校验器。只输出 JSON，不要输出其他内容。"

INPUT_VERIFICATION_PROMPT = """\
用户描述了一个手势交互，请检查以下问题：
1. 术语是否正确：用户使用的术语是否在术语枚举中？如果不在，最接近的正确术语是什么？
2. 组合是否合理：用户描述的控件形态+交互机制组合是否在结构化知识库中有支持？
3. 属性是否矛盾：用户描述的属性之间是否存在逻辑矛盾？

术语枚举（部分）：
{term_inventory_excerpt}

相关结构化知识：
{structured_knowledge_excerpt}

用户提取的术语：{terms}
用户原始输入：{query}

请输出 JSON：
{{"issues": [{{"original": "...", "canonical": "...", "issue_type": "wrong_term|impossible_combination|contradictory|ambiguous", "explanation": "..."}}]}}
如果没有问题，输出 {{"issues": []}}。"""

OUTPUT_VERIFICATION_SYSTEM = "你是手势交互词典的输出质量检查器。只输出 JSON，不要输出其他内容。"

OUTPUT_VERIFICATION_PROMPT = """\
请检查以下回答是否符合手势词典规范：
1. 是否使用了不在术语枚举中的"新发明"术语？
2. 是否有与结构化知识库矛盾的事实性错误？
3. 是否缺少 output_frame 要求的章节？

术语枚举（部分）：
{term_inventory_excerpt}

相关结构化知识：
{structured_knowledge_excerpt}

期望的章节标题：{output_frame}

模型输出：
{output}

请输出 JSON：
{{"issues": [{{"issue_type": "missing_section|invalid_term|knowledge_conflict|format_error", "location": "...", "description": "...", "severity": "error|warning"}}]}}
如果没有问题，输出 {{"issues": []}}。"""

OUTPUT_CORRECTION_PROMPT = """\
你的上一次回答存在以下问题，请修正后重新输出完整回答：

{correction_hints}

请直接输出修正后的完整 Markdown 回答，不要解释修改了什么。"""
