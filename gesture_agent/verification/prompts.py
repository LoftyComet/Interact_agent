from __future__ import annotations

INPUT_VERIFICATION_SYSTEM = "你是手势交互词典的术语对齐器。只输出 JSON，不要输出其他内容。"

INPUT_VERIFICATION_PROMPT = """\
用户用日常口语描述手势交互，可能用了不在词典枚举里的非标准词（口语说法、近义词、错别字）。
你的任务：找出这些非标准词，并把每个映射到**最接近的枚举术语**。

判定规则：
1. 只处理"指代某个枚举概念但用词不标准"的情况，例如把「转盘/拨盘」映射到「旋钮」、把「握把」映射到某个手柄类控件。
2. 如果某个词本来就是枚举里的标准词，不要输出。
3. 如果某个词在枚举里**没有**足够接近、足够确定的对应项，就**不要**输出它（宁可放过，也不要硬凑）。
4. canonical 必须严格来自下面的枚举清单，逐字一致，不要自造或改写。
5. 子集不能看作新的命令

术语枚举：
{term_inventory_excerpt}

相关结构化知识：
{structured_knowledge_excerpt}

已识别到的标准术语（这些无需处理）：{terms}
用户原始输入：{query}

请输出 JSON：
{{"issues": [{{"original": "用户原词", "canonical": "枚举标准词", "issue_type": "wrong_term", "explanation": "一句话说明为何这样映射"}}]}}
没有需要纠正的词时，输出 {{"issues": []}}。"""

OUTPUT_VERIFICATION_SYSTEM = "你是手势交互词典的输出质量检查器。只输出 JSON，不要输出其他内容。"

OUTPUT_VERIFICATION_PROMPT = """\
请检查以下回答是否符合手势词典规范：
1. 是否使用了不在术语枚举中的"新发明"术语？
2. 是否有与结构化知识库矛盾的事实性错误？
3. 是否缺少 output_frame 要求的章节？
4. 是否前后矛盾，前后的判断/说明要一致
5. 在对交互机制的适用场景做判断时，要参考书中的内容做判断，不能做出跟书中内容相矛盾的表述（比如按下不适合作为确认提交）

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
