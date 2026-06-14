"""独立的 HCI 论文 RAG 子系统。

抓取 CHI/UIST 等会议论文的元数据 + 摘要(OpenAlex),用 SiliconFlow 的
embedding 模型做语义向量检索,生成带引用的回答。与 gesture_agent 的中文
手势词典系统相互独立,仅复用 SiliconFlow provider 与 env 工具。
"""

from .models import Paper, RetrievedPaper

__all__ = ["Paper", "RetrievedPaper"]
