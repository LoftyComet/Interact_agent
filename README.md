# 手势词典学习 Agent 原型

这是一个面向“手势词典”的学习与设计评估 agent。它先在本地完成问题结构拆分和资料检索，再通过 OpenAI 兼容 SDK 把结构化上下文交给硅基流动 Chat Completions API 生成答案。

## 当前支持的 Intent

1. `basic_interaction_mechanism`：基础交互机制，例如“什么是单击？”
2. `advanced_interaction_mechanism`：高级交互机制，例如“快击如何解决单击和长按的冲突？”
3. `control_form`：控件形态，例如“旋钮可以承载哪些属性？”
4. `basic_property`：基础属性，例如“二元属性的关键性质是什么？”
5. `multimodal_interaction`：多模态交互，例如“语音和手势如何分工？”
6. `voice_interaction`：语音交互，例如“语音交互需要怎样设计反馈闭环？”
7. `podcast_content`：播客内容，例如“帮我写一期关于长按的播客脚本”
8. `interaction_compare`：交互机制对比，例如“单击和长按有什么区别？”
9. `background_knowledge`：背景知识，例如“交互的本质和操控力视角是什么？”
10. `case_analysis`：理解交互案例，支持文字案例，也预留图片案例输入。
11. `design_evaluation`：设计方案评估，例如“请评估这个设计方案是否合理，并用手势词典术语给修改建议。”

## 问题结构

每个用户问题会先被拆成：

- `intent`：上述 11 类任务类型
- `layers`：`basic_property`、`interaction_mechanism`、`control_form`、`interaction_case`、`design_evaluation`
- `terms`：命中的词典术语
- `focus`：定义、属性、逻辑关系、交互特性、适用边界等
- `output_frame`：回答时必须遵循的结构

## 使用

```bash
uv sync --group dev
cp .env.example .env
# 编辑 .env，将 SILICONFLOW_API_KEY 改成你的硅基流动 API Key
uv run python -m gesture_agent.cli "单击和长按有什么区别？" --show-structure --show-context
```

只看本地拆解和检索，不调用 API：

```bash
uv run python -m gesture_agent.cli "拖拽是什么？" --dry-run
```

交互机制对比资料会读取 `data/交互机制对比.md`，并按小节切块检索。例如“快击和点击缓冲有什么区别？”会优先命中“快击 VS 点击缓冲”小节，而不是只检索到整篇文档。

连续提问：

```bash
uv run python -m gesture_agent.cli --interactive
```

交互式模式会管理澄清 session 和短期对话记忆。模糊问题不会直接回答，而是先反问；用户补充信息后，系统会合并上下文重新判断 intent。已经回答过的轮次会被记录，后续追问如“那它和长按有什么区别？”会结合前文补全“它”的指代，并重新判断 intent。

使用大模型辅助判断 intent：

```bash
uv run python -m gesture_agent.cli --interactive --llm-intent
```

交互式模式默认使用“规则 + LLM 兜底”：先用本地规则判断；如果本地规则认为信息不足、准备反问，系统会调用硅基流动模型结合对话记忆再判断一次。如果 API 不可用，会回退到本地规则。

`--llm-intent` 会让每一轮都使用大模型辅助判断 intent。`--no-llm-clarify` 可以关闭默认的 LLM 兜底。`--dry-run` 永远不调用 API，因此会自动跳过大模型 intent 判断。

检查 API 连通性：

```bash
uv run python -m gesture_agent.cli --check-api --timeout 20
```

流式输出：

```bash
uv run python -m gesture_agent.cli "什么是单击？" --stream --timeout 60
```

图片案例：

```bash
uv run python -m gesture_agent.cli "请用手势词典结构分析这张图里的交互" --image ./case.png
```

传入 `--image` 时，如果没有显式设置 `--model`，系统会优先使用 `.env` 中的 `SILICONFLOW_VISION_MODEL`。

设计方案评估：

```bash
uv run python -m gesture_agent.cli \
  "请评估这个设计方案：在音乐播放器界面，用户长按音量旋钮后拖动来调节音量，松手后系统高亮确认。" \
  --dry-run
```

评估模式会先把方案拆成“控件形态、基础属性、交互机制、响应逻辑、系统反馈”，再按手势词典术语输出问题诊断和修改建议。

## 配置

默认会自动读取项目根目录的 `.env`，也可以用 shell 环境变量覆盖：

- `SILICONFLOW_API_KEY`：必填。
- `SILICONFLOW_MODEL`：文本问答默认模型，默认 `Qwen/Qwen3-32B`。
- `SILICONFLOW_VISION_MODEL`：图片案例默认模型，需要填写硅基流动支持视觉输入的模型；命令行 `--model` 会覆盖它。
- `SILICONFLOW_BASE_URL`：默认 `https://api.siliconflow.cn/v1`。
- `SILICONFLOW_TIMEOUT`：默认 `60` 秒。推理模型如 DeepSeek-R1 可能更慢，可适当调大；如果希望快速响应，建议换非推理模型。
- `SILICONFLOW_MAX_RETRIES`：默认 `0`，避免网络不通时等待多次重试。

如果命令长时间没有输出，可以先用 `--dry-run` 确认本地检索是否正常；如果出现连接错误，用下面的命令检查网络/DNS：

```bash
curl -I https://api.siliconflow.cn/v1/models
```

如果出现 `request timed out`，说明请求已经发出但模型没有在超时时间内完整返回。可以优先使用 `--stream`，或把 `--timeout` 调大到 `60` / `120`。

如果回答停在半句，通常是输出达到 `--max-tokens` 上限。把它调大即可：

```bash
uv run python -m gesture_agent.cli "什么是单击？" --stream --timeout 120 --max-tokens 1200 --top-k 3
```

调用方式使用 OpenAI 兼容接口：

```python
import os

from openai import OpenAI

client = OpenAI(
    api_key=os.environ["SILICONFLOW_API_KEY"],
    base_url="https://api.siliconflow.cn/v1",
)

response = client.chat.completions.create(
    model="Qwen/Qwen3-32B",
    messages=[{"role": "user", "content": "什么是单击？"}],
)
```

## 设计说明

本原型不是简单 RAG 聊天。它把你的资料按三层知识组织：基本属性、交互机制、控件形态。回答时要求模型使用“控件属性 -> 交互机制 -> 响应逻辑 -> 适用边界”的结构，避免只给泛泛解释。

## 汇报文档

- [从用户输入到回答的处理流程](docs/agent_flow.md)
