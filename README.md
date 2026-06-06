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

知识库加载时还会生成 `term_inventory` 术语枚举，按 `basic_property`、`interaction_mechanism`、`control_form` 等层级保存手势词典中的规范术语。模型回答时会把这些枚举注入 prompt，要求控件形态、基础属性、交互机制等专有名词必须来自枚举或检索资料标题，避免创造混乱术语。

专家标注后的术语可以写在 `by_type` 中，例如“基础交互机制、控件形态、响应类型”；旧的 `by_layer` 仍兼容，用于和代码内部检索层级对齐。

`term_inventory.json` 还支持 `aliases`，用于把用户口语映射为规范术语：

```json
{
  "aliases": {
    "点一下": "单击",
    "按住": "长按",
    "拖动": "拖拽"
  }
}
```

查看当前生效的术语枚举：

```bash
uv run python -m gesture_agent.cli --show-term-inventory
```

导出当前术语枚举作为可编辑配置：

```bash
uv run python -m gesture_agent.cli --export-term-inventory data/term_inventory.json
```

也可以从模板开始：

```bash
cp data/term_inventory.example.json data/term_inventory.json
```

`data/term_inventory.json` 会被默认读取。配置中的 `mode` 支持：

- `merge`：把你定义的术语放在自动生成术语前面，同时保留自动生成结果。
- `replace`：完全使用配置文件中的术语枚举。

如果想使用其他路径：

```bash
uv run python -m gesture_agent.cli "什么是单击？" --term-inventory ./my_terms.json
```

推荐的专家标注导入格式：

```json
{
  "mode": "merge",
  "by_type": {
    "基础交互机制": ["单击", "双击", "长按", "拖拽"],
    "高级交互机制": ["快击", "点击缓冲", "长按拖拽"],
    "控件形态": ["按钮", "旋钮", "触控面"],
    "响应类型": ["微变", "确认反馈"]
  }
}
```

Intent 输出框架也可以由专家标注后配置。默认读取 `data/output_frames.json`，也可以从模板开始：

```bash
cp data/output_frames.example.json data/output_frames.json
```

查看或导出当前生效的输出框架：

```bash
uv run python -m gesture_agent.cli --show-output-frames
uv run python -m gesture_agent.cli --export-output-frames data/output_frames.json
```

配置中的 `mode` 支持：

- `merge`：只覆盖配置中写到的 intent，未写的 intent 继续使用默认输出框架。
- `replace`：完全使用配置文件；必须提供全部 11 类 intent 的输出框架。

如果想使用其他路径：

```bash
uv run python -m gesture_agent.cli "什么是单击？" --output-frames ./my_output_frames.json
```

结构化知识库可以从 Markdown 自动生成，也可以导出后人工编辑：

```bash
uv run python -m gesture_agent.cli --export-structured-knowledge data/structured_knowledge.json
```

`data/structured_knowledge.json` 会被默认读取。它用于 Query Rewrite、混合检索和 rerank，例如把“单击”扩展为对应的术语类型、相关属性、关联机制和响应逻辑。

## 使用

```bash
uv sync --group dev
cp .env.example .env
# 编辑 .env，将 SILICONFLOW_API_KEY 改成你的硅基流动 API Key
cp agent_config.example.json agent_config.json
# 编辑 agent_config.json，配置模型参数、检索数量、是否流式输出和 Prompt 文案
uv run python -m gesture_agent.cli
```

默认命令 `uv run python -m gesture_agent.cli` 会读取 `agent_config.json` 并进入连续问答模式；启动后直接在命令行输入问题即可。`agent_config.json` 已加入 `.gitignore`，适合存放本机运行偏好；需要给别人参考时改 `agent_config.example.json`。

`agent_config.json` 支持 `//` 单行注释和 `/* ... */` 块注释，可以直接在参数旁边写说明。

如果只想临时问一次，也可以继续把问题放在命令里：

```bash
uv run python -m gesture_agent.cli "单击和长按有什么区别？"
```

只看本地拆解和检索，不调用 API：

```bash
uv run python -m gesture_agent.cli "拖拽是什么？" --dry-run
```

交互机制对比资料会读取 `data/交互机制对比.md`，并按小节切块检索。例如“快击和点击缓冲有什么区别？”会优先命中“快击 VS 点击缓冲”小节，而不是只检索到整篇文档。

连续提问：

```bash
uv run python -m gesture_agent.cli
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

## Web 界面

除了 CLI，项目还提供一个 Flask 后端 + 纯静态前端的 Web 界面，复用与 CLI 完全相同的 `KnowledgeBase` / `QuestionParser` / `ConversationSession` / `SiliconFlowClient` 接口。

```bash
# 推荐：用 uv 启动，自动注入 flask、flask-cors
./web/run.sh

# 或用本机 python
pip install -r web/requirements.txt
PYTHONPATH=. python web/backend/app.py
```

默认监听 `http://127.0.0.1:5050`，前端由 Flask 同源托管，浏览器打开即可使用。常用环境变量：

- `WEB_HOST`：监听地址，默认 `127.0.0.1`。
- `WEB_PORT`：监听端口，默认 `5050`。
- `WEB_DEBUG`：设为 `1` 启用 Flask 调试。
- `AGENT_CONFIG_PATH`：复用与 CLI 相同的 agent 配置，默认 `agent_config.json`。
- `SILICONFLOW_API_KEY` 仍从项目根的 `.env` 读取。

后端 API（详见 [web/README.md](web/README.md)）：

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET`  | `/api/health` | 后端、模型、知识库状态 |
| `GET`  | `/api/intents` | 当前生效的 intent 输出框架 |
| `POST` | `/api/session` | 创建新的 `session_id` |
| `POST` | `/api/reset` | 清空指定会话的多轮记忆 |
| `POST` | `/api/ask` | `{question, session_id, images?}` 一次性回答 |
| `POST` | `/api/ask_stream` | 同上，返回 SSE 流式输出 |

前端会把 `session_id` 写入 `localStorage`，刷新页面复用同一会话，支持图片上传、流式输出、中断回答和清空会话；模糊问题会先反问澄清。

## 服务器部署

`deploy/` 提供了基于 Nginx + systemd 的生产部署模板，完整步骤见 [deploy/README.md](deploy/README.md)：

- `deploy/systemd/gesture-agent.service`：systemd 服务单元。
- `deploy/nginx/gesture-agent.conf`：Nginx 反向代理配置。

## 配置

运行配置优先从 `agent_config.json` 读取，适合放模型参数、检索参数、是否流式输出、是否显示结构、Prompt 文案等。命令行参数仍然保留，用于临时覆盖配置文件。

常用配置项：

- `data.data_dir`：资料目录，默认 `data`。
- `data.term_inventory`：术语枚举配置，默认可指向 `data/term_inventory.json`。
- `data.output_frames`：Intent 输出框架配置，默认可指向 `data/output_frames.json`。
- `data.structured_knowledge`：结构化知识库配置，默认可指向 `data/structured_knowledge.json`。
- `retrieval.top_k`：检索资料片段数量。
- `model.model`：硅基流动模型名；为 `null` 时读取 `.env` 中的 `SILICONFLOW_MODEL`。
- `model.timeout`、`model.max_tokens`、`model.temperature`：模型调用参数。
- `model.enable_thinking`：是否向硅基流动发送 `enable_thinking`；默认 `null` 表示不发送，视觉模型通常应保持 `null`。
- `runtime.default_interactive`：没有在命令中输入问题时是否默认进入连续问答。
- `runtime.stream`：是否流式输出。
- `intent.llm_intent`：是否每轮都用大模型判断 intent。
- `intent.llm_clarify`：本地规则信息不足时是否用大模型二次判断。
- `prompt.system_prompt`：完整替换系统提示词；为 `null` 时使用内置默认系统提示词。
- `prompt.extra_system_prompt`：追加到默认系统提示词后。
- `prompt.response_instructions`：完整替换回答规则；为 `null` 时使用内置默认回答规则。
- `prompt.extra_response_instructions`：追加回答规则。

API key 仍然放在 `.env`，也可以用 shell 环境变量覆盖：

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
- [检索能力升级汇报](docs/retrieval_upgrade_report.md)
