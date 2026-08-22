# 手势词典学习 Agent 原型

这是一个面向”手势词典”的学习与设计评估 agent，提供 Web 界面交互。它在后台完成问题结构拆分和资料检索，再通过 OpenAI 兼容 SDK 把结构化上下文交给多 Provider（DeepSeek、Kimi 等）大模型 API 生成结构化答案。

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

从模板开始：

```bash
cp data/term_inventory.example.json data/term_inventory.json
```

`data/term_inventory.json` 会被默认读取。配置中的 `mode` 支持：

- `merge`：把你定义的术语放在自动生成术语前面，同时保留自动生成结果。
- `replace`：完全使用配置文件中的术语枚举。

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

配置中的 `mode` 支持：

- `merge`：只覆盖配置中写到的 intent，未写的 intent 继续使用默认输出框架。
- `replace`：完全使用配置文件；必须提供全部 11 类 intent 的输出框架。

结构化知识库可以从 Markdown 自动生成，导出为 `data/structured_knowledge.json` 后可供人工编辑。

`data/structured_knowledge.json` 会被默认读取。它用于 Query Rewrite、混合检索和 rerank，例如把“单击”扩展为对应的术语类型、相关属性、关联机制和响应逻辑。

## 使用

```bash
# 1. 安装依赖
uv sync --group dev

# 2. 配置环境
cp .env.example .env
# 编辑 .env，填入 API Key（SILICONFLOW_API_KEY 等）
cp agent_config.example.json agent_config.json
# 编辑 agent_config.json，配置模型参数、检索数量、Prompt 文案等

# 3. 启动 Web 服务
./web/run.sh
# 或: PYTHONPATH=. python web/backend/app.py
```

打开浏览器访问 `http://127.0.0.1:5050` 即可使用。

### 构建完整语料索引

```bash
.venv/bin/python scripts/build_knowledge_index.py /path/to/corpus --output-dir knowledge_index

# 可选：配置 SILICONFLOW_API_KEY 后同时构建向量索引
.venv/bin/python scripts/build_knowledge_index.py /path/to/corpus --output-dir knowledge_index --vectors
```

Web 后端会自动发现 `knowledge_index/manifest.json`，也可以在 `agent_config.json` 的 `data.knowledge_index` 中显式指定。完整的纳入规则、索引产物、离线评测与发布说明见 [知识索引构建与评测](docs/knowledge_index.md)。

`agent_config.json` 支持 `//` 单行注释和 `/* ... */` 块注释，可以直接在参数旁边写说明。该文件已加入 `.gitignore`，适合存放本机运行偏好；需要给别人参考时改 `agent_config.example.json`。

交互式会话管理澄清 session 和短期对话记忆。模糊问题不会直接回答，而是先反问；用户补充信息后，系统会合并上下文重新判断 intent。已经回答过的轮次会被记录，后续追问如”那它和长按有什么区别？”会结合前文补全”它”的指代，并重新判断 intent。

使用大模型辅助判断 intent：默认使用”规则 + LLM 兜底”——先用本地规则判断；如果本地规则认为信息不足、准备反问，系统会调用大模型结合对话记忆再判断一次。如果 API 不可用，会回退到本地规则。

设计方案评估模式会先把方案拆成”控件形态、基础属性、交互机制、响应逻辑、系统反馈”，再按手势词典术语输出问题诊断和修改建议。

## Web 界面

项目提供 Flask 后端 + 纯静态前端的 Web 界面，核心管线为 `KnowledgeBase` / `QuestionParser` / `ConversationSession` / `InputVerifier` / `OutputVerifier` / 多 Provider 客户端。前端是无构建依赖的 HTML/CSS/JS，由 Flask 同源托管，打开浏览器即可使用。

```bash
# 推荐：用 uv 启动，自动注入 flask、flask-cors
./web/run.sh

# 或用本机 python
pip install -r web/requirements.txt
PYTHONPATH=. python web/backend/app.py
```

默认监听 `http://127.0.0.1:5050`，常用环境变量：

- `WEB_HOST`：监听地址，默认 `127.0.0.1`。
- `WEB_PORT`：监听端口，默认 `5050`（`web/run.sh` 设定，`app.py` 直接运行时默认 `5000`）。
- `WEB_DEBUG`：设为 `1` 启用 Flask 调试。
- `AGENT_CONFIG_PATH`：复用与 CLI 相同的 agent 配置，默认 `agent_config.json`。
- API Key 仍从项目根的 `.env` 读取：`DEEPSEEK_API_KEY`（DeepSeek 官方）、`SILICONFLOW_API_KEY`（SiliconFlow）、`KIMI_API_KEY`（Kimi）。

### 前端功能

- 多轮会话：`session_id` 写入 `localStorage`，刷新页面复用同一会话；「清空会话」调用 `/api/reset` 清掉多轮记忆。
- 模型切换：顶部下拉框列出已配置的 Provider（DeepSeek 官方 / SiliconFlow / Kimi），未配置 API Key 的会标注「未配置」并禁用；选择会记到 `localStorage`。
- 回答风格：`简洁` / `详细` 两档，影响回答篇幅。
- 流式输出：默认开启，逐段渲染；可随时点「停止」中断当前回答并保留已生成内容。
- 图片案例：支持多张上传（单张上限 10MB），走视觉模型分析。
- Markdown 渲染：用 marked + DOMPurify 渲染回答；模型若把整段答案包进 ```json``` / ```markdown``` 代码块，会自动展开成正文。
- 引用溯源：回答里的 `[1]`、`[2]` 角标和底部来源 chip 可点击，弹出对应知识库片段（标题、出处、命中术语、原文）。
- 思考指示：请求发出后先显示「正在思考…」，首段返回后消失。
- 问题结构面板：右侧实时显示本轮解析出的 `QuestionStructure`。

### 后端 API

详见 [web/README.md](web/README.md)。

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET`  | `/api/health` | 后端、模型、知识库状态（含术语数、文档数） |
| `GET`  | `/api/intents` | 当前生效的 intent 输出框架 |
| `GET`  | `/api/providers` | 可选模型列表及是否已配置 API Key |
| `GET`  | `/api/images/<filename>` | 知识库抽取出的配图静态托管 |
| `POST` | `/api/session` | 创建新的 `session_id` |
| `POST` | `/api/reset` | 清空指定会话的多轮记忆 |
| `POST` | `/api/ask` | `{question, session_id, images?, style?, provider?}` 一次性回答 |
| `POST` | `/api/ask_stream` | 同上，返回 SSE 流式输出 |

`/api/ask` 与 `/api/ask_stream` 在回答前后会按 `agent_config.json` 的 `verification` 配置做输入对齐、输出格式校验和 Claim—Evidence 语料一致性校验。校验器逐条检查事实陈述是否被其引用片段支持，将结果分为 `supported`、`partially_supported`、`unsupported`、`conflicted`；失败时让模型重写一次。结果通过 `input_corrections`、`output_issues` 和 `grounding` 返回。SSE 在重写时会额外发 `retry` 和 `replace` 事件。

所有 Web 端收到的用户问题会按行追加到 `data/logs/web_questions.jsonl`，用于后续标注和分析。

## 服务器部署

`deploy/` 提供了基于 Nginx + systemd 的生产部署模板，完整步骤见 [deploy/README.md](deploy/README.md)：

- `deploy/systemd/gesture-agent.service`：systemd 服务单元。
- `deploy/nginx/gesture-agent.conf`：Nginx 反向代理配置。

## 配置

运行配置从 `agent_config.json` 读取，适合放模型参数、检索参数、是否流式输出、Prompt 文案等。

常用配置项：

- `data.data_dir`：资料目录，默认 `data`。
- `data.term_inventory`：术语枚举配置，默认可指向 `data/term_inventory.json`。
- `data.output_frames`：Intent 输出框架配置，默认可指向 `data/output_frames.json`。
- `data.structured_knowledge`：结构化知识库配置，默认可指向 `data/structured_knowledge.json`。
- `data.knowledge_index`：可部署知识索引目录；未设置时自动发现项目根的 `knowledge_index/`。
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
- `verification.verify_grounding`：启用逐条 Claim—Evidence 语料一致性校验。
- `verification.grounding_provider` / `grounding_model`：校验使用的 Provider 和模型，推荐 `deepseek` / `deepseek-v4-flash`。
- `verification.grounding_strict`：严格模式下，部分支持的陈述也需要缩小范围或重写。
- `verification.grounding_minimum_score`：回答通过语料一致性校验的最低分数。

API key 仍然放在 `.env`，也可以用 shell 环境变量覆盖：

- `SILICONFLOW_API_KEY`：必填。
- `SILICONFLOW_MODEL`：文本问答默认模型，默认 `Qwen/Qwen3-32B`。
- `SILICONFLOW_VISION_MODEL`：图片案例默认模型，需要填写硅基流动支持视觉输入的模型；命令行 `--model` 会覆盖它。
- `SILICONFLOW_BASE_URL`：默认 `https://api.siliconflow.cn/v1`。
- `SILICONFLOW_TIMEOUT`：默认 `60` 秒。推理模型如 DeepSeek-R1 可能更慢，可适当调大；如果希望快速响应，建议换非推理模型。
- `SILICONFLOW_MAX_RETRIES`：默认 `0`，避免网络不通时等待多次重试。
- `DEEPSEEK_API_KEY`：DeepSeek 官方 API Key。
- `DEEPSEEK_MODEL`：默认 `deepseek-v4-flash`；质量优先可使用 `deepseek-v4-pro`。
- `DEEPSEEK_BASE_URL`：默认 `https://api.deepseek.com`。

DeepSeek V4 在本项目中默认关闭思考模式，避免推理过程耗尽 `max_tokens` 后留下空正文；需要思考模式时可把 `model.enable_thinking` 显式设为 `true`，并相应提高 `model.max_tokens`。

启用 DeepSeek 生成或语料一致性校验时，当前问题检索到的语料片段会发送给 DeepSeek 官方 API。部署前应确认语料允许发送至该外部服务；敏感语料应改用本地模型或私有部署的校验 Adapter。

## 设计说明

本原型不是简单 RAG 聊天。它把你的资料按三层知识组织：基本属性、交互机制、控件形态。回答时要求模型使用“控件属性 -> 交互机制 -> 响应逻辑 -> 适用边界”的结构，避免只给泛泛解释。

## 汇报文档

- [从用户输入到回答的处理流程](docs/agent_flow.md)
- [检索能力升级汇报](docs/retrieval_upgrade_report.md)
