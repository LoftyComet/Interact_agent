# Web 前端 (Flask + 静态页面)

`web/` 目录与核心 `gesture_agent/` 模块解耦，只通过 `import gesture_agent.*` 调用既有接口（`KnowledgeBase` / `QuestionParser` / `ConversationSession` / `InputVerifier` / `OutputVerifier` / `build_messages` / 多 Provider 客户端），完整复用 CLI 的处理管线。

```
web/
├── backend/        # Flask 后端
│   └── app.py
├── frontend/       # 静态前端，无需打包
│   ├── index.html
│   ├── styles.css
│   └── app.js
├── requirements.txt
├── run.sh
└── README.md
```

## 启动

任选其一：

```bash
# 1. 用 uv（推荐，自动注入 flask、flask-cors）
./web/run.sh

# 2. 用本机 python
pip install -r web/requirements.txt
PYTHONPATH=. python web/backend/app.py
```

默认监听 `http://127.0.0.1:5050`（`run.sh` 设定；直接运行 `app.py` 时默认 `5000`）。可用环境变量：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `WEB_HOST` | `127.0.0.1` | 监听地址 |
| `WEB_PORT` | `5050` / `5000` | 监听端口（`run.sh` 用 `5050`，`app.py` 用 `5000`） |
| `WEB_DEBUG` | `0` | 设为 `1` 启用 Flask 调试 |
| `AGENT_CONFIG_PATH` | `agent_config.json` | 复用与 CLI 相同的 agent 配置 |

API Key 仍读取项目根的 `.env`：`SILICONFLOW_API_KEY`（DeepSeek V3.2）、`KIMI_API_KEY`（Kimi）。由各 Provider 客户端的 `from_env` 处理。

## API

所有响应均为 JSON（流式接口除外）。

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET`  | `/api/health` | 后端、模型、知识库状态 |
| `GET`  | `/api/intents` | 当前生效的 intent 输出框架 |
| `GET`  | `/api/providers` | 可选模型列表及是否已配置 API Key |
| `GET`  | `/api/images/<filename>` | 知识库抽取出的配图静态托管 |
| `POST` | `/api/session` | 创建一个新的 `session_id` |
| `POST` | `/api/reset`   | `{session_id}` 清空会话记忆 |
| `POST` | `/api/ask`     | 一次性回答 |
| `POST` | `/api/ask_stream` | 返回 `text/event-stream` 流式输出 |

### 请求体（`/api/ask` 与 `/api/ask_stream`）

```jsonc
{
  "question": "拖拽和滑动有什么区别？",
  "session_id": "…",              // 缺省时后端生成
  "images": ["data:image/png;base64,…"],  // 可选，图片案例
  "style": "concise",             // "concise" | "detailed"，缺省 concise
  "provider": "siliconflow"       // 可选，见 /api/providers；无效值回退默认 provider
}
```

### `/api/health` 返回

```jsonc
{
  "status": "ok",
  "data_dir": "data",
  "model": "…",
  "base_url": "…",
  "top_k": 6,
  "term_count": 37,
  "doc_count": 128
}
```

### `/api/providers` 返回

```jsonc
{
  "providers": [
    { "id": "siliconflow", "label": "DeepSeek V3.2", "configured": true },
    { "id": "kimi",        "label": "Kimi",          "configured": false }
  ]
}
```

新增 Provider 只需在 `gesture_agent/providers/registry.py` 的 `PROVIDERS` 里加一项；前端通过 `/api/providers` 自动拿到列表并回传选中的 `id`。

### `/api/ask` 返回字段

```jsonc
{
  "session_id": "…",
  "status": "ready" | "clarify",   // clarify 时只带 message
  "answer": "…",
  "structure": { /* QuestionStructure.to_dict() */ },
  "chunks":    [ /* {id,title,source,citation,layer,score,start_line,end_line,terms,text} */ ],
  "memory_context": "…",
  "input_corrections": "…",        // 输入对齐摘要，无修正时为空串
  "output_issues": [ "…" ],        // 输出校验残留问题
  "error": null
}
```

模糊问题返回 `{"session_id", "status": "clarify", "message"}`，由前端提示用户补充信息。

### `/api/ask_stream` 的 SSE 事件

| event | data |
|---|---|
| `meta`    | `{structure, chunks, memory_context, input_corrections, session_id}` |
| `delta`   | `{text}` 单段增量回答 |
| `retry`   | `{reason}` 输出校验未通过，开始重写 |
| `replace` | `{text}` 用重写后的完整答案整体替换 |
| `clarify` | `{message}` 需要用户澄清 |
| `error`   | `{message}` |
| `done`    | `{session_id, answer?, output_issues?}` |

### 输入对齐 / 输出校验

`/api/ask` 与 `/api/ask_stream` 在回答前后会按 `agent_config.json` 的 `verification` 配置：

- 输入对齐：把用户口语、近义、错写的术语对齐到规范的「36+1」枚举（规则层零成本常驻，可选 LLM 语义兜底），结果通过 `input_corrections` 返回。
- 输出校验：检查回答的章节完整性和术语合规；未通过时最多让模型重写 `output_max_retries` 次，残留问题通过 `output_issues` 返回。

### 配图与引用

- 知识库 Markdown 中以 `![alt](image:<id>)` 引用的配图，回答里会被解析为 `/api/images/<filename>`（文件名做 URL 编码，避免空格/中文截断），图片实体来自 `data/pictures/extracted/`。
- `chunks` 携带 `title`/`citation`/`layer`/`terms`/`text`，前端据此把回答中的 `[1]`、`[2]` 角标和底部来源 chip 做成可点击的溯源弹窗。

### 日志

每个收到的用户问题按行追加到 `data/logs/web_questions.jsonl`（含时间戳、`session_id`、`endpoint`、问题、图片数、`style`），写日志失败不会影响请求。

## 前端

`web/frontend/` 是纯 HTML/CSS/JS，无构建工具依赖。

- 默认假设前端与 Flask 同源（由 Flask 直接静态托管），打开 `http://127.0.0.1:5050/` 即可使用。
- Markdown 渲染依赖 CDN 上的 marked + DOMPurify；若 CDN 不可用，会回退到内置的极简 Markdown 渲染。
- 如果想把前端单独部署到别处，先在 HTML 里加：

  ```html
  <script>window.GESTURE_AGENT_API = "http://127.0.0.1:5050";</script>
  <script src="./app.js" defer></script>
  ```

  CORS 已默认放开。

### 交互能力

- 会话状态：`session_id` 写入 `localStorage`，刷新页面复用同一会话；「清空会话」调用 `/api/reset` 清掉 `ConversationSession` 的多轮记忆。
- 模型切换：下拉框来自 `/api/providers`，未配置 API Key 的标注「（未配置）」并禁用；选择记到 `localStorage`。
- 回答风格：`简洁` / `详细` 两档。
- 流式输出：默认开启，逐段渲染；点「停止」用 `AbortController` 中断并保留已生成内容。
- 图片上传：支持多张（单张上限 10MB），客户端读成 dataURL 后随请求发送。
- 溯源：回答中的 `[n]` 角标和底部来源 chip 可点击，弹窗展示对应 chunk。
- 思考指示：请求发出后显示「正在思考…」，首段返回后消失。
- 输入法兼容：处理中文 IME 的 `compositionend`，避免回车误触发送。
