# Web 前端 (Flask + 静态页面)

`web/` 目录与核心 `gesture_agent/` 模块解耦，只通过 `import gesture_agent.*` 调用既有接口（`KnowledgeBase` / `QuestionParser` / `ConversationSession` / `build_messages` / `SiliconFlowClient`）。

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

默认监听 `http://127.0.0.1:5050`。可用环境变量：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `WEB_HOST` | `127.0.0.1` | 监听地址 |
| `WEB_PORT` | `5050` | 监听端口 |
| `WEB_DEBUG` | `0` | 设为 `1` 启用 Flask 调试 |
| `AGENT_CONFIG_PATH` | `agent_config.json` | 复用与 CLI 相同的 agent 配置 |

`SILICONFLOW_API_KEY` 仍读取项目根的 `.env`（由 `gesture_agent.providers.SiliconFlowClient.from_env` 处理）。

## API

所有响应均为 JSON（流式接口除外）。

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET`  | `/api/health` | 后端、模型、知识库状态 |
| `GET`  | `/api/intents` | 当前生效的 intent 输出框架 |
| `POST` | `/api/session` | 创建一个新的 `session_id` |
| `POST` | `/api/reset`   | `{session_id}` 清空会话记忆 |
| `POST` | `/api/ask`     | `{question, session_id, images?}` 一次性回答 |
| `POST` | `/api/ask_stream` | 同上，返回 `text/event-stream` 流式输出 |

`/api/ask` 返回字段：

```jsonc
{
  "session_id": "…",
  "status": "ready" | "clarify",
  "answer": "…",
  "structure": { /* QuestionStructure.to_dict() */ },
  "chunks":    [ /* {id,title,source,citation,layer,score,…} */ ],
  "memory_context": "…",
  "error": null
}
```

`/api/ask_stream` 的 SSE 事件：

| event | data |
|---|---|
| `meta`    | `{structure, chunks, memory_context, session_id}` |
| `delta`   | `{text}` 单段增量回答 |
| `clarify` | `{message}` 需要用户澄清 |
| `error`   | `{message}` |
| `done`    | `{session_id, answer?}` |

## 前端

`web/frontend/` 是纯 HTML/CSS/JS，无构建工具依赖。

- 默认假设前端与 Flask 同源（由 Flask 直接静态托管），打开 `http://127.0.0.1:5050/` 即可使用。
- 如果想把前端单独部署到别处，先在 HTML 里加：

  ```html
  <script>window.GESTURE_AGENT_API = "http://127.0.0.1:5050";</script>
  <script src="./app.js" defer></script>
  ```

  CORS 已默认放开。

会话状态：前端把 `session_id` 写入 `localStorage`，刷新页面会复用同一会话；点击「清空会话」会调用 `/api/reset` 把 `ConversationSession` 的多轮记忆清掉。
