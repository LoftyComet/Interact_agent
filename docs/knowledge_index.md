# 知识索引构建与评测

## 设计目标

索引是可独立部署的知识包，问答代码只依赖 `HybridRetriever` 接口，不感知文档格式、切分规则或底层检索引擎。当前支持：

- DOC、DOCX、PDF、Markdown、TXT、RTF、XLS/XLSX 统一抽取；
- 按标题层级切分，保留页码、段落、表格等来源位置；
- SHA-256 精确去重、主资料/背景资料权威级别；
- 术语、别名、关系和 chunk-term 映射；
- SQLite FTS5/BM25 中文双字词法索引；
- 可选向量索引、RRF 融合、权威度加权、邻块扩展；
- `manifest.json` 版本号和逐文件哈希校验。

## 一键构建

在项目根目录执行：

```bash
uv sync --group dev
.venv/bin/python scripts/build_knowledge_index.py /path/to/corpus \
  --output-dir knowledge_index
```

默认只构建本地词法索引，不需要 API Key。若 `.env` 已配置 `SILICONFLOW_API_KEY`，可以同时构建向量索引：

```bash
.venv/bin/python scripts/build_knowledge_index.py /path/to/corpus \
  --output-dir knowledge_index \
  --vectors
```

向量模型由 `SILICONFLOW_EMBEDDING_MODEL` 指定，默认 `Pro/BAAI/bge-m3`。增量重建时，相同内容哈希的向量会复用。

主要产物：

| 文件 | 用途 |
|---|---|
| `corpus_inventory.json` | 文件纳入、排除、权威度和去重审计 |
| `documents.jsonl` | 与文件格式无关的标准化文档 |
| `chunks.jsonl` | 结构感知的检索片段及来源位置 |
| `terminology.json` / `relations.jsonl` | 术语、别名和知识关系 |
| `knowledge.sqlite` | 元数据与 BM25 词法索引 |
| `vectors/` | 可选的向量矩阵与行映射 |
| `manifest.json` | 构建 ID、数量、模型信息和完整性哈希 |

索引构建会忽略 `knowledge_index/`、虚拟环境、隐藏生成目录和项目派生资料；`用户输入.xlsx` 被识别为评测集，不会混进回答语料。

## 离线评测

```bash
.venv/bin/python scripts/evaluate_retrieval.py /path/to/用户输入.xlsx \
  --index-dir knowledge_index \
  --output knowledge_index/evaluation_baseline.json
```

评测同时记录逐题结果和汇总指标：意图准确率、非空检索率、主资料命中率、期望术语命中率及宏平均术语召回。`control_form_compare` 和 `control_form_application` 会按项目现行 taxonomy 分别映射到 `interaction_compare` 和 `design_suggestion`；`feedback_*` 属于多轮反馈标签，不计入单轮 intent 准确率。

当前 33 条本地基线（词法模式，top-k=6）：

- 可评测 intent 准确率：93.10%（27/29）；
- 非空检索率：100%；
- 主资料命中率：87.88%；
- 有期望术语的问题至少命中一个术语：100%；
- 宏平均期望术语召回：85.12%。

## 运行时接入

`agent_config.json` 可显式配置：

```json
{
  "data": {
    "data_dir": "data",
    "knowledge_index": "knowledge_index"
  }
}
```

若未配置，但项目根存在 `knowledge_index/manifest.json`，Web 后端会自动启用。`/api/health` 的 `retrieval_mode` 为 `lexical` 或 `hybrid`，`index_chunk_count` 显示已加载 chunk 数。向量包存在且服务器配置了兼容的 SiliconFlow embedding 模型时自动启用混合检索；否则安全回退到词法检索。

回答仍以流式 Markdown 返回。完成后会经过 `AnswerDocument` 契约校验：必须先给直接结论，再按 intent 模板完整输出非空章节，引用编号不得超出本轮检索资料范围。有效回答会按模板顺序规范化渲染。

## 更新与发布

语料变更后重新执行一键构建和离线评测。部署时至少上传以下内容：

- 项目代码与 `data/` 中的术语、结构化知识、输出框架配置；
- 完整的 `knowledge_index/`；
- 图片回答需要的 `data/pictures/extracted/`；
- 服务器自己的 `.env` 和 `agent_config.json`。

原始 DOCX/PDF 不参与线上检索运行；`documents.jsonl` 和 `chunks.jsonl` 已携带标准化语料及溯源位置。若出于审计或重新构建需要，可以另行保存原始语料，不必放在 Web 进程目录。
