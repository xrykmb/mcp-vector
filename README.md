# mcp-vector

轻量 **MCP Server**：把文档变成向量，用 **HNSW** 做近似最近邻（ANN）检索，并支持**元数据过滤**。给 Cursor / Claude 等 Agent 当 RAG 工具用，而不是再包一层巨型向量云。

范围收窄：讲清 MCP 工具怎么暴露、向量怎么来、HNSW 在搜什么。

## 解决什么问题

Agent 需要「按语义找文档」，但 embedding API、索引、过滤经常散落在脚本里。用 MCP 做成三个稳定工具：`upsert`、`search`、`stats`，任何兼容 MCP 的客户端都能调。

## 技术栈

- Python 3.10+
- 自研简化 HNSW（分层图 + 贪心下降 + `ef` 候选集），零原生编译依赖，Windows/CI 可复现
- 默认 **signed hashing** 向量（离线可跑）；可选 OpenAI 兼容 `POST /embeddings`
- MCP：官方 SDK `mcp`（stdio）

## 架构

```
文本 → Embedder（hash / HTTP）→ L2 归一化向量
                              → HNSW 分层图
查询 → 同 Embedder → ANN top-N → 元数据精确过滤 → 返回片段
MCP stdio ── tools: upsert_document, search, index_stats
```

HNSW 实现是简化版（邻居选取为「最近 M 个」，不是论文里完整 heuristic）。原理与生产库（hnswlib）同一家族，便于对照实现差异。

## 快速启动

```powershell
git clone https://github.com/xrykmb/mcp-vector.git
cd mcp-vector
python -m venv .venv
.\ .venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m pytest
mcp-vector upsert --id pump --text "轴承每 2000 小时补脂" --meta source=manual
mcp-vector search "润滑周期" -k 3
```

MCP（Cursor `mcp.json` 示例）：

```json
{
  "mcpServers": {
    "mcp-vector": {
      "command": "mcp-vector-server"
    }
  }
}
```

## 演示

索引默认写在 `.mcp-vector/index.json`。`search` 会打印 `id`、余弦相似度、文本、metadata。同一 `--id` 再次 `upsert` 会覆盖旧向量（重建图），不会留下幽灵邻居。

## 未来规划

- [ ] 真实 embedding（BGE / text-embedding-3）
- [ ] 与 [mechmanual](https://github.com/xrykmb/mechmanual) 手册切分打通
- [ ] 可选 hnswlib 后端做对照实验

## License

MIT
