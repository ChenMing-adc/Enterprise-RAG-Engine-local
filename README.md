# 📚 Enterprise RAG Engine (私有知识库引擎)

本项目是一个完全解耦的 RAG (检索增强生成) 系统，基于 DeepSeek 大模型与 LangChain 生态构建。解决了传统 RAG 系统在面对数学公式、多轮代词指代、长尾词召回等场景下的工业级痛点。

## ✨ 核心特性
- **多模态极速解析**：集成 `LlamaParse`，完美提取 PDF 中的复杂表格与 LaTeX 数学公式。
- **智能意图路由 (Agentic Router)**：毫秒级区分“日常闲聊”与“知识检索”，避免无效算力开销。
- **Multi-Query 意图重写**：结合滑动窗口记忆，自动补全多轮对话中的代词指代，并发检索。
- **双路召回 + RRF 融合**：纯手工实现 Dense (M3E语义) + Sparse (BM25关键词) 混合检索与倒数排名融合。
- **Cross-Encoder 交叉重排**：使用 `bge-reranker` 深度计算相关性，剥离噪声，实现极低幻觉率。
- **增量更新持久化**：基于 ChromaDB 实现单篇文档的极速入库与精准擦除。

## 🚀 快速开始 (How to run)

### 1. 克隆项目
```bash
git clone https://github.com/ChenMing-adc/Enterprise-RAG-Engine-local.git
cd Enterprise-RAG-Engine-local
