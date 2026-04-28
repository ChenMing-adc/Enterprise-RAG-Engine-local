import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 数据目录
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# 【新增】向量数据库持久化目录
CHROMA_PERSIST_DIR = "chroma_db"
os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)

# 模型配置
EMBEDDING_MODEL = "moka-ai/m3e-base"
RERANKER_MODEL = "BAAI/bge-reranker-base"
LLM_MODEL = "qwen3-max"
LLM_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
LLM_API_KEY = os.environ.get("YOUR_DASHSCOPE_API_KEY")

# RAG 超参数配置
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
RETRIEVER_K = 20       # 粗搜召回数量
RERANK_TOP_K = 5       # 精排最终保留数量
RRF_K = 60             # RRF 融合平滑常数