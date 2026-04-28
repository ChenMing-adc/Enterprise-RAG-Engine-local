import os
from llama_parse import LlamaParse
import nest_asyncio
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder
from langchain_core.runnables import RunnableLambda

from config import *

nest_asyncio.apply()


def get_vectorstore():
    """获取持久化的 Chroma 向量库连接"""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    # 指定 persist_directory 实现从内存模式切换为【硬盘持久化模式】
    return Chroma(persist_directory=CHROMA_PERSIST_DIR, embedding_function=embeddings)


def add_pdf_to_db(file_path):
    """【增】：解析单个新 PDF 并追加进数据库 (带空数据防御机制)"""
    parser = LlamaParse(
        api_key=os.environ.get("LLAMA_CLOUD_API_KEY"),
        result_type="markdown",
        verbose=True
    )
    llama_docs = parser.load_data([file_path])

    langchain_docs = []
    filename = os.path.basename(file_path)
    for doc in llama_docs:
        # 有些极端情况下 doc.text 可能是 None 或者空字符串，我们也要过滤掉
        text_content = getattr(doc, 'text', '').strip()
        if not text_content:
            continue

        meta = doc.metadata or {}
        # 强行注入 filename 标识，这是我们以后精准删除的“主键”
        meta["file_name"] = filename
        langchain_docs.append(Document(page_content=text_content, metadata=meta))

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    splits = text_splitter.split_documents(langchain_docs)

    # 【新增：防御性拦截】如果切出来的块是空的（说明 PDF 没提取出有效文字），直接终止，不写入数据库！
    if not splits:
        print(f"⚠️ 警告：文件 {filename} 解析后未提取到任何有效文本，已跳过入库。")
        return

    vs = get_vectorstore()
    vs.add_documents(splits)


def delete_pdf_from_db(filename):
    """【删】：根据文件名从数据库中精确清除对应的向量切块"""
    vs = get_vectorstore()
    try:
        # 调用 Chroma 底层的 delete 方法，利用 metadata 过滤器精准打击
        vs._collection.delete(where={"file_name": filename})
    except Exception as e:
        print(f"数据库中未找到 {filename} 或删除失败: {e}")


def build_advanced_retriever():
    """构建双路检索器"""
    vs = get_vectorstore()

    # 【神级优化】：我们直接从硬盘数据库中拉取现有数据来初始化 BM25！
    # 彻底告别每次启动都要重新解析一遍 PDF 的噩梦
    db_data = vs.get()
    if not db_data['documents']:
        return None

    splits = [
        Document(page_content=txt, metadata=meta)
        for txt, meta in zip(db_data['documents'], db_data['metadatas'])
    ]

    # 1. 向量检索 (Dense)
    vector_retriever = vs.as_retriever(search_kwargs={"k": RETRIEVER_K})

    # 2. 关键词检索 (Sparse)
    bm25_retriever = BM25Retriever.from_documents(splits)
    bm25_retriever.k = RETRIEVER_K

    # 3. 精排模型
    cross_encoder = CrossEncoder(RERANKER_MODEL)

    def advanced_retrieve(query: str):
        dense_docs = vector_retriever.invoke(query)
        sparse_docs = bm25_retriever.invoke(query)

        rrf_scores = {}
        doc_map = {}

        for rank, doc in enumerate(dense_docs):
            content = doc.page_content
            doc_map[content] = doc
            rrf_scores[content] = rrf_scores.get(content, 0) + 1.0 / (RRF_K + rank + 1)

        for rank, doc in enumerate(sparse_docs):
            content = doc.page_content
            doc_map[content] = doc
            rrf_scores[content] = rrf_scores.get(content, 0) + 1.0 / (RRF_K + rank + 1)

        fused_contents = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:RETRIEVER_K]
        candidate_docs = [doc_map[content] for content in fused_contents]

        if not candidate_docs:
            return []

        pairs = [[query, doc.page_content] for doc in candidate_docs]
        scores = cross_encoder.predict(pairs)

        for doc, score in zip(candidate_docs, scores):
            doc.metadata['relevance_score'] = float(score)

        candidate_docs.sort(key=lambda x: x.metadata['relevance_score'], reverse=True)
        return candidate_docs[:RERANK_TOP_K]

    return RunnableLambda(advanced_retrieve)