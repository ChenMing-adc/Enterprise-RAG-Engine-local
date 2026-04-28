from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder
from langchain_core.runnables import RunnableLambda

from config import *


def build_advanced_retriever():
    """构建 双路召回 + RRF融合 + Cross-Encoder重排 的终极检索器"""
    loader = PyPDFDirectoryLoader(DATA_DIR)
    docs = loader.load()

    if not docs:
        return None

    # 文档切块
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    splits = text_splitter.split_documents(docs)

    # 向量检索 (Dense)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K})

    # 关键词检索 (Sparse)
    bm25_retriever = BM25Retriever.from_documents(splits)
    bm25_retriever.k = RETRIEVER_K

    # 精排模型
    cross_encoder = CrossEncoder(RERANKER_MODEL)

    # 手写 RRF 混合检索核心逻辑
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