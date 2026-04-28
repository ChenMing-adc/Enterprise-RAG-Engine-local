import os
import streamlit as st
from dotenv import load_dotenv

# 文档处理组件
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 模型与 LCEL 组件
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

# 【第一处核心新增】引入 Rerank 相关的 LangChain 模块
# 【替换掉之前的 LangChain 压缩包导入】
from sentence_transformers import CrossEncoder
from langchain_core.runnables import RunnableLambda
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

load_dotenv()
os.makedirs("data", exist_ok=True)

st.set_page_config(page_title="私有文献知识库", page_icon="📚", layout="wide")
st.title("📚 私有文献知识库 (Rerank重排加强版)")

# ================= 侧边栏：文件上传与管理 =================
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("📁 文档管理")
    uploaded_files = st.file_uploader(
        "请在这里拖拽或选择 PDF 文献",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}"
    )

    if st.button("保存并更新知识库"):
        if uploaded_files:
            for uploaded_file in uploaded_files:
                file_path = os.path.join("data", uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
            st.cache_resource.clear()
            st.session_state.uploader_key += 1
            st.session_state.messages = []
            st.rerun()
        else:
            st.warning("请先上传文件！")

    st.markdown("---")
    st.markdown("### 当前已有文献：")
    existing_files = [f for f in os.listdir("data") if f.endswith('.pdf')]
    if existing_files:
        for f in existing_files:
            st.text(f"📄 {f}")
    else:
        st.text("暂无文献，请上传")


# ================= 核心逻辑：解耦检索与生成 =================
@st.cache_resource
def init_knowledge_base():
    loader = PyPDFDirectoryLoader("data")
    docs = loader.load()

    if not docs:
        return None, None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="shibing624/text2vec-base-chinese")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

    # ================= 【纯手工底层重排架构】 =================
    # 1. 基础检索器（召回）：一次性取出 10 个片段
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

    # 2. 原生加载 BAAI 重排模型
    cross_encoder = CrossEncoder("BAAI/bge-reranker-base")

    # 3. 手写重排核心逻辑！(面试可以重点讲这个函数)
    def advanced_retrieve(query: str):
        # 阶段一：粗搜召回 Top 10
        docs = base_retriever.invoke(query)
        if not docs:
            return []

        # 阶段二：重排打分
        # 将用户问题和每个文档片段组合成一对 (Pair)
        pairs = [[query, doc.page_content] for doc in docs]
        # Cross-Encoder 进行交叉注意力计算，输出得分
        scores = cross_encoder.predict(pairs)

        # 将算出来的精确分数存入文档的 metadata 中
        for doc, score in zip(docs, scores):
            doc.metadata['relevance_score'] = float(score)

        # 按照分数从高到低排序 (降序)
        docs.sort(key=lambda x: x.metadata['relevance_score'], reverse=True)

        # 阶段三：切片，只返回最精准的 Top 3 给大模型
        return docs[:3]

    # 4. 用 RunnableLambda 将我们的手写函数无缝接入 LangChain 的 LCEL 管道
    advanced_retriever = RunnableLambda(advanced_retrieve)
    # =========================================================

    api_key = os.environ.get("YOUR_DASHSCOPE_API_KEY")
    if not api_key:
        st.error("⚠️ 未读取到YOUR_DASHSCOPE_API_KEY，请检查 .env 文件！")
        st.stop()

    llm = ChatOpenAI(
        model="qwen3-max",
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature=0,
        streaming=True
    )

    system_template = """你是一个专业的文献助手。请使用以下检索到的背景信息来回答用户的问题。
    如果你不知道答案，请直接说不知道，不要编造。
    在回答的末尾，你可以简要提及你是根据哪些资料得出的结论。

    背景信息：
    {context}
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])

    rag_chain = prompt | llm | StrOutputParser()

    # 注意这里返回的是加强版的 advanced_retriever
    return advanced_retriever, rag_chain


retriever, rag_chain = init_knowledge_base()

# ================= 主界面：流式聊天与溯源 =================
if retriever is None:
    st.info("👈 请先在左侧边栏上传 PDF 文献，并点击【保存并更新知识库】")
else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "sources" in msg and msg["sources"]:
                with st.expander("📚 查看参考来源 (Rerank Top-3)"):
                    for i, source in enumerate(msg["sources"]):
                        # Rerank 后返回的文档，metadata 里会多出一个 'relevance_score' 字段！
                        score_text = f" (相关度得分: {source['score']:.4f})" if 'score' in source else ""
                        st.markdown(f"**[{i + 1}] {source['file']} (第 {source['page']} 页){score_text}**")
                        st.caption(source['content'])

    if question := st.chat_input("请根据文献内容提问（支持上下文追问）..."):
        with st.chat_message("user"):
            st.markdown(question)
        st.session_state.messages.append({"role": "user", "content": question})

        chat_history = []
        for msg in st.session_state.messages[:-1]:
            if msg["role"] == "user":
                chat_history.append(HumanMessage(content=msg["content"]))
            else:
                chat_history.append(AIMessage(content=msg["content"]))

        with st.chat_message("assistant"):
            with st.spinner("🔍 Reranker 引擎正在进行交叉重排精搜..."):
                # 这里调用的是 advanced_retriever，它会自动执行：取10个 -> Rerank打分 -> 返回Top 3
                retrieved_docs = retriever.invoke(question)

            context_text = ""
            source_data = []
            for i, doc in enumerate(retrieved_docs):
                file_name = os.path.basename(doc.metadata.get('source', '未知文档'))
                page_num = doc.metadata.get('page', 0) + 1
                # 获取 Reranker 算出来的精确得分
                score = doc.metadata.get('relevance_score', 0)

                context_text += f"\n[文档 {i + 1}] {file_name} (第{page_num}页):\n{doc.page_content}\n"
                source_data.append({
                    "file": file_name,
                    "page": page_num,
                    "score": score,  # 保存得分用于UI展示
                    "content": doc.page_content[:150] + "..."
                })

            response_placeholder = st.empty()
            full_response = ""

            for chunk in rag_chain.stream({
                "context": context_text,
                "question": question,
                "chat_history": chat_history
            }):
                full_response += chunk
                response_placeholder.markdown(full_response + "▌")

            response_placeholder.markdown(full_response)

            if source_data:
                with st.expander("📚 查看参考来源 (Rerank Top-3)"):
                    for i, source in enumerate(source_data):
                        score_text = f" (相关度得分: {source['score']:.4f})" if 'score' in source else ""
                        st.markdown(f"**[{i + 1}] {source['file']} (第 {source['page']} 页){score_text}**")
                        st.caption(source['content'])

            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "sources": source_data
            })