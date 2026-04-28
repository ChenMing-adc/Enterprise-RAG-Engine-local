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

# 1. 加载环境变量
load_dotenv()
os.makedirs("data", exist_ok=True)

st.set_page_config(page_title="私有文献知识库", page_icon="📚", layout="wide")
st.title("📚 私有文献知识库 (专业级 RAG 版)")

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

    # 拿到检索器
    retriever = vectorstore.as_retriever()

    api_key = os.environ.get("YOUR_DASHSCOPE_API_KEY")
    if not api_key:
        st.error(" 未读取到 YOUR_DASHSCOPE_API_KEY，请检查 .env 文件！")
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

    # 【核心改动】我们把检索器抽离出来了，这里的 rag_chain 只负责生成答案
    rag_chain = prompt | llm | StrOutputParser()

    return retriever, rag_chain


# 获取检索器和生成链
retriever, rag_chain = init_knowledge_base()

# ================= 主界面：流式聊天与溯源 =================
if retriever is None:
    st.info("👈 请先在左侧边栏上传 PDF 文献，并点击【保存并更新知识库】")
else:
    # 渲染历史聊天记录（带上来源信息）
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # 如果这条消息有来源信息，就展示出来
            if "sources" in msg and msg["sources"]:
                with st.expander("📚 查看参考来源"):
                    for i, source in enumerate(msg["sources"]):
                        st.markdown(f"**[{i + 1}] {source['file']} (第 {source['page']} 页)**")
                        st.caption(source['content'])

    if question := st.chat_input("请根据文献内容提问（支持上下文追问）..."):
        # 1. 存入并展示用户问题
        with st.chat_message("user"):
            st.markdown(question)
        st.session_state.messages.append({"role": "user", "content": question})

        # 组装历史记录
        chat_history = []
        for msg in st.session_state.messages[:-1]:
            if msg["role"] == "user":
                chat_history.append(HumanMessage(content=msg["content"]))
            else:
                chat_history.append(AIMessage(content=msg["content"]))

        # 2. AI 处理环节
        with st.chat_message("assistant"):
            # 先检索文档（溯源的秘密就在这里）
            with st.spinner("🔍 正在检索相关文献..."):
                retrieved_docs = retriever.invoke(question)

            # 格式化检索到的文档用于大模型，同时提取来源数据用于 UI 展示
            context_text = ""
            source_data = []
            for i, doc in enumerate(retrieved_docs):
                # 从 PDF 的 metadata 里提取文件名和页码
                file_name = os.path.basename(doc.metadata.get('source', '未知文档'))
                # PDF页码默认从0开始，我们加1变成人类可读的页码
                page_num = doc.metadata.get('page', 0) + 1

                context_text += f"\n[文档 {i + 1}] {file_name} (第{page_num}页):\n{doc.page_content}\n"
                source_data.append({
                    "file": file_name,
                    "page": page_num,
                    "content": doc.page_content[:150] + "..."  # 截取部分文本用于UI展示
                })

            # 【核心魔法：流式输出】
            response_placeholder = st.empty()  # 创建一个空的占位符
            full_response = ""

            # 使用 .stream() 替代 .invoke()，实现打字机效果
            for chunk in rag_chain.stream({
                "context": context_text,
                "question": question,
                "chat_history": chat_history
            }):
                full_response += chunk
                # 动态加上光标，视觉效果拉满
                response_placeholder.markdown(full_response + "▌")

            # 输出结束后去掉光标
            response_placeholder.markdown(full_response)

            # 展示参考来源组件
            if source_data:
                with st.expander("📚 查看参考来源"):
                    for i, source in enumerate(source_data):
                        st.markdown(f"**[{i + 1}] {source['file']} (第 {source['page']} 页)**")
                        st.caption(source['content'])

            # 将完整的回答和来源存入记忆
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "sources": source_data
            })