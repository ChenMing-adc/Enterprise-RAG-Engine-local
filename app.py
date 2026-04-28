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
# 【新增】引入 LangChain 的消息格式
from langchain_core.messages import HumanMessage, AIMessage

# 1. 加载环境变量
load_dotenv()

# 确保 data 文件夹存在
os.makedirs("data", exist_ok=True)

st.set_page_config(page_title="私有文献知识库", page_icon="📚", layout="wide")
st.title("📚 私有文献知识库 (记忆对话版)")

# ================= 侧边栏：文件上传与管理 =================
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
# 【新增】初始化聊天记录状态
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
            # 【细节】更新知识库后，清空之前的聊天记录，避免串戏
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


# ================= 核心逻辑 =================
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


@st.cache_resource
def init_knowledge_base():
    loader = PyPDFDirectoryLoader("data")
    docs = loader.load()

    if not docs:
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="shibing624/text2vec-base-chinese")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        st.error("⚠️ 未读取到 DEEPSEEK_API_KEY，请检查 .env 文件！")
        st.stop()

    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=api_key,
        base_url="https://api.deepseek.com",
        temperature=0
    )

    # 【升级】使用 from_messages 组合 System 提示、历史记录占位符和当前问题
    system_template = """你是一个专业的文献助手。请使用以下检索到的背景信息来回答用户的问题。
    如果你不知道答案，请直接说不知道，不要编造。

    背景信息：
    {context}
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        MessagesPlaceholder(variable_name="chat_history"),  # 动态插入历史记录
        ("human", "{question}")
    ])

    # 【升级】LCEL 字典映射，将输入的 question 和 history 分配给组件
    rag_chain = (
            {
                "context": lambda x: format_docs(retriever.invoke(x["question"])),
                "question": lambda x: x["question"],
                "chat_history": lambda x: x["chat_history"]
            }
            | prompt
            | llm
            | StrOutputParser()
    )
    return rag_chain


rag_chain = init_knowledge_base()

# ================= 主界面：微信式聊天交互 =================
if rag_chain is None:
    st.info("👈 请先在左侧边栏上传 PDF 文献，并点击【保存并更新知识库】")
else:
    # 1. 展示历史聊天记录（让你能像看微信一样看到之前的对话）
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 2. 底部浮动的聊天输入框
    if question := st.chat_input("请根据文献内容提问（支持上下文追问）..."):
        # 将用户问题显示在页面上并存入记忆
        with st.chat_message("user"):
            st.markdown(question)
        st.session_state.messages.append({"role": "user", "content": question})

        # 转换 Streamlit 的记忆格式为 LangChain 需要的格式
        chat_history = []
        for msg in st.session_state.messages[:-1]:  # 不包含刚发送的这句
            if msg["role"] == "user":
                chat_history.append(HumanMessage(content=msg["content"]))
            else:
                chat_history.append(AIMessage(content=msg["content"]))

        # AI 思考并回答
        with st.chat_message("assistant"):
            with st.spinner("DeepSeek 正在翻阅文献思考中..."):
                try:
                    # 将问题和历史记录一起喂给大模型
                    response = rag_chain.invoke({
                        "question": question,
                        "chat_history": chat_history
                    })
                    st.markdown(response)
                    # 将 AI 的回答也存入记忆
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"出错了：{e}")