import os
import streamlit as st
from dotenv import load_dotenv

# 文档处理组件
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 模型与全新 LCEL 组件 (完全不需要 langchain.chains)
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. 加载环境变量
load_dotenv()

st.set_page_config(page_title="私有文献知识库", page_icon="📚")
st.title("📚 私有文献知识库 (纯 LCEL 版)")


# 将多个检索到的文档拼接成一个长文本
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

    # 向量化与存储
    embeddings = HuggingFaceEmbeddings(model_name="shibing624/text2vec-base-chinese")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        # 如果没读到 Key，直接在网页上报错拦截
        st.error("⚠️ 未读取到 DEEPSEEK_API_KEY，请检查 .env 文件！")
        st.stop()

    # 大模型
    llm = ChatOpenAI(
        model="deepseek-v4-pro",
        api_key=api_key,  # 确保这里传进去的是纯字符串
        base_url="https://api.deepseek.com",
        temperature=0
    )

    # 构建 Prompt
    template = """你是一个专业的文献助手。请使用以下检索到的背景信息来回答用户的问题。
    如果你不知道答案，请直接说不知道，不要编造。

    背景信息：
    {context}

    用户问题：{question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    # 【最现代的 LCEL 写法】像流水线一样把组件串起来，告别传统的 chains
    rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )

    return rag_chain


rag_chain = init_knowledge_base()

if rag_chain is None:
    st.warning("⚠️ 没找到文献！请在 `data` 文件夹中放入至少一个 PDF 文件，然后刷新页面！")
else:
    st.success("✅ 知识库加载成功！")

    question = st.text_input("请根据文献内容提问：")
    if st.button("发送") and question:
        with st.spinner("DeepSeek 正在翻阅文献思考中..."):
            try:
                # 直接传字符串进去就行
                response = rag_chain.invoke(question)
                st.markdown("### 🤖 DeepSeek 回答:")
                st.info(response)
            except Exception as e:
                st.error(f"出错了：{e}")