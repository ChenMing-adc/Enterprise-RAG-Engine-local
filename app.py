import os
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

from config import DATA_DIR
from retrieval import build_advanced_retriever, add_pdf_to_db, delete_pdf_from_db  # 【引入 CRUD 函数】
from generation import build_rag_chain

st.set_page_config(page_title="私有文献知识库", page_icon="📚", layout="wide")
st.title("📚 私有文献知识库 (增量更新数据库版)")

# ================= 状态与初始化 =================
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
if "messages" not in st.session_state:
    st.session_state.messages = []


@st.cache_resource
def init_system():
    retriever = build_advanced_retriever()
    chain = build_rag_chain()
    return retriever, chain


retriever, rag_chain = init_system()

# ================= 侧边栏：上传与删除联动 =================
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
                file_path = os.path.join(DATA_DIR, uploaded_file.name)
                # 写入本地 data 文件夹
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # 【新增】：通知数据库解析并存入这个新文件
                with st.spinner(f"正在解析并写入数据库: {uploaded_file.name}"):
                    add_pdf_to_db(file_path)

            st.cache_resource.clear()  # 刷新检索器缓存（为了让 BM25 读取最新数据库）
            st.session_state.uploader_key += 1
            st.session_state.messages = []
            st.rerun()
        else:
            st.warning("请先上传文件！")

    st.markdown("---")
    st.markdown("### 当前已有文献：")
    existing_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.pdf')]
    if existing_files:
        for f in existing_files:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.text(f"📄 {f}")
            with col2:
                if st.button("❌", key=f"del_{f}", help=f"删除 {f}"):
                    file_path = os.path.join(DATA_DIR, f)
                    if os.path.exists(file_path):
                        os.remove(file_path)

                    # 【新增】：通知数据库擦除这篇文献的所有向量记忆
                    with st.spinner(f"正在从数据库擦除: {f}"):
                        delete_pdf_from_db(f)

                    st.cache_resource.clear()
                    st.session_state.messages = []
                    st.rerun()
    else:
        st.text("暂无文献，请上传")

# ================= 主聊天 UI =================
if retriever is None:
    st.info("👈 请先在左侧边栏上传 PDF 文献，并点击【保存并更新知识库】")
else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "sources" in msg and msg["sources"]:
                with st.expander("📚 查看参考来源 (RRF混合检索 + 精排 Top-3)"):
                    for i, source in enumerate(msg["sources"]):
                        score_text = f" (精排得分: {source['score']:.4f})" if 'score' in source else ""
                        st.markdown(f"**[{i + 1}] {source['file']} (第 {source['page']} 页){score_text}**")
                        st.caption(source['content'])

    if question := st.chat_input("请根据文献内容提问..."):
        with st.chat_message("user"):
            st.markdown(question)
        st.session_state.messages.append({"role": "user", "content": question})

        chat_history = []
        history_msgs = st.session_state.messages[:-1]

        MAX_HISTORY_LENGTH = 4
        if len(history_msgs) > MAX_HISTORY_LENGTH:
            history_msgs = history_msgs[-MAX_HISTORY_LENGTH:]

        for msg in history_msgs:
            if msg["role"] == "user":
                chat_history.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                clean_content = msg["content"].split("📚 查看参考来源")[0].strip()
                chat_history.append(AIMessage(content=clean_content))

        with st.chat_message("assistant"):
            with st.spinner("🔍 正在执行双路召回、RRF融合与交叉重排..."):
                retrieved_docs = retriever.invoke(question)

            context_text = ""
            source_data = []
            for i, doc in enumerate(retrieved_docs):
                file_name = os.path.basename(doc.metadata.get('source', '未知文档'))
                page_num = doc.metadata.get('page', 0) + 1
                score = doc.metadata.get('relevance_score', 0)

                context_text += f"\n[文档 {i + 1}] {file_name} (第{page_num}页):\n{doc.page_content}\n"
                source_data.append({
                    "file": file_name,
                    "page": page_num,
                    "score": score,
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
                with st.expander("📚 查看参考来源 (RRF混合检索 + 精排 Top-3)"):
                    for i, source in enumerate(source_data):
                        score_text = f" (精排得分: {source['score']:.4f})" if 'score' in source else ""
                        st.markdown(f"**[{i + 1}] {source['file']} (第 {source['page']} 页){score_text}**")
                        st.caption(source['content'])

            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "sources": source_data
            })