import os
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

from config import DATA_DIR
from retrieval import build_advanced_retriever, add_pdf_to_db, delete_pdf_from_db
from generation import build_rag_chain, build_query_rewriter

st.set_page_config(page_title="私有文献知识库", page_icon="📚", layout="wide")
st.title("📚 私有文献知识库 (Multi-Query 架构版)")

# ================= 状态与初始化 =================
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
if "messages" not in st.session_state:
    st.session_state.messages = []


@st.cache_resource
def init_system():
    retriever = build_advanced_retriever()
    chain = build_rag_chain()
    rewriter = build_query_rewriter()
    return retriever, chain, rewriter


retriever, rag_chain, query_rewriter = init_system()

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

                # 通知数据库解析并存入这个新文件
                with st.spinner(f"正在解析并写入数据库: {uploaded_file.name}"):
                    add_pdf_to_db(file_path)

            st.cache_resource.clear()
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

                    # 通知数据库擦除这篇文献的所有向量记忆
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
    # 渲染历史记录
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "sources" in msg and msg["sources"]:
                with st.expander("📚 查看参考来源 (Multi-Query + 混合检索 + 精排)"):
                    for i, source in enumerate(msg["sources"]):
                        score_text = f" (精排得分: {source['score']:.4f})" if 'score' in source else ""
                        st.markdown(f"**[{i + 1}] {source['file']} (第 {source['page']} 页){score_text}**")
                        st.caption(source['content'])

    if question := st.chat_input("请根据文献内容提问（支持上下文追问）..."):
        # 1. 存入并展示用户问题
        with st.chat_message("user"):
            st.markdown(question)
        st.session_state.messages.append({"role": "user", "content": question})

        # 组装滑动窗口历史记录
        chat_history = []
        text_history = ""  # 用于提供���重写器的纯文本历史

        history_msgs = st.session_state.messages[:-1]
        MAX_HISTORY_LENGTH = 4  # 只保留最近 2 轮对话
        if len(history_msgs) > MAX_HISTORY_LENGTH:
            history_msgs = history_msgs[-MAX_HISTORY_LENGTH:]

        for msg in history_msgs:
            if msg["role"] == "user":
                chat_history.append(HumanMessage(content=msg["content"]))
                text_history += f"用户: {msg['content']}\n"
            elif msg["role"] == "assistant":
                # 去除来源小尾巴，保持历史纯净
                clean_content = msg["content"].split("📚 查看参考来源")[0].strip()
                chat_history.append(AIMessage(content=clean_content))
                text_history += f"助手: {clean_content}\n"

        with st.chat_message("assistant"):
            # 【第一阶段：Multi-Query 上下文重写与扩展】
            with st.spinner("🧠 正在理解上下文并进行检索词扩展..."):
                try:
                    expanded_queries = query_rewriter.invoke({
                        "chat_history": text_history if text_history else "无",
                        "question": question
                    })
                    # 确保原问题在列表中，且去重
                    if question not in expanded_queries:
                        expanded_queries.insert(0, question)
                except Exception as e:
                    expanded_queries = [question]  # 降级保护

            # 在 UI 展示大模型的重写结果
            with st.expander(f"🔍 触发 Multi-Query 多路扩展检索 ({len(expanded_queries)}路并发)"):
                for idx, q in enumerate(expanded_queries):
                    st.code(f"Query {idx + 1}: {q}", language="text")

            # 【第二阶段：混合检索与 RRF 精排】
            with st.spinner("⚡ 正在执行多路高并发融合检索与精排..."):
                retrieved_docs = retriever.invoke(expanded_queries)

            # 【第三阶段：拼装 Prompt 与流式生成】
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
                "question": question,  # 这里用原问题回答，防止模型跑题
                "chat_history": chat_history
            }):
                full_response += chunk
                response_placeholder.markdown(full_response + "▌")

            response_placeholder.markdown(full_response)

            if source_data:
                with st.expander("📚 查看参考来源 (Multi-Query + 混合检索 + 精排)"):
                    for i, source in enumerate(source_data):
                        score_text = f" (精排得分: {source['score']:.4f})" if 'score' in source else ""
                        st.markdown(f"**[{i + 1}] {source['file']} (第 {source['page']} 页){score_text}**")
                        st.caption(source['content'])

            # 完整数据存入记忆
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "sources": source_data
            })