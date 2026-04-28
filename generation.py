import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

from config import *


def build_rag_chain():
    """构建防幻觉 Prompt 与大模型生成链路"""
    if not LLM_API_KEY:
        st.error("⚠️ 未读取到 DEEPSEEK_API_KEY，请检查 .env 文件！")
        st.stop()

    llm = ChatOpenAI(
        model=LLM_MODEL,
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        temperature=0,
        streaming=True
    )

    system_template = """你是一个严谨的私有文献知识助手。请严格遵守以下【核心指令】进行作答：

    【核心指令】
    1. 忠于原文：你必须且只能基于下方 <documents> 标签内的文档内容回答问题。
    2. 拒绝编造：如果 <documents> 中的信息不足以回答问题，请直接回答“抱歉，当前文献库中未提供相关信息”。
    3. 结构清晰：回答要有条理，并在适当的地方提及是参考了哪个文档。
    4. 公式渲染格式：如果回答中包含数学公式，请务必使用 Markdown 原生的 LaTeX 语法：
       - 行内公式必须使用单个美元符号包裹（例如：$a^2 + b^2 = c^2$）。
       - 独立公式块必须使用双美元符号包裹（例如：$$E=mc^2$$）。
       - 绝对禁止使用 \\( \\) 或 \\[ \\] 来包裹公式。
       - 绝对禁止将公式写在 ```latex 这样的代码块中。

    <documents>
    {context}
    </documents>
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])

    return prompt | llm | StrOutputParser()