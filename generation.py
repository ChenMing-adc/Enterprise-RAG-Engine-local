import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

from config import *


def _get_llm(temperature=0.0):
    """提取一个公共的 LLM 初始化函数"""
    if not LLM_API_KEY:
        st.error("⚠️ 未读取到 DEEPSEEK_API_KEY，请检查 .env 文件！")
        st.stop()
    return ChatOpenAI(
        model=LLM_MODEL,
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        temperature=temperature,
        streaming=True
    )


def build_query_rewriter():
    """【新增】构建 Multi-Query 与上下文重写引擎"""
    # 这里给一点温度 (0.2)，让模型生成的扩展词稍微有些发散性
    llm = _get_llm(temperature=0.2)

    rewrite_template = """你是一个专业的学术知识库检索词优化专家。
    请根据用户的【历史对话上下文】和【最新问题】，将其重写并扩展为 3 个不同表述的独立检索词（Query），以提高向量数据库的召回率。

    【核心要求】：
    1. 补全代词：如果最新问题中使用了代词（如“它”、“这个”、“该方法”），必须根据历史对话替换为具体的专有名词。
    2. 多角度扩展：3个检索词应包含同义词、学术术语或从不同侧面（如原理、应用、优缺点）进行表述。
    3. 纯净输出：严格每行输出一个检索词。绝对不要输出任何序号（如 1. 2. 3.）、破折号、前缀或任何解释性废话。

    【历史对话】：
    {chat_history}

    【最新问题】：
    {question}
    """

    prompt = ChatPromptTemplate.from_template(rewrite_template)

    # 解析器：把大模型输出的多行文本切分成 Python 列表
    def parse_queries(text: str):
        queries = [q.strip() for q in text.strip().split('\n') if q.strip()]
        return queries[:3]  # 确保只返回最多 3 个

    return prompt | llm | StrOutputParser() | RunnableLambda(parse_queries)


def build_rag_chain():
    """构建防幻觉 Prompt 与大模型生成链路"""
    llm = _get_llm(temperature=0.0)

    system_template = """你是一个严谨的私有文献知识助手。请严格遵守以下【核心指令】进行作答：

    【核心指令】
    1. 忠于原文：你必须且只能基于下方 <documents> 标签内的文档内容回答问题。
    2. 拒绝编造：如果 <documents> 中的信息不足以回答问题，请直接回答“抱歉，当前文献库中未提供相关信息”。
    3. 结构清晰：回答要有条理，并在适当的地方提及是参考了哪个文档。
    4. 公式渲染格式：如果回答中包含数学公式，请务必使用 Markdown 原生的 LaTeX 语法（行内公式使用 $...$，独立公式使用 $$...$$）。绝对禁止使用 \\( \\) 或 ```latex。

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


from langchain_core.runnables import RunnableLambda  # 确保导入