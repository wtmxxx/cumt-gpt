import json
import time
from typing import TypedDict, List, Any, Dict, Iterator, Optional, AsyncIterator
from datetime import datetime
from zoneinfo import ZoneInfo

from flask import Flask, request, jsonify, Response
from langchain_community.document_transformers import LongContextReorder
from langchain_core.callbacks import CallbackManagerForRetrieverRun, AsyncCallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor, LLMChainFilter
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.llms.ollama import Ollama
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.runnables.utils import Input, Output
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import MilvusCollectionHybridSearchRetriever
from pymilvus import WeightedRanker, Collection, connections, RRFRanker
import re

# 初始化 Flask 应用程序
app = Flask(__name__)

# 使用 Chuxin-Embedding 作为嵌入模型
embedding_model = HuggingFaceEmbeddings(model_name="chuxin-llm/Chuxin-Embedding")
# 初始化 Ollama 模型
llm = Ollama(model="qwen2.5:7b")
connections.connect(
    alias="milvus",  # 可以给连接起个别名
    host="192.168.200.130",  # 这是 Milvus 服务器的 IP
    port="19530"  # Milvus 默认端口
)

# 配置 Milvus 向量数据库
retriever = MilvusCollectionHybridSearchRetriever(
    collection=Collection(name="cumt_gpt_chuxin", using="milvus"),
    text_field="content",
    output_fields=["id", "url", "content", "title", "publish_time"],
    anns_fields=["content_vector", "title_vector"],
    rerank=WeightedRanker(0.7, 0.3),
    field_embeddings=[embedding_model, embedding_model],
    field_search_params=[
        {"metric_type": "COSINE", "params": {"ef": 150, "radius": 0.35, "limit": 5}},
        {"metric_type": "COSINE", "params": {"ef": 150, "radius": 0.35, "limit": 5}}
    ],
    top_k=5
)


# 自定义 Retriever 继承 ContextualCompressionRetriever 返回排序后的文档
class CustomContextualCompressionRetriever(ContextualCompressionRetriever):
    def _get_relevant_documents(
            self,
            query: str,
            *,
            run_manager: CallbackManagerForRetrieverRun,
            **kwargs: Any,
    ) -> List[Document]:
        """Get documents relevant for a query.

        Args:
            query: string to find relevant documents for

        Returns:
            Sequence of relevant documents
        """
        docs = self.base_retriever.invoke(
            query, config={"callbacks": run_manager.get_child()}, **kwargs
        )
        if docs:
            compressed_docs = self.base_compressor.compress_documents(
                docs, query, callbacks=run_manager.get_child()
            )
            reordering = LongContextReorder()
            reordered_docs = reordering.transform_documents(compressed_docs)
            return list(reordered_docs)
        else:
            return []

    async def _aget_relevant_documents(
            self,
            query: str,
            *,
            run_manager: AsyncCallbackManagerForRetrieverRun,
            **kwargs: Any,
    ) -> List[Document]:
        """Get documents relevant for a query.

        Args:
            query: string to find relevant documents for

        Returns:
            List of relevant documents
        """
        docs = await self.base_retriever.ainvoke(
            query, config={"callbacks": run_manager.get_child()}, **kwargs
        )
        if docs:
            compressed_docs = await self.base_compressor.acompress_documents(
                docs, query, callbacks=run_manager.get_child()
            )
            reordering = LongContextReorder()
            reordered_docs = reordering.transform_documents(compressed_docs)
            return list(reordered_docs)
        else:
            return []


#文本过滤器
_filter = LLMChainFilter.from_llm(llm)
compression_retriever = CustomContextualCompressionRetriever(
    base_compressor=_filter, base_retriever=retriever
)

### 情境化问题 ###
contextualize_q_system_prompt = (
    "根据聊天记录和最新用户问题，"
    "可能引用了聊天记录中的上下文，"
    "重组一个独立的问题，"
    "使其在没有聊天记录的情况下也能被理解。"
    "不要回答问题，只需在必要时重新表述它，否则原样返回问题。"
    "如果问题中包含‘矿大’或‘中国矿业大学’字样，且其含义为地点（如表示某人或活动将在该地点发生），将其替换为‘学校’；如果不代表地点（如活动名称或特定事件），则保持原样。"
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history", n_messages=20),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, compression_retriever, contextualize_q_prompt
)

beijing_time = datetime.now(ZoneInfo("Asia/Shanghai"))
time_prompt = "现在是北京时间：{}。".format(beijing_time.strftime('%Y-%m-%d %H:%M:%S'))

### 回答问题 ###
system_prompt = (
        time_prompt +
        "你是中国矿业大学服务师生的智能助手。"
        "根据检索到的上下文回答问题。"
        "如果答案不明确或不在上下文中，请直接说明。"
        "回答要简明扼要。"
        "问题中提到的‘学校’默认指中国矿业大学，除非有特指。"
        "\n\n"
        "{context}"
)
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history", n_messages=20),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


# class CustomStreamRunnableChain(type(rag_chain)):
#     def invoke(self, input: Input, config: Optional[RunnableConfig] = None, **kwargs: Any) -> Output:
#         return rag_chain.invoke(input, config, **kwargs)
#
#     def stream(
#             self,
#             input: Input,
#             config: Optional[RunnableConfig] = None,
#             **kwargs: Optional[Any],
#     ) -> Iterator[Output]:
#         """
#         重载流式输出的stream方法。可以通过流式方式逐步输出。
#
#         Args:
#             input: 传递给Runnable的输入
#             config: 用于运行的配置
#             kwargs: 其他传递的参数
#
#         Returns:
#             Iterator: 逐步输出结果的生成器
#         """
#         # 调用检索链，并在检索或生成时逐步返回结果
#         yield self.invoke(input, config, **kwargs)
#
#     async def astream(
#             self,
#             input: Input,
#             config: Optional[RunnableConfig] = None,
#             **kwargs: Optional[Any],
#     ) -> AsyncIterator[Output]:
#         yield await self.ainvoke(input, config, **kwargs)

# stream_chain = CustomStreamRunnableChain()

# 定义状态，包含用户输入、历史聊天记录、上下文和模型答案
class State(TypedDict):
    input: str
    chat_history: List[BaseMessage]
    context: str
    answer: str
    citations: List[Dict[str, str]]  # 包含引文的URL和标题


# 定义调用模型的方法，用于处理输入和历史聊天记录
def call_model(state: State):
    # response = rag_chain.stream(state)  # 调用模型获取上下文和答案

    # 模型逐步生成数据，使用流式输出
    for chunk in rag_chain.stream(state):
        print(chunk, end="|", flush=True)
        json_chunk = json.dumps(
            {
                "content": chunk["answer"],
                "citations": [
                    {"url": doc.metadata['url'], "title": doc.metadata['title']}
                    for doc in chunk["context"] if 'url' in doc.metadata and 'title' in doc.metadata
                ]  # 获取所有文档的引文（URL和标题）
            }
        )

        yield f"{json_chunk}\n"
        # time.sleep(1)  # 这里可以根据实际模型输出的时间删除或调整


@app.route('/gpt/chat', methods=['POST'])
def gpt_response():
    try:
        # 获取请求数据
        data = request.get_json()
        messages = data.get('messages', [])

        # 生成RAG
        rag_urls = []
        history_messages = []
        for message in messages[:-1]:
            if message.get('role') == 'system':
                history_messages.append(SystemMessage(content=message.get('content')))
            elif message.get('role') == 'human':
                history_messages.append(HumanMessage(content=message.get('content')))
            elif message.get('role') == 'ai':
                history_messages.append(AIMessage(content=message.get('content')))
            else:
                history_messages.append(BaseMessage(content=message.get('content')))

        # 取出用户最后一个问题的 "content" 字段
        current_query = messages[-1].get("content")

        # print(history_messages)
        # print(current_query)

        # 创建一个包含历史消息和输入问题的状态
        initial_state: State = {
            "input": current_query,  # 用户输入的最新问题
            "chat_history": history_messages,  # 手动提供的历史消息
            "context": "",
            "answer": "",
            "citations": []
        }

        # 使用模型调用函数
        # result = call_model(initial_state)
        # print(result["answer"])
        # print(result["citations"])
        # return jsonify({'content': result["answer"], 'citations': result["citations"]}), 200

        return Response(call_model(initial_state), content_type='application/json')
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/gpt/get_title', methods=['POST'])
def get_title():
    try:
        data = request.get_json()
        messages = data.get('messages', [])

        # 拼接所有对话内容
        conversation = "\n".join([message.get('content') for message in messages])

        # 定义一个模板，生成标题
        prompt_template = f"""
            为下面的对话生成一个简洁的标题(只需标题内容，不要添加其他东西，只能包含中文 英文 ： : - — 空格,尽量用中文生成标题)：
            对话: {conversation}
        """

        # 最多尝试三次生成合规的标题
        title = ""
        for _ in range(3):
            title = llm.invoke(prompt_template)
            if re.fullmatch(r'[\u4e00-\u9fa5a-zA-Z\s：:-—]+', title):
                break

        # 如果标题仍然不合规，返回错误
        if not re.fullmatch(r'[\u4e00-\u9fa5a-zA-Z\s：:-—]+', title):
            return jsonify({'error': '标题生成失败，请重试'}), 500

        print("title：" + title)
        return jsonify({'title': title}), 200
        # return "title test", 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
