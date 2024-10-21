from datetime import datetime
from typing import TypedDict, List
from zoneinfo import ZoneInfo

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor, LLMChainFilter, DocumentCompressorPipeline
from langchain_community.retrievers import MilvusRetriever
from langchain_community.vectorstores import Milvus
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms.ollama import Ollama
from langchain_milvus import MilvusCollectionHybridSearchRetriever
from pymilvus import WeightedRanker, Collection, connections, RRFRanker
from pymilvus.client.abstract import BaseRanker

# 使用 Chuxin-Embedding 作为嵌入模型
embedding_model = HuggingFaceEmbeddings(model_name="chuxin-llm/Chuxin-Embedding")
llm = Ollama(model="qwen2.5:7b")
connections.connect(
    alias="milvus",  # 可以给连接起个别名
    host="192.168.200.130",  # 这是 Milvus 服务器的 IP
    port="19530"  # Milvus 默认端口
)

# 配置 Milvus 向量数据库
retriever = MilvusCollectionHybridSearchRetriever(
    # connection_args={"uri": "http://192.168.200.130:19530"},  # Milvus 的地址
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

# vectorstore = MilvusRetriever(
#     collection_name="cumt_gpt_chuxin",
#     embedding_function=embedding_model,
#     collection_properties={
#         "fields": [
#             {"name": "id", "type": "INT64", "auto_id": True, "is_primary": True},  # 自动生成 ID，主键
#             {"name": "url", "type": "VARCHAR", "max_length": 65535},  # URL 字段
#             {"name": "content", "type": "VARCHAR", "max_length": 65535},  # 内容字段
#             {"name": "content_vector", "type": "FLOAT_VECTOR", "dim": 1024},  # 向量字段，维度为1024
#             {"name": "publish_time", "type": "INT64"}  # 时间戳字段
#         ]
#     },
#     connection_args={"host": "192.168.200.130", "port": "19530"},
#     consistency_level="Bounded",
#     search_params = {
#         "metric_type": "COSINE",
#         "params": {"ef": 150, "radius": 0.36},  # Search parameters
#     }
# )

# vectorstore = Milvus(
#     embedding_function=embedding_model,
#     collection_name="cumt_gpt_chuxin",
#     collection_description="CUMT_GPT的Chuxin数据集",
#     collection_properties={
#         "fields": [
#             {"name": "id", "type": "INT64", "auto_id": True, "is_primary": True},  # 自动生成 ID，主键
#             {"name": "url", "type": "VARCHAR", "max_length": 65535},  # URL 字段
#             {"name": "content", "type": "VARCHAR", "max_length": 65535},  # 内容字段
#             {"name": "content_vector", "type": "FLOAT_VECTOR", "dim": 1024},  # 向量字段，维度为1024
#             {"name": "publish_time", "type": "INT64"}  # 时间戳字段
#         ]
#     },
#     connection_args={"uri": "http://192.168.200.130:19530"},
#     consistency_level="Bounded",
#     index_params={},
#     search_params={
#         "metric_type": "COSINE",
#         "params": {"ef": 150, "radius": 0.36},  # Search parameters
#     },
#     drop_old=True,
#     auto_id=True,
#     primary_field="pk",
#     text_field="content",
#     vector_field="content_vector",
#     # metadata_field=["id", "url", "content", "publish_time"],
#     # partition_key_field: Optional[str] = None,
#     # partition_names: Optional[list] = None,
#     # replica_number: int = 1,
#     timeout=30,
#     # num_shards: Optional[int] = None,
# )


# 使用 LangChain 的检索器
# retriever = vectorstore.as_retriever(
#     search_type="similarity",
#     search_kwargs={'k': 3},
# )
# response = retriever.invoke(input="孙杨什么时候来矿大")


# 输出文档
def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )

# 文本压缩器
# compressor = LLMChainExtractor.from_llm(llm)
# compression_retriever = ContextualCompressionRetriever(
#     base_compressor=compressor, base_retriever=retriever
# )
# compressed_docs = compression_retriever.invoke(
#     "什么时候举行庆祝矿大115周年篮球友谊赛?"
# )
# pretty_print_docs(compressed_docs)

#文本过滤器
_filter = LLMChainFilter.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=_filter, base_retriever=retriever
)

# compressed_docs = compression_retriever.invoke(
#     "什么时候举行庆祝矿大115周年篮球友谊赛?"
# )
# pretty_print_docs(compressed_docs)



# 可以结合 RetrievalQA 使用      map_reduce
# qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=compression_retriever)

# 进行检索并获取结果
# response = qa_chain.invoke(input="什么时候举行庆祝矿大115周年篮球友谊赛?")
# print(response)



### Contextualize question ###
contextualize_q_system_prompt = (
    "根据聊天记录和最新用户问题，"
    "可能引用了聊天记录中的上下文，"
    "重组一个独立的问题，"
    "使其在没有聊天记录的情况下也能被理解。"
    "不要回答问题，只需在必要时重新表述它，否则原样返回问题。"
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, compression_retriever, contextualize_q_prompt
)

beijing_time = datetime.now(ZoneInfo("Asia/Shanghai"))
time_prompt = "现在是北京时间：{}".format(beijing_time.strftime('%Y-%m-%d %H:%M:%S'))
### Answer question ###
system_prompt = (
    time_prompt +
    "你是一个负责问答任务的助手。"
    "使用以下检索到的上下文来回答问题。"
    "如果你不知道答案，请说明你不知道。"
    "保持回答简洁。"
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

# 定义状态，包含用户输入、历史聊天记录、上下文和模型答案
class State(TypedDict):
    input: str
    chat_history: List[BaseMessage]
    context: str
    answer: str

# 定义调用模型的方法，用于处理输入和历史聊天记录
def call_model(state: State):
    response = rag_chain.invoke(state)  # 调用模型获取上下文和答案
    return {
        "chat_history": state["chat_history"] + [
            HumanMessage(state["input"]),
            AIMessage(response["answer"]),
        ],
        "context": response["context"],
        "answer": response["answer"],
    }

# 创建一个包含历史消息和输入问题的状态
initial_state: State = {
    "input": "什么时候举行庆祝矿大115周年篮球友谊赛?",  # 用户输入的最新问题
    "chat_history": [
        # HumanMessage("之前的用户问题1"),
        # AIMessage("之前的AI回复1"),
        # HumanMessage("之前的用户问题2"),
        # AIMessage("之前的AI回复2"),
    ],  # 手动提供的历史消息
    "context": "",
    "answer": ""
}

# 使用模型调用函数
result = call_model(initial_state)

# 输出模型的最新答案和完整的聊天记录
print("模型答案:", result["answer"])
print("完整聊天记录:", result["chat_history"])