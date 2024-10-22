import json
import time
from datetime import datetime
from zoneinfo import ZoneInfo

from flask import Flask, request, jsonify, Response
from langchain_community.document_transformers import LongContextReorder
from langchain_community.llms import Ollama
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.prompts import PromptTemplate
from sklearn.preprocessing import normalize
from FlagEmbedding import FlagModel
from pymilvus import MilvusClient
import re

# 初始化 Flask 应用程序
app = Flask(__name__)

# 初始化 Ollama 模型
model = Ollama(model="qwen2.5:7b")
client = MilvusClient(uri="http://192.168.200.130:19530")

def get_rag(texts: list):
    embedding_model = FlagModel('chuxin-llm/Chuxin-Embedding', use_fp16=True)
    doc_vector = embedding_model.encode_queries(texts, batch_size=256)
    # doc_vector = normalize(doc_vector, norm='l2')
    # search_params = {
    #     "metric_type": "COSINE",
    #     "params": {}
    # }

    # IVF_FLAT
    # res = client.search(
    #     collection_name="cumt_gpt",
    #     data=[doc_vector],
    #     limit=1,
    #     output_fields=["id", "url", "content", "publish_time"],
    #     search_params=search_params
    # )

    # HNSW
    res = client.search(
        collection_name="cumt_gpt_chuxin",  # Collection name
        data=doc_vector.tolist(),  # Replace with your query vector
        search_params={
            "metric_type": "COSINE",
            "params": {"ef": 150, "radius": 0.36},  # Search parameters
        },  # Search parameters
        limit=3,  # Max. number of search results to return
        output_fields=["id", "url", "content", "publish_time"],  # Fields to return in the search results
        consistency_level="Bounded"
    )

    reordering = LongContextReorder()
    reordered_docs = reordering.transform_documents(res[0])
    return reordered_docs

    # return res[0]
    # return res[0][::-1]

@app.route('/gpt/chat', methods=['POST'])
def gpt_response():
    try:
        # 获取请求数据
        data = request.get_json()
        messages = data.get('messages', [])

        # 生成RAG
        rags = get_rag([messages[-1].get('content')])
        rag_contents = []
        rag_urls = []
        for rag in rags:
            entity = rag.get("entity")
            rag_contents.append(entity.get("content"))
            rag_urls.append(entity.get("url"))
        beijing_time = datetime.now(ZoneInfo("Asia/Shanghai"))
        rag_contents.append("现在是北京时间：{}".format(beijing_time.strftime('%Y-%m-%d %H:%M:%S')))

        prompt_template = """
                以下是对话的历史内容：
                {history}

                根据以下内容回答用户的问题：
                内容：
                {documents}

                问题：{question}
                """
        prompt = PromptTemplate.from_template(prompt_template)

        # 拼接历史消息为文本，取出 "role" 和 "content" 并拼接
        history_text = "\n".join([msg.get("role") + ": " + msg.get("content") for msg in messages[:-1]])
        # 将 rag_contents 的内容拼接为文本
        documents_text = "\n".join([doc for doc in rag_contents])
        # 取出用户最后一个问题的 "content" 字段
        current_query = messages[-1].get("content")

        # print(history_text)
        print(documents_text)
        # print(current_query)

        chain = prompt | model

        def generate_stream():
            # 模型逐步生成数据，使用流式输出
            for chunk in chain.stream({
            "history": history_text,
            "documents": documents_text,
            "question": current_query
        }):  # 假设 role_messages 是空列表
                print(chunk, end="|", flush=True)
                json_chunk = json.dumps({'content': chunk})
                yield f"{json_chunk}\n"
                # time.sleep(1)  # 这里可以根据实际模型输出的时间删除或调整

        # 返回流式响应
        return Response(generate_stream(), content_type='application/json')
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
            title = model.invoke(prompt_template)
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