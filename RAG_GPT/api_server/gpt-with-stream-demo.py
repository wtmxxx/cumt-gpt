import json
import time
from datetime import datetime
from zoneinfo import ZoneInfo

from flask import Flask, request, jsonify, Response
from langchain_community.llms import Ollama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from pymilvus import MilvusClient
import re

# 初始化 Flask 应用程序
app = Flask(__name__)

# 初始化 Ollama 模型
model = Ollama(model="qwen2.5:7b")
client = MilvusClient(uri="http://192.168.200.130:19530")

def get_rag(text):
    embedding_model = SentenceTransformer('maidalun1020/bce-embedding-base_v1')
    doc_vector = embedding_model.encode(text, batch_size=128)
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

    #HNSW
    res = client.search(
        collection_name="cumt_gpt",  # Collection name
        data=[doc_vector],  # Replace with your query vector
        search_params={
            "metric_type": "COSINE",
            "params": {"ef": 150, "radius": 0.5},  # Search parameters
        },  # Search parameters
        limit=10,  # Max. number of search results to return
        output_fields=["id", "url", "content", "publish_time"],  # Fields to return in the search results
        consistency_level="Bounded"
    )

    return res[0]

@app.route('/gpt/chat', methods=['POST'])
def gpt_response():
    try:
        # 获取请求数据
        data = request.get_json()
        messages = data.get('messages', [])

        role_messages = []

        beijing_time = datetime.now(ZoneInfo("Asia/Shanghai"))
        role_messages.append(SystemMessage(content="System提示为导入的RAG内容，是额外内容参考，每条System消息可能并没有关联，但与用户问题可能有关联，如果是根据System消息回答有关内容，请不要指出，另外，现在是北京时间：{}".format(beijing_time.strftime('%Y-%m-%d %H:%M:%S'))))

        # 生成RAG
        rags = get_rag(messages[-1].get('content'))
        rag_contents = []
        rag_urls = []
        for rag in rags:
            entity = rag.get("entity")
            rag_contents.append(entity.get("content"))
            rag_urls.append(entity.get("url"))
        for system_message in rag_contents:
            role_messages.append(SystemMessage(system_message))

        for message in messages:
            if message.get('role') == 'system':
                role_messages.append(SystemMessage(content=message.get('content')))
            elif message.get('role') == 'human':
                role_messages.append(HumanMessage(content=message.get('content')))
            elif message.get('role') == 'ai':
                role_messages.append(AIMessage(content=message.get('content')))
            else:
                role_messages.append(BaseMessage(content=message.get('content')))

        print(role_messages)

        def generate_stream():
            # 模型逐步生成数据，使用流式输出
            for chunk in model.stream(role_messages):  # 假设 role_messages 是空列表
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