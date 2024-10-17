import json

from flask import Flask, request, jsonify
from langchain_community.llms import Ollama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient

model = Ollama(model="qwen2.5:7b")
client = MilvusClient(uri="http://192.168.200.130:19530")

def get_rag(text):
    embedding_model = SentenceTransformer('maidalun1020/bce-embedding-base_v1')
    doc_vector = embedding_model.encode(text)
    search_params = {
        "metric_type": "COSINE",
        "params": {}
    }

    res = client.search(
        collection_name="cumt_gpt",
        data=[doc_vector],
        limit=3,
        output_fields=["id", "url", "content", "publish_time"],
        search_params=search_params
    )
    return res[0]

rags = get_rag("孙杨什么时候来矿大")

contents = []
urls = []

for rag in rags:
    entity = rag.get("entity")
    contents.append(entity.get("content"))
    urls.append(entity.get("url"))

print(contents)
print(urls)