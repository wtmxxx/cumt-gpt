from datetime import datetime
from zoneinfo import ZoneInfo

from flask import Flask, request, jsonify
from langchain_community.llms import Ollama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient
import re

model = Ollama(model="qwen2.5:7b")

for chunk in model.stream("你好啊，你是什么"):
    print(chunk, end="|", flush=True)