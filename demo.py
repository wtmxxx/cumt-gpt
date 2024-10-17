from flask import Flask, request, jsonify
from langchain_community.llms import Ollama
import datetime

# 初始化 Flask 应用程序
app = Flask(__name__)

# 初始化 Ollama 模型
llm = Ollama(model="qwen2.5:7b")

@app.route('/gpt', methods=['POST'])
def gpt_response():
    try:
        # 获取请求数据
        data = request.get_json()
        messages = data.get('messages', [])

        # 将所有消息内容连接为单个字符串，作为模型输入
        prompt = "\n".join([msg['content'] for msg in messages])

        # 使用 Ollama 模型生成回复
        response_content = llm(prompt)

        print(response_content)

        # 返回结果字符串
        return response_content, 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)