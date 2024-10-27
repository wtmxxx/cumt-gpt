import asyncio
import json
from datetime import datetime
from zoneinfo import ZoneInfo

import websockets
from langchain_community.llms.ollama import Ollama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from websockets.server import WebSocketServerProtocol

# 初始化 Ollama 模型
model = Ollama(model="qwen2.5:7b")

# 处理来自客户端的连接
async def gpt_response(websocket: WebSocketServerProtocol, path: str):
    try:
        async for message in websocket:
            print(message)
            data = json.loads(message)
            messages = data.get('messages', [])
            role_messages = []

            # 获取当前北京时间
            beijing_time = datetime.now(ZoneInfo("Asia/Shanghai"))
            role_messages.append(SystemMessage(content="现在是北京时间：{}".format(beijing_time.strftime('%Y-%m-%d %H:%M:%S'))))

            # 将接收到的消息转换成对应的role消息
            for msg in messages:
                if msg.get('role') == 'system':
                    role_messages.append(SystemMessage(content=msg.get('content')))
                elif msg.get('role') == 'human':
                    role_messages.append(HumanMessage(content=msg.get('content')))
                elif msg.get('role') == 'ai':
                    role_messages.append(AIMessage(content=msg.get('content')))
                else:
                    role_messages.append(BaseMessage(content=msg.get('content')))

            # 模型生成响应
            async def generate_stream():
                for chunk in model.stream(role_messages):
                    print(chunk, end="|", flush=True)
                    # 使用你想要的 JSON 格式返回
                    json_chunk = json.dumps({
                        'type': 'content',
                        'content': chunk
                    })
                    await websocket.send(json_chunk)
                    await asyncio.sleep(0.1)  # 模拟逐步发送

            # 调用生成函数，发送模型结果
            await generate_stream()

            # 当所有消息处理完后发送 citations
            await websocket.send(
                json.dumps({
                    'type': 'citations',
                    'citations': [
                        {'url': "https://www.wotemo.com", 'title': "Wotemo"},
                        {'url': "https://www.cumt.edu.cn", 'title': "CUMT"}
                    ]
                })
            )

    except websockets.exceptions.ConnectionClosedError as e:
        print(f"连接已关闭: {e}")
    except Exception as e:
        print(f"发生错误: {e}")

# 启动WebSocket服务器
async def main():
    async with websockets.serve(gpt_response, "localhost", 8765):
        print("WebSocket服务器已启动...")
        await asyncio.Future()  # 运行服务器直到手动停止

# 运行事件循环
if __name__ == "__main__":
    asyncio.run(main())
