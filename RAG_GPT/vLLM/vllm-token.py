from vllm import LLM, SamplingParams

# 初始化并加载模型
llm = LLM(model="Qwen/Qwen-7B")

# 输入文本，确保是字符串或字符串列表
input_text = "请解释什么是图神经网络。"

# 设置采样参数
sampling_params = SamplingParams(max_tokens=128000, temperature=0.7, top_p=0.9)

# 使用模型生成文本
outputs = llm.generate(input_text, sampling_params=sampling_params)

# 打印生成的结果
print(outputs[0])
