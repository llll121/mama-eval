"""
统一的LLM接口
支持不同模型的各自调用方式，不模拟OpenAI接口
"""

import json
import os
import time
import random
import boto3
from botocore.exceptions import ClientError
from openai import OpenAI
from abc import ABC, abstractmethod

class LLMInterface(ABC):
    """LLM接口抽象基类"""
    
    @abstractmethod
    def generate(self, prompt, **kwargs):
        """生成文本响应"""
        pass


# def _is_rate_limit_error(exception: Exception) -> bool:
#     """判断是否为限流/节流错误（适配boto3与OpenAI SDK）。"""
#     # boto3 ClientError
#     if isinstance(exception, ClientError):
#         try:
#             error_code = exception.response.get("Error", {}).get("Code")
#             http_status = exception.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
#             if http_status == 429:
#                 return True
#             if error_code in {"ThrottlingException", "TooManyRequestsException", "Throttling", "TooManyRequests"}:
#                 return True
#         except Exception:
#             pass

#     # OpenAI 或其它 SDK 的通用判断
#     status_code = getattr(exception, "status_code", None)
#     if status_code == 429:
#         return True
#     message = str(exception).lower()
#     for keyword in ("rate limit", "too many requests", "throttl", "exceeded quota"):
#         if keyword in message:
#             return True
#     return False


def _sleep_with_exponential_backoff(attempt_index: int, base_seconds: float = 0.5, cap_seconds: float = 5.0, jitter: bool = True) -> None:
    """按指数退避暂停一段时间，带抖动。attempt_index从0开始。"""
    delay = min(cap_seconds, base_seconds * (2 ** attempt_index))
    if jitter:
        delay *= (0.5 + random.random())  # [0.5x, 1.5x] 抖动
    time.sleep(delay)

class LlamaInterface(LLMInterface):

    def __init__(self):
        self.model_id = "meta.llama3-1-70b-instruct-v1:0"
        self.client = boto3.client("bedrock-runtime", region_name="us-west-2")

    def build_prompt(self, query):
        # query为用户输入内容
        return f"""
<|begin_of_text|>
<|start_header_id|>user<|end_header_id|>
{query}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""

    def generate(self, prompt, temperature=0.7, max_tokens=512, top_p=0.9, max_retries=5, **kwargs):
        # prompt为用户query内容（不含格式化），此处自动格式化
        formatted_prompt = self.build_prompt(prompt)
        native_request = {
            "prompt": formatted_prompt,
            "temperature": temperature,
            "max_gen_len": max_tokens,
            "top_p": top_p,
        }
        request = json.dumps(native_request)

        last_error = None
        for attempt_index in range(max_retries + 1):
            try:
                response = self.client.invoke_model(modelId=self.model_id, body=request)
                response_body = json.loads(response["body"].read())
                return response_body["generation"]
            except Exception as e:
                last_error = e
                if attempt_index < max_retries:
                    _sleep_with_exponential_backoff(attempt_index)
                    continue
                raise

    def chat_generate(self, messages, temperature=0.7, max_tokens=512, top_p=0.9, **kwargs):
        max_chars = max_tokens * 4  # 粗略估算token上限
        system_prompt = ""
        last_user_msg = None
        # 1. 找到system prompt和最后一条user
        for m in messages:
            if m["role"] == "system":
                system_prompt = m["content"]
        for m in reversed(messages):
            if m["role"] == "user":
                last_user_msg = m["content"]
                break
        # 2. 收集中间历史消息（去掉system和最后一条user），并倒序（最近的在前）
        history_msgs = []
        found_last_user = False
        for m in reversed(messages):
            if m["role"] == "user" and not found_last_user:
                found_last_user = True
                continue
            if m["role"] != "system":
                history_msgs.append(m)
        # 3. 从最近的历史消息开始往前拼接，直到长度接近max_chars
        history_parts = []
        total_chars = len(system_prompt) + len(last_user_msg or "")
        for m in history_msgs:
            msg_str = f"\n[{m['role']}] {m['content']}"
            if total_chars + len(msg_str) > max_chars:
                break  # 超长就不再加
            history_parts.insert(0, msg_str)  # 插到最前面，恢复原顺序
            total_chars += len(msg_str)
        # 4. 拼接最终prompt
        prompt_parts = [system_prompt] + history_parts
        if last_user_msg:
            prompt_parts.append(f"\n[last_user] {last_user_msg}")
        full_prompt = "".join(prompt_parts)
        return self.generate(full_prompt, temperature, max_tokens, top_p, **kwargs)

class ClaudeInterface(LLMInterface):
    """Claude模型接口 - 使用Boto3调用Bedrock"""
    
    def __init__(self):
        self.model_id = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
        self.client = boto3.client("bedrock-runtime", region_name="us-west-2")

    def generate(self, prompt, temperature=0.7, max_tokens=512, top_p=0.9, max_retries=5, **kwargs):
        """使用Claude Messages API生成响应"""
        native_request = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": prompt}]}
            ]
        }
        request = json.dumps(native_request)

        last_error = None
        for attempt_index in range(max_retries + 1):
            try:
                response = self.client.invoke_model_with_response_stream(modelId=self.model_id, body=request)
                
                # 收集流式响应
                full_response = ""
                for event in response["body"]:
                    chunk = json.loads(event["chunk"]["bytes"])
                    if chunk.get("type") == "content_block_delta":
                        text_delta = chunk["delta"].get("text", "")
                        full_response += text_delta
                
                return full_response
            except Exception as e:
                last_error = e
                if attempt_index < max_retries:
                    _sleep_with_exponential_backoff(attempt_index)
                    continue
                raise RuntimeError(f"Claude API error: {e}") from e

    def chat_generate(self, messages, temperature=0.7, max_tokens=512, top_p=0.9, **kwargs):
        """聊天模式生成响应 - 使用与Llama相同的消息处理逻辑"""
        max_chars = max_tokens * 4  # 粗略估算token上限
        system_prompt = ""
        last_user_msg = None
        # 1. 找到system prompt和最后一条user
        for m in messages:
            if m["role"] == "system":
                system_prompt = m["content"]
        for m in reversed(messages):
            if m["role"] == "user":
                last_user_msg = m["content"]
                break
        # 2. 收集中间历史消息（去掉system和最后一条user），并倒序（最近的在前）
        history_msgs = []
        found_last_user = False
        for m in reversed(messages):
            if m["role"] == "user" and not found_last_user:
                found_last_user = True
                continue
            if m["role"] != "system":
                history_msgs.append(m)
        # 3. 从最近的历史消息开始往前拼接，直到长度接近max_chars
        history_parts = []
        total_chars = len(system_prompt) + len(last_user_msg or "")
        for m in history_msgs:
            msg_str = f"\n[{m['role']}] {m['content']}"
            if total_chars + len(msg_str) > max_chars:
                break  # 超长就不再加
            history_parts.insert(0, msg_str)  # 插到最前面，恢复原顺序
            total_chars += len(msg_str)
        # 4. 拼接最终prompt
        prompt_parts = [system_prompt] + history_parts
        if last_user_msg:
            prompt_parts.append(f"\n[last_user] {last_user_msg}")
        full_prompt = "".join(prompt_parts)
        return self.generate(full_prompt, temperature, max_tokens, top_p, **kwargs)

class DeepSeekInterface(LLMInterface):
    """DeepSeek模型接口 - 使用Boto3调用Bedrock"""
    
    def __init__(self):
        self.model_id = "deepseek.v3-v1:0"
        self.client = boto3.client("bedrock-runtime", region_name="us-west-2")

    def generate(self, prompt, temperature=0.7, max_tokens=512, top_p=0.9, max_retries=5, **kwargs):
        """使用DeepSeek Converse API生成响应"""
        last_error = None
        for attempt_index in range(max_retries + 1):
            try:
                response = self.client.converse(
                    modelId=self.model_id,
                    messages=[
                        {"role": "user", "content": [{"text": prompt}]}
                    ],
                    inferenceConfig={
                        "maxTokens": max_tokens,
                        "temperature": temperature,
                        "topP": top_p
                    }
                )
                return response["output"]["message"]["content"][0]["text"]
            except Exception as e:
                last_error = e
                if attempt_index < max_retries:
                    _sleep_with_exponential_backoff(attempt_index)
                    continue
                raise RuntimeError(f"DeepSeek API error: {e}") from e

    def chat_generate(self, messages, temperature=0.7, max_tokens=512, top_p=0.9, **kwargs):
        """聊天模式生成响应 - 使用与Llama相同的消息处理逻辑"""
        max_chars = max_tokens * 4  # 粗略估算token上限
        system_prompt = ""
        last_user_msg = None
        # 1. 找到system prompt和最后一条user
        for m in messages:
            if m["role"] == "system":
                system_prompt = m["content"]
        for m in reversed(messages):
            if m["role"] == "user":
                last_user_msg = m["content"]
                break
        # 2. 收集中间历史消息（去掉system和最后一条user），并倒序（最近的在前）
        history_msgs = []
        found_last_user = False
        for m in reversed(messages):
            if m["role"] == "user" and not found_last_user:
                found_last_user = True
                continue
            if m["role"] != "system":
                history_msgs.append(m)
        # 3. 从最近的历史消息开始往前拼接，直到长度接近max_chars
        history_parts = []
        total_chars = len(system_prompt) + len(last_user_msg or "")
        for m in history_msgs:
            msg_str = f"\n[{m['role']}] {m['content']}"
            if total_chars + len(msg_str) > max_chars:
                break  # 超长就不再加
            history_parts.insert(0, msg_str)  # 插到最前面，恢复原顺序
            total_chars += len(msg_str)
        # 4. 拼接最终prompt
        prompt_parts = [system_prompt] + history_parts
        if last_user_msg:
            prompt_parts.append(f"\n[last_user] {last_user_msg}")
        full_prompt = "".join(prompt_parts)
        return self.generate(full_prompt, temperature, max_tokens, top_p, **kwargs)

class OpenAIInterface(LLMInterface):
    """OpenAI模型接口 - 使用原生OpenAI调用方式
    
    API Key 配置方式（优先级从高到低）：
    1. 通过参数传递：OpenAIInterface(model_id="gpt-4o-mini", api_key="sk-...")
    2. 通过环境变量：export OPENAI_API_KEY="sk-..."
    
    推荐在 SLURM 脚本中设置环境变量，例如：
        export OPENAI_API_KEY="sk-your-api-key-here"
    """
    
    def __init__(self, model_id, api_key=None):
        self.model_id = model_id  # OpenAI模型ID，如 "gpt-4o-mini", "gpt-4", "gpt-3.5-turbo"
        self.api_key = api_key or os.environ.get('OPENAI_API_KEY')
        
        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. Please set it via:\n"
                "  1. Environment variable: export OPENAI_API_KEY='sk-...'\n"
                "  2. Function parameter: OpenAIInterface(model_id='...', api_key='sk-...')\n"
                "  3. In SLURM script: add 'export OPENAI_API_KEY=\"sk-...\"' before running experiments"
            )
        
        self.client = OpenAI(api_key=self.api_key)
    
    def generate(self, prompt, temperature=0.7, max_tokens=512, max_retries=5, **kwargs):
        """使用OpenAI原生方式生成响应"""
        last_error = None
        for attempt_index in range(max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_id,  # 使用动态模型ID
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                )
                return response.choices[0].message.content
            except Exception as e:
                last_error = e
                if attempt_index < max_retries:
                    _sleep_with_exponential_backoff(attempt_index)
                    continue
                raise RuntimeError(f"OpenAI API error: {e}") from e
    
    def chat_generate(self, messages, temperature=0.7, max_tokens=512, max_retries=5, **kwargs):
        """聊天模式生成响应"""
        last_error = None
        for attempt_index in range(max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_id,  # 使用动态模型ID
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                )
                return response.choices[0].message.content
            except Exception as e:
                last_error = e
                if attempt_index < max_retries:
                    _sleep_with_exponential_backoff(attempt_index)
                    continue
                raise RuntimeError(f"OpenAI API error: {e}") from e

class LLMFactory:
    """LLM工厂类 - 根据模型类型创建对应的接口"""
    
    # 支持的模型映射
    SUPPORTED_MODELS = {
        # Llama模型
        "llama3-8b": "meta.llama3-1-8b-instruct-v1:0",
        "llama3-70b": "meta.llama3-1-70b-instruct-v1:0",
        "llama3.1-8b": "meta.llama3.1-8b-instruct-v1:0",
        "llama3.1-70b": "meta.llama3.1-70b-instruct-v1:0",
        "llama2-7b": "meta.llama2-7b-chat-v1:0",
        "llama2-13b": "meta.llama2-13b-chat-v1:0",
        "llama2-70b": "meta.llama2-70b-chat-v1:0",
        
        # Claude模型
        "claude-3.7-sonnet": "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        
        # DeepSeek模型
        "deepseek-v3.1": "deepseek.v3-v1:0",
        
        # OpenAI模型
        "gpt-4o-mini": "gpt-4o-mini",
        "gpt-4o": "gpt-4o",
        "gpt-4": "gpt-4",
        "gpt-3.5-turbo": "gpt-3.5-turbo",
        "gpt-5.2": "gpt-5.2",
        "gpt-5-nano": "gpt-5-nano",
    }
    
    @staticmethod
    def is_llama_model(model_name):
        """检查是否为Llama模型"""
        return model_name in LLMFactory.SUPPORTED_MODELS and "llama" in model_name.lower()
    
    @staticmethod
    def is_claude_model(model_name):
        """检查是否为Claude模型"""
        return model_name in LLMFactory.SUPPORTED_MODELS and "claude" in model_name.lower()
    
    @staticmethod
    def is_deepseek_model(model_name):
        """检查是否为DeepSeek模型"""
        return model_name in LLMFactory.SUPPORTED_MODELS and "deepseek" in model_name.lower()
    
    @staticmethod
    def is_openai_model(model_name):
        """检查是否为OpenAI模型"""
        return model_name in LLMFactory.SUPPORTED_MODELS and "gpt" in model_name.lower()
    
    @staticmethod
    def create_llm(model_name, **kwargs):
        """创建LLM接口实例"""
        if LLMFactory.is_llama_model(model_name):
            return LlamaInterface()
        elif LLMFactory.is_claude_model(model_name):
            return ClaudeInterface()
        elif LLMFactory.is_deepseek_model(model_name):
            return DeepSeekInterface()
        elif LLMFactory.is_openai_model(model_name):
            return OpenAIInterface(model_id=model_name, **kwargs)
        else:
            raise ValueError(f"Unsupported model: {model_name}") 