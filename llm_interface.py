"""
Unified LLM interface.

Each provider is called through its own native API rather than being forced
behind an OpenAI-compatible shim.
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
    """Abstract base class for the provider interfaces."""
    
    @abstractmethod
    def generate(self, prompt, **kwargs):
        """Generate a text completion."""
        pass


# def _is_rate_limit_error(exception: Exception) -> bool:
#     """Return True for throttling/rate-limit errors (boto3 and the OpenAI SDK)."""
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

#     # Generic check for the OpenAI and other SDKs
#     status_code = getattr(exception, "status_code", None)
#     if status_code == 429:
#         return True
#     message = str(exception).lower()
#     for keyword in ("rate limit", "too many requests", "throttl", "exceeded quota"):
#         if keyword in message:
#             return True
#     return False


def _sleep_with_exponential_backoff(attempt_index: int, base_seconds: float = 0.5, cap_seconds: float = 5.0, jitter: bool = True) -> None:
    """Sleep with jittered exponential backoff. attempt_index starts at 0."""
    delay = min(cap_seconds, base_seconds * (2 ** attempt_index))
    if jitter:
        delay *= (0.5 + random.random())  # jitter in [0.5x, 1.5x]
    time.sleep(delay)

class LlamaInterface(LLMInterface):

    def __init__(self):
        self.model_id = "meta.llama3-1-70b-instruct-v1:0"
        self.client = boto3.client("bedrock-runtime", region_name="us-west-2")

    def build_prompt(self, query):
        # `query` is the raw user turn
        return f"""
<|begin_of_text|>
<|start_header_id|>user<|end_header_id|>
{query}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""

    def generate(self, prompt, temperature=0.7, max_tokens=512, top_p=0.9, max_retries=5, **kwargs):
        # `prompt` is the unformatted user turn; the chat template is applied here
        formatted_prompt = self.build_prompt(prompt)
        native_request = {
            "prompt": formatted_prompt,
            "temperature": temperature,
            "max_gen_len": max_tokens,
            "top_p": top_p,
        }
        request = json.dumps(native_request)

        for attempt_index in range(max_retries + 1):
            try:
                response = self.client.invoke_model(modelId=self.model_id, body=request)
                response_body = json.loads(response["body"].read())
                return response_body["generation"]
            except Exception as e:
                if attempt_index < max_retries:
                    _sleep_with_exponential_backoff(attempt_index)
                    continue
                raise RuntimeError(
                    f"Llama request failed after {max_retries + 1} attempts: {e}"
                ) from e

    def chat_generate(self, messages, temperature=0.7, max_tokens=512, top_p=0.9, **kwargs):
        max_chars = max_tokens * 4  # rough character budget standing in for the token limit
        system_prompt = ""
        last_user_msg = None
        # 1. Locate the system prompt and the most recent user turn
        for m in messages:
            if m["role"] == "system":
                system_prompt = m["content"]
        for m in reversed(messages):
            if m["role"] == "user":
                last_user_msg = m["content"]
                break
        # 2. Collect the intervening history (dropping the system prompt and the
        #    final user turn), most recent first
        history_msgs = []
        found_last_user = False
        for m in reversed(messages):
            if m["role"] == "user" and not found_last_user:
                found_last_user = True
                continue
            if m["role"] != "system":
                history_msgs.append(m)
        # 3. Walk backwards from the most recent message until max_chars is reached
        history_parts = []
        total_chars = len(system_prompt) + len(last_user_msg or "")
        for m in history_msgs:
            msg_str = f"\n[{m['role']}] {m['content']}"
            if total_chars + len(msg_str) > max_chars:
                break  # stop once the budget is exhausted
            history_parts.insert(0, msg_str)  # prepend, restoring chronological order
            total_chars += len(msg_str)
        # 4. Assemble the final prompt
        prompt_parts = [system_prompt] + history_parts
        if last_user_msg:
            prompt_parts.append(f"\n[last_user] {last_user_msg}")
        full_prompt = "".join(prompt_parts)
        return self.generate(full_prompt, temperature, max_tokens, top_p, **kwargs)

class ClaudeInterface(LLMInterface):
    """Claude interface, calling Bedrock through boto3."""
    
    def __init__(self):
        self.model_id = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
        self.client = boto3.client("bedrock-runtime", region_name="us-west-2")

    def generate(self, prompt, temperature=0.7, max_tokens=512, top_p=0.9, max_retries=5, **kwargs):
        """Generate a response with the Claude Messages API."""
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

        for attempt_index in range(max_retries + 1):
            try:
                response = self.client.invoke_model_with_response_stream(modelId=self.model_id, body=request)
                
                # Accumulate the streamed response
                full_response = ""
                for event in response["body"]:
                    chunk = json.loads(event["chunk"]["bytes"])
                    if chunk.get("type") == "content_block_delta":
                        text_delta = chunk["delta"].get("text", "")
                        full_response += text_delta
                
                return full_response
            except Exception as e:
                if attempt_index < max_retries:
                    _sleep_with_exponential_backoff(attempt_index)
                    continue
                raise RuntimeError(
                    f"Claude API error after {max_retries + 1} attempts: {e}"
                ) from e

    def chat_generate(self, messages, temperature=0.7, max_tokens=512, top_p=0.9, **kwargs):
        """Chat-mode generation, flattening messages exactly as LlamaInterface does."""
        max_chars = max_tokens * 4  # rough character budget standing in for the token limit
        system_prompt = ""
        last_user_msg = None
        # 1. Locate the system prompt and the most recent user turn
        for m in messages:
            if m["role"] == "system":
                system_prompt = m["content"]
        for m in reversed(messages):
            if m["role"] == "user":
                last_user_msg = m["content"]
                break
        # 2. Collect the intervening history (dropping the system prompt and the
        #    final user turn), most recent first
        history_msgs = []
        found_last_user = False
        for m in reversed(messages):
            if m["role"] == "user" and not found_last_user:
                found_last_user = True
                continue
            if m["role"] != "system":
                history_msgs.append(m)
        # 3. Walk backwards from the most recent message until max_chars is reached
        history_parts = []
        total_chars = len(system_prompt) + len(last_user_msg or "")
        for m in history_msgs:
            msg_str = f"\n[{m['role']}] {m['content']}"
            if total_chars + len(msg_str) > max_chars:
                break  # stop once the budget is exhausted
            history_parts.insert(0, msg_str)  # prepend, restoring chronological order
            total_chars += len(msg_str)
        # 4. Assemble the final prompt
        prompt_parts = [system_prompt] + history_parts
        if last_user_msg:
            prompt_parts.append(f"\n[last_user] {last_user_msg}")
        full_prompt = "".join(prompt_parts)
        return self.generate(full_prompt, temperature, max_tokens, top_p, **kwargs)

class DeepSeekInterface(LLMInterface):
    """DeepSeek interface, calling Bedrock through boto3."""
    
    def __init__(self):
        self.model_id = "deepseek.v3-v1:0"
        self.client = boto3.client("bedrock-runtime", region_name="us-west-2")

    def generate(self, prompt, temperature=0.7, max_tokens=512, top_p=0.9, max_retries=5, **kwargs):
        """Generate a response with the DeepSeek Converse API."""
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
                if attempt_index < max_retries:
                    _sleep_with_exponential_backoff(attempt_index)
                    continue
                raise RuntimeError(
                    f"DeepSeek API error after {max_retries + 1} attempts: {e}"
                ) from e

    def chat_generate(self, messages, temperature=0.7, max_tokens=512, top_p=0.9, **kwargs):
        """Chat-mode generation, flattening messages exactly as LlamaInterface does."""
        max_chars = max_tokens * 4  # rough character budget standing in for the token limit
        system_prompt = ""
        last_user_msg = None
        # 1. Locate the system prompt and the most recent user turn
        for m in messages:
            if m["role"] == "system":
                system_prompt = m["content"]
        for m in reversed(messages):
            if m["role"] == "user":
                last_user_msg = m["content"]
                break
        # 2. Collect the intervening history (dropping the system prompt and the
        #    final user turn), most recent first
        history_msgs = []
        found_last_user = False
        for m in reversed(messages):
            if m["role"] == "user" and not found_last_user:
                found_last_user = True
                continue
            if m["role"] != "system":
                history_msgs.append(m)
        # 3. Walk backwards from the most recent message until max_chars is reached
        history_parts = []
        total_chars = len(system_prompt) + len(last_user_msg or "")
        for m in history_msgs:
            msg_str = f"\n[{m['role']}] {m['content']}"
            if total_chars + len(msg_str) > max_chars:
                break  # stop once the budget is exhausted
            history_parts.insert(0, msg_str)  # prepend, restoring chronological order
            total_chars += len(msg_str)
        # 4. Assemble the final prompt
        prompt_parts = [system_prompt] + history_parts
        if last_user_msg:
            prompt_parts.append(f"\n[last_user] {last_user_msg}")
        full_prompt = "".join(prompt_parts)
        return self.generate(full_prompt, temperature, max_tokens, top_p, **kwargs)

class OpenAIInterface(LLMInterface):
    """OpenAI model interface, using the native OpenAI client.
    
    The API key is resolved in this order:
    1. Constructor argument: OpenAIInterface(model_id="gpt-4o-mini", api_key="sk-...")
    2. Environment variable: export OPENAI_API_KEY="sk-..."
    
    Exporting the environment variable before launching a run is recommended:
        export OPENAI_API_KEY="sk-your-api-key-here"
    """
    
    def __init__(self, model_id, api_key=None):
        self.model_id = model_id  # e.g. "gpt-4o-mini", "gpt-4o", "gpt-5-nano"
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
        """Generate a response with the native OpenAI client."""
        for attempt_index in range(max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_id,  # model id chosen at construction time
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                )
                return response.choices[0].message.content
            except Exception as e:
                if attempt_index < max_retries:
                    _sleep_with_exponential_backoff(attempt_index)
                    continue
                raise RuntimeError(
                    f"OpenAI API error after {max_retries + 1} attempts: {e}"
                ) from e
    
    def chat_generate(self, messages, temperature=0.7, max_tokens=512, max_retries=5, **kwargs):
        """Chat-mode generation."""
        for attempt_index in range(max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_id,  # model id chosen at construction time
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                )
                return response.choices[0].message.content
            except Exception as e:
                if attempt_index < max_retries:
                    _sleep_with_exponential_backoff(attempt_index)
                    continue
                raise RuntimeError(
                    f"OpenAI API error after {max_retries + 1} attempts: {e}"
                ) from e

class LLMFactory:
    """Factory mapping a model name onto the matching provider interface."""
    
    # Supported model names and the ids they resolve to
    SUPPORTED_MODELS = {
        # Llama
        # LlamaInterface hardcodes this one Bedrock model id and ignores the
        # mapping below, so only the 70B variant may be listed here: any other
        # Llama name would be accepted and then silently served by llama3.1-70b.
        "llama3.1-70b": "meta.llama3-1-70b-instruct-v1:0",
        
        # Claude
        "claude-3.7-sonnet": "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        
        # DeepSeek
        "deepseek-v3.1": "deepseek.v3-v1:0",
        
        # OpenAI
        "gpt-4o-mini": "gpt-4o-mini",
        "gpt-4o": "gpt-4o",
        "gpt-4": "gpt-4",
        "gpt-3.5-turbo": "gpt-3.5-turbo",
        "gpt-5.2": "gpt-5.2",
        "gpt-5-nano": "gpt-5-nano",
    }
    
    @staticmethod
    def is_llama_model(model_name):
        """True if `model_name` is a supported Llama model."""
        return model_name in LLMFactory.SUPPORTED_MODELS and "llama" in model_name.lower()
    
    @staticmethod
    def is_claude_model(model_name):
        """True if `model_name` is a supported Claude model."""
        return model_name in LLMFactory.SUPPORTED_MODELS and "claude" in model_name.lower()
    
    @staticmethod
    def is_deepseek_model(model_name):
        """True if `model_name` is a supported DeepSeek model."""
        return model_name in LLMFactory.SUPPORTED_MODELS and "deepseek" in model_name.lower()
    
    @staticmethod
    def is_openai_model(model_name):
        """True if `model_name` is a supported OpenAI model."""
        return model_name in LLMFactory.SUPPORTED_MODELS and "gpt" in model_name.lower()
    
    @staticmethod
    def create_llm(model_name, **kwargs):
        """Instantiate the interface matching `model_name`."""
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