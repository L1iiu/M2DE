import os
import requests
from openai import OpenAI
from dotenv import load_dotenv
import logging
import json
import time

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv('config/.env')

def get_model_call_func(model_name):
    """根据模型名称获取对应的调用函数"""
    if model_name.startswith("ollama:"):
        model_spec = model_name[7:]
        if ":" in model_spec and "://" in model_spec:
            parts = model_spec.split(":", 1)
            ollama_model = parts[0]
            ollama_base_url = parts[1]
        else:
            ollama_model = model_spec
            ollama_base_url = "http://localhost:11434"
        
        # 返回对应的Ollama调用函数
        if ollama_model == "llama2:7b":
            return lambda prompt: call_llama2_7b(prompt, ollama_base_url)
        elif ollama_model == "llama2:13b":
            return lambda prompt: call_llama2_13b(prompt, ollama_base_url)
        elif ollama_model == "llama3:8b":
            return lambda prompt: call_llama3_8b(prompt, ollama_base_url)
        elif ollama_model == "mistral:7b":
            return lambda prompt: call_mistral_7b(prompt, ollama_base_url)
        else:
            raise ValueError(f"不支持的Ollama模型: {ollama_model}")
    
    elif model_name == "qwen-turbo":
        return call_qwen_turbo
    elif model_name == "glm-4.5":
        return call_glm45
    elif model_name.startswith("gpt"):
        return lambda prompt: call_gpt(prompt, model=model_name)
    else:
        raise ValueError(f"不支持的模型: {model_name}")

def call_llama2_7b(prompt, base_url="http://localhost:11434"):
    """调用Llama2 7B模型"""
    try:
        if not base_url.startswith(('http://', 'https://')):
            base_url = f"http://{base_url}"
        
        url = f"{base_url}/api/generate"
        data = {
            "model": "llama2:7b",
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": 512
            }
        }
        
        print(f"调用Llama2 7B: URL={url}")
        
        response = requests.post(url, json=data, timeout=60)
        response.raise_for_status()
        result = response.json()
        
        return result.get("response", "[Llama2 7B返回内容为空]")
    
    except Exception as e:
        print(f"Llama2 7B调用异常: {str(e)}")
        return f"[Llama2 7B调用异常: {str(e)}]"

def call_llama2_13b(prompt, base_url="http://localhost:11434"):
    """调用Llama2 13B模型"""
    try:
        if not base_url.startswith(('http://', 'https://')):
            base_url = f"http://{base_url}"
        
        url = f"{base_url}/api/generate"
        data = {
            "model": "llama2:13b",
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": 512
            }
        }
        
        print(f"调用Llama2 13B: URL={url}")
        
        response = requests.post(url, json=data, timeout=60)
        response.raise_for_status()
        result = response.json()
        
        return result.get("response", "[Llama2 13B返回内容为空]")
    
    except Exception as e:
        print(f"Llama2 13B调用异常: {str(e)}")
        return f"[Llama2 13B调用异常: {str(e)}]"

def call_llama3_8b(prompt, base_url="http://localhost:11434"):
    """调用Llama3 8B模型"""
    try:
        if not base_url.startswith(('http://', 'https://')):
            base_url = f"http://{base_url}"
        
        url = f"{base_url}/api/generate"
        data = {
            "model": "llama3:8b",
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": 512
            }
        }
        
        print(f"调用Llama3 8B: URL={url}")
        
        response = requests.post(url, json=data, timeout=60)
        response.raise_for_status()
        result = response.json()
        
        return result.get("response", "[Llama3 8B返回内容为空]")
    
    except Exception as e:
        print(f"Llama3 8B调用异常: {str(e)}")
        return f"[Llama3 8B调用异常: {str(e)}]"

def call_mistral_7b(prompt, base_url="http://localhost:11434"):
    """调用Mistral 7B模型"""
    try:
        if not base_url.startswith(('http://', 'https://')):
            base_url = f"http://{base_url}"
        
        url = f"{base_url}/api/generate"
        data = {
            "model": "mistral:7b",
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": 512
            }
        }
        
        print(f"调用Mistral 7B: URL={url}")
        
        response = requests.post(url, json=data, timeout=60)
        response.raise_for_status()
        result = response.json()
        
        return result.get("response", "[Mistral 7B返回内容为空]")
    
    except Exception as e:
        print(f"Mistral 7B调用异常: {str(e)}")
        return f"[Mistral 7B调用异常: {str(e)}]"

def call_qwen_turbo(prompt, api_key=None):
    """千问API调用"""
    try:
        if api_key is None:
            api_key = os.getenv("DASHSCOPE_API_KEY")
            
        client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        
        completion = client.chat.completions.create(
            model="qwen-turbo",
            messages=[{"role": "user", "content": str(prompt)}],
            temperature=0.7,
            max_tokens=512,
            top_p=0.9
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"千问 API 调用失败: {str(e)}")
        return "[内容审核未通过]"

def call_glm45(prompt, api_key=None):
    """GLM-4.5 API调用"""
    try:
        if api_key is None:
            api_key = os.getenv("OPEN_BIGMODEL_API_KEY")
            
        client = OpenAI(
            api_key=api_key,
            base_url="https://open.bigmodel.cn/api/paas/v4/"
        )
        
        completion = client.chat.completions.create(
            model="glm-4.5",
            messages=[{"role": "user", "content": str(prompt)}],
            temperature=0.7,
            max_tokens=512
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"GLM-4.5 API 调用失败: {str(e)}")
        return "[内容审核未通过]"
    
def call_gpt(prompt, api_key=None, model=None):
    """GPT API 调用（model 需显式传入：如 'gpt-3.5-turbo'、'gpt-4.1'、'gpt-5'）"""
    try:
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")

        if model is None:
            raise ValueError(
                "call_gpt() 需要显式指定 model，例如 'gpt-3.5-turbo'、'gpt-4.1' 或 'gpt-5'"
            )

        client = OpenAI(
            api_key=api_key,
            base_url="https://api.gpt.ge/v1/",
        )

        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": str(prompt)}],
            temperature=0.7,
            max_tokens=1024,
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"GPT API 调用失败: {str(e)}")
        return f"[GPT API调用异常: {str(e)}]"


def call_cuc_deepseek(prompt, system_prompt=None, **kwargs):
    """调用中传DeepSeek API - 修复 reasoning_content 问题"""
    try:
        client = OpenAI(
            api_key=os.getenv("CUC_API_KEY"),
            base_url=os.getenv("CUC_API_URL", "https://aihub.cuc.edu.cn/console/v1")
        )
        
        messages = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": str(prompt)})
        
        model_name = os.getenv("CUC_MODEL_NAME", "DeepSeek-R1-Distill-Qwen-32B")
        
        # 自动重试机制，处理 reasoning_content 占用 tokens 问题
        max_retries = 2
        current_kwargs = kwargs.copy()
        
        # 设置默认 max_tokens
        if "max_tokens" not in current_kwargs:
            current_kwargs["max_tokens"] = 800
        
        for attempt in range(max_retries):
            try:
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    **current_kwargs
                )
                
                # 检查是否有内容
                content = completion.choices[0].message.content
                finish_reason = completion.choices[0].finish_reason
                
                if content and content.strip():
                    # 有有效内容，直接返回
                    return content
                elif finish_reason == "length" and attempt < max_retries - 1:
                    # 因长度截断且还有重试机会，增加 tokens 重试
                    current_tokens = current_kwargs.get("max_tokens", 800)
                    new_tokens = min(1600, current_tokens * 2)  # 最大不超过1600
                    current_kwargs["max_tokens"] = new_tokens
                    print(f"因长度截断，增加 max_tokens 到 {new_tokens} 重试")
                    continue
                else:
                    # 其他情况返回 None
                    return None
                    
            except Exception as e:
                print(f"中传DeepSeek API调用异常: {str(e)}")
                if attempt == max_retries - 1:
                    return "[API调用异常]"
                time.sleep(1)
                continue

        return None

    except Exception as e:
        print(f"中传DeepSeek API调用失败: {str(e)}")
        return "[API调用异常]"