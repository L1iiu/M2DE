import os
import requests
from openai import OpenAI
from dotenv import load_dotenv
import logging
import json

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv('config/.env')

### 待评测模型调取

def call_llama2_7b(prompt, base_url="http://localhost:11434"):
    """调用Llama2 7B模型"""
    try:
        # 确保 base_url 有正确的协议前缀
        if not base_url.startswith(('http://', 'https://')):
            base_url = f"http://{base_url}"  # 默认使用 http
        
        # 构造Ollama原生接口请求参数
        url = f"{base_url}/api/generate"
        data = {
            "model": "llama2:7b",
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": 1024
            }
        }
        
        print(f"调用Llama2 7B: URL={url}")
        
        # 发送请求
        response = requests.post(url, json=data, timeout=60)
        response.raise_for_status()
        result = response.json()
        
        # 提取回答内容
        return result.get("response", "[Llama2 7B返回内容为空]")
    
    except Exception as e:
        print(f"Llama2 7B调用异常: {str(e)}")
        return f"[Llama2 7B调用异常: {str(e)}]"

def call_llama2_13b(prompt, base_url="http://localhost:11434"):
    """调用Llama2 13B模型"""
    try:
        # 确保 base_url 有正确的协议前缀
        if not base_url.startswith(('http://', 'https://')):
            base_url = f"http://{base_url}"  # 默认使用 http
        
        # 构造Ollama原生接口请求参数
        url = f"{base_url}/api/generate"
        data = {
            "model": "llama2:13b",
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": 1024
            }
        }
        
        print(f"调用Llama2 13B: URL={url}")
        
        # 发送请求
        response = requests.post(url, json=data, timeout=60)
        response.raise_for_status()
        result = response.json()
        
        # 提取回答内容
        return result.get("response", "[Llama2 13B返回内容为空]")
    
    except Exception as e:
        print(f"Llama2 13B调用异常: {str(e)}")
        return f"[Llama2 13B调用异常: {str(e)}]"

def call_llama3_8b(prompt, base_url="http://localhost:11434"):
    """调用Llama3 8B模型"""
    try:
        # 确保 base_url 有正确的协议前缀
        if not base_url.startswith(('http://', 'https://')):
            base_url = f"http://{base_url}"  # 默认使用 http
        
        # 构造Ollama原生接口请求参数
        url = f"{base_url}/api/generate"
        data = {
            "model": "llama3:8b",
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": 1024
            }
        }
        
        print(f"调用Llama3 8B: URL={url}")
        
        # 发送请求
        response = requests.post(url, json=data, timeout=60)
        response.raise_for_status()
        result = response.json()
        
        # 提取回答内容
        return result.get("response", "[Llama3 8B返回内容为空]")
    
    except Exception as e:
        print(f"Llama3 8B调用异常: {str(e)}")
        return f"[Llama3 8B调用异常: {str(e)}]"

def call_mistral_7b(prompt, base_url="http://localhost:11434"):
    """调用Mistral 7B模型"""
    try:
        # 确保 base_url 有正确的协议前缀
        if not base_url.startswith(('http://', 'https://')):
            base_url = f"http://{base_url}"  # 默认使用 http
        
        # 构造Ollama原生接口请求参数
        url = f"{base_url}/api/generate"
        data = {
            "model": "mistral:7b",
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": 1024
            }
        }
        
        print(f"调用Mistral 7B: URL={url}")
        
        # 发送请求
        response = requests.post(url, json=data, timeout=60)
        response.raise_for_status()
        result = response.json()
        
        # 提取回答内容
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
            max_tokens=1024,
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
            max_tokens=1024
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
            default_headers={"x-foo": "true"}
        )
        
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": str(prompt)}],
            temperature=0.7,
            max_tokens=1024
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"GPT API 调用失败: {str(e)}")
        return f"[GPT API调用异常: {str(e)}]"


###工具模型调取

def call_cuc_deepseek(prompt, system_prompt=None, **kwargs):
    """调用中传DeepSeek API"""
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
        
        completion = client.chat.completions.create(
            model=model_name,
            messages=messages,** kwargs
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"中传DeepSeek API调用失败: {str(e)}")
        return "[API调用异常]"