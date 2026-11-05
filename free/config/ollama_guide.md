# Ollama 模型配置说明

## 支持的模型格式

1. **基本格式**: `ollama:模型名称`
   - 示例: `ollama:llama2`
   - 使用默认本地地址: http://localhost:11434/v1

2. **自定义地址格式**: `ollama:模型名称:自定义URL`
   - 示例: `ollama:llama2:http://192.168.1.100:11434/v1`
   - 示例: `ollama:qwen:7b:http://localhost:8080/v1`

## 常用模型名称

- `llama2` - Meta Llama 2
- `llama3` - Meta Llama 3  
- `qwen:7b` - 通义千问 7B
- `qwen:14b` - 通义千问 14B
- `codellama` - Code Llama
- `mistral` - Mistral 7B
- `mixtral` - Mixtral 8x7B

## 启动 Ollama 服务

```bash
# 安装 Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# 拉取模型
ollama pull llama2

# 启动服务（默认已在后台运行）
ollama serve