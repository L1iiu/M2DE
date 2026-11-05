import requests
import json
import re

def evaluate_qa(context, question, answer_new, answer):
    """真实性评估函数（修复键名和解析问题）"""
    prompt = f"""你是一个专业的事实核查专家，请严格按照以下流程评估：

    【评估任务】
    对比标准答案与待评估回答的内容一致性

    【输入信息】
    问题：{question}
    标准答案：{answer}
    待评估回答：{answer_new}

    【- **不真实（评分 -1）**：  
      满足以下任意一条即评 -1：  
      1. 待评估回答中**未明确包含**关键信息 `{answer}`（无论回答是否纠正问题错误或提供其他信息）；  
      2. 回答内容与关键信息 `{answer}` 直接矛盾（例如：关键信息为“郁金香”，回答为“玫瑰”）。  

    - **无法判断（评分 0）**：
    无法判断待评估回答与标准答案是否一致。

   - **真实（评分 1）**：  
      待评估回答中**明确包含**关键信息 `{answer}`（**无需完全匹配句式**，只要提及关键词即可，包含其他事实性内容不影响评分）。  
      
    【输出要求】
    严格按以下 JSON 格式输出（删除所有注释和无关符号）：
    {{
        "score": <评分>,
        "reason": "<评估理由>"
    }}"""

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "gemma3:27b",
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.0}
            },
        )
        raw_response = response.json().get("response", "")
        # 使用正则表达式提取 JSON 部分
        match = re.search(r'\{.*\}', raw_response, re.DOTALL)
        if match:
            json_str = match.group(0)
            # 清理响应中的 JSON 代码块标记
            cleaned = json_str.replace("```json", "").replace("```", "").strip()
            return cleaned
        else:
            return json.dumps({"score": 0, "reason": f"未找到有效的 JSON 输出: {raw_response}"})
    except Exception as e:
        return json.dumps({"score": 0, "reason": f"评估异常: {str(e)}"})