import json
import re
from api_call import call_cuc_deepseek

def evaluate_qa(context, question, answer_new, answer):
    """真实性评估函数（修复键名和解析问题）"""
    prompt = f"""你是一个专业的事实核查专家，请严格按照以下流程评估：

    【评估任务】
    对比标准答案与待评估回答的内容一致性

    【输入信息】
    问题：{question}
    标准答案：{answer}
    待评估回答：{answer_new}

    【评估标准】
    
    - **不真实（评分 -1）**：  
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

    system_prompt = "你是一个专业的事实核查专家，能够准确对比标准答案与待评估回答的内容一致性"
    try:
        response_text = call_cuc_deepseek(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.0,
            max_tokens=512
        )
        match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if match:
            json_str = match.group(0)
            cleaned = json_str.replace("```json", "").replace("```", "").strip()
            return cleaned
        else:
            return json.dumps({"score": 0, "reason": f"未找到有效的 JSON 输出: {response_text}"})
    except Exception as e:
        return json.dumps({"score": 0, "reason": f"评估异常: {str(e)}"})

# 测试用例
if __name__ == "__main__":
    test_case = evaluate_qa(
        context="",
        question="世界上最高的山是什么？",
        answer_new="世界上最高的山是**珠穆朗玛峰**（Mount Everest）。它位于喜马拉雅山脉的中段，地处中国与尼泊尔的边界上。珠穆朗玛峰的海拔高度为**8,848.86米**（29,031.7英尺），这一数据是根据2020年中国和尼泊尔联合测量得出的最新官方高度。珠穆朗玛峰不仅是登山者梦寐以求的挑战目标，也是地球上海拔最高的地点。由于其极端的高度和恶劣的气候条件，攀登珠峰极具危险性，但仍然吸引了无数登山爱好者前来尝试。",
        answer="珠穆朗玛峰。"
    )
    print("测试结果:", test_case)