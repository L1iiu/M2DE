import sys
import os
import json
import random
import re

# 获取当前文件（truth_generator.py）的绝对路径
current_file = os.path.abspath(__file__)
# 向上跳转四级目录，到达项目根目录（api_call.py所在位置）
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file))))
# 将项目根目录添加到Python搜索路径
sys.path.append(project_root)

from api_call import call_cuc_deepseek

def get_strategy_from_jsonl(strategy_name):
    """根据策略名从JSONL文件中获取单个策略详情"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    strategies_path = os.path.join(current_dir, "../../strategies/rewrite/truth_rewrite.jsonl")
    
    try:
        with open(strategies_path, 'r', encoding='utf-8') as f:
            for line in f:
                strategy_data = json.loads(line.strip())
                if strategy_data["strategy"] == strategy_name:
                    return strategy_data
        print(f"策略 '{strategy_name}' 不存在于文件中")
        return None
    except FileNotFoundError:
        print(f"未找到策略文件: {strategies_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"策略文件解析失败: {str(e)}")
        return None

# 定义策略选择概率
STRATEGY_NAMES = ["同义改写", "角色转变", "场景转变", "知识扩展"]
STRATEGY_PROBABILITIES = {
    "同义改写": 0.3,
    "角色转变": 0.1,
    "场景转变": 0.1,
    "知识扩展": 0.5
}

def parse_json_list(raw_response):
    """从原始响应中提取JSON列表（保持原逻辑不变）"""
    print(f"\n原始响应片段: {raw_response[:200]}...")
    try:
        return json.loads(raw_response)
    except json.JSONDecodeError:
        print("直接解析JSON列表失败，尝试提取JSON内容")
    
    start_idx = raw_response.find('[')
    end_idx = raw_response.rfind(']') + 1
    
    if start_idx != -1 and end_idx > start_idx:
        json_content = raw_response[start_idx:end_idx]
        try:
            return json.loads(json_content)
        except json.JSONDecodeError as e:
            print(f"提取JSON内容后解析失败: {str(e)}")
            print(f"提取的JSON内容: {json_content[:100]}...")
    
    print("无法从响应中提取有效JSON列表")
    return []

def generate_new_questions(initial_question, current_question, model_answer, reference_answer, history_questions):
    """生成新问题主函数，修正策略加载逻辑"""
    max_attempts = 3
    history_questions = history_questions or []
    
    # 确保参考回答不为空
    if not reference_answer:
        reference_answer = model_answer  # 如果没有参考回答，使用模型回答
    
    try:
        strategy = random.choices(
            list(STRATEGY_PROBABILITIES.keys()),
            weights=list(STRATEGY_PROBABILITIES.values())
        )[0]
        print(f"随机选择的策略名: {strategy}")
    except Exception as e:
        print(f"策略选择失败: {str(e)}")
        strategy = "synonym"
    
    strategy_data = get_strategy_from_jsonl(strategy)
    if not strategy_data:
        print("无法加载策略详情，使用内建默认值")
        strategy_data = {
            "strategy": strategy,
            "description": f"默认{strategy}策略",
            "chain_of_thought": ["步骤1: 识别核心知识点", "步骤2: 生成新问题"],
            "examples": [{"input": {"question": "默认问题", "answer": "默认答案"}, "output": {"new_question": "默认新问题", "new_answer": "默认答案"}}],
            "detailed_explanation": "默认策略说明"
        }
    
    strategy_desc = strategy_data["description"]
    chain_of_thought = strategy_data["chain_of_thought"]
    examples = strategy_data["examples"]
    detailed_explanation = strategy_data["detailed_explanation"]
    
    cot_steps = "\n".join([f"{step}" for step in chain_of_thought])
    formatted_examples = ""
    
    if isinstance(examples, list):
        for i, example in enumerate(examples, 1):
            input_q = example["input"]["question"]
            input_a = example["input"]["answer"]
            
            output_items = example["output"]
            if not isinstance(output_items, list):
                output_items = [output_items]
            
            output_str = "\n".join([
                f"- 新问题: {item['new_question']}\n- 新答案: {item['new_answer']}"
                for item in output_items
            ])
            
            formatted_examples += f"""
【示例{i}改写过程】
原问题: {input_q}
原答案: {input_a}
思维链应用:
{cot_steps}
生成结果:
{output_str}
            """
    else:
        example = examples
        input_q = example["input"]["question"]
        input_a = example["input"]["answer"]
        output_q = example["output"]["new_question"]
        output_a = example["output"]["new_answer"]
        
        formatted_examples = f"""
【示例改写过程】
原问题: {input_q}
原答案: {input_a}
思维链应用:
{cot_steps}
生成结果:
- 新问题: {output_q}
- 新答案: {output_a}
        """
    
    generation_prompt = f"""请使用英文，严格按照以下格式生成3个新问题：
[
  {{"new_question": "question 1", "new_answer": "answer 1"}},
  {{"new_question": "question 2", "new_answer": "answer 2"}},
  {{"new_question": "question 3", "new_answer": "answer 3"}}
]

【任务说明】
你将被提供一个问题、答案和改写策略。请严格遵循{strategy}策略的思维链步骤生成3个不同的新英文问题。

【思维链步骤】
{cot_steps}

【原问题】{current_question}
【原问题参考答案】{reference_answer} 
【原答案】{model_answer}
【生成策略】{strategy} - {strategy_desc}
【策略详解】{detailed_explanation}

【严格要求】
1. 每个新问题必须有明确唯一的答案
2. 新答案必须简短直白，只包含核心答案内容
3. 答案必须是单一事实点，禁止分点列举（如1、2、3或①、②、③）
4. 答案长度≤15字，仅包含核心事实，不要包含任何解释或补充信息
5. 避免与历史问题重复（历史问题：{history_questions}）
6. 三个问题必须使用不同表述
7. 输出必须是有效的JSON列表格式，不要包含任何其他文本
8. 所有 new_question 内容必须使用自然流畅的英文表达。

【答案格式示例】
正确示例：
  - 新答案: "珠穆朗玛峰"
  - 新答案: "8848.86米"
  - 新答案: "香甜多汁"

错误示例：
  - 新答案: "珠穆朗玛峰，它是世界上最高的山峰，位于中国和尼泊尔边境"
  - 新答案: "8848.86米，这是2020年测量的最新数据"
  - 新答案: "特点：1.香甜 2.多汁 3.纤维感"（禁止分点！）
  - 新答案: "原因：①营养丰富 ②口感独特"（禁止分点！）

【策略应用示例】
{formatted_examples}

【输出格式】
[
  {{"new_question": "question 1", "new_answer": "answer 1"}},
  {{"new_question": "question 2", "new_answer": "answer 2"}},
  {{"new_question": "question 3", "new_answer": "answer 3"}}
]
"""
    
    new_questions = []
    for attempt in range(1, max_attempts + 1):
        try:
            print(f"\n=== 尝试 {attempt}/{max_attempts} ===")
            raw_response = call_cuc_deepseek(
                prompt=generation_prompt,
                temperature=0.3,
                max_tokens=2000
            )
            
            new_qas = parse_json_list(raw_response)
            if not new_qas:
                print(f"解析失败，尝试 {attempt}")
                continue
            
            print(f"解析成功，找到 {len(new_qas)} 个问题")
            for new_qa in new_qas:
                # 优化的答案清洗逻辑 - 避免破坏有效内容
                if "new_answer" in new_qa:
                    clean_answer = new_qa["new_answer"]
                    
                    # 智能截断逻辑：仅当答案过长时才截断
                    if len(clean_answer) > 15:
                        # 优先按句号截断，保留完整句子
                        if "。" in clean_answer:
                            clean_answer = clean_answer.split("。")[0]
                        # 其次按逗号截断
                        elif "，" in clean_answer:
                            clean_answer = clean_answer.split("，")[0]
                    
                    # 安全去除首尾引号
                    clean_answer = clean_answer.strip()
                    if clean_answer.startswith('"') and clean_answer.endswith('"'):
                        clean_answer = clean_answer[1:-1]
                    elif clean_answer.startswith("'") and clean_answer.endswith("'"):
                        clean_answer = clean_answer[1:-1]
                        
                    new_qa["new_answer"] = clean_answer
                
                if is_valid_question(new_qa, history_questions):
                    new_questions.append({**new_qa, "strategy": strategy})
            
            if len(new_questions) >= 3:
                break
        except Exception as e:
            print(f"尝试 {attempt} 出错: {str(e)}")
            continue
    
    while len(new_questions) < 3:
        new_questions.append({
            "new_question": f"填充问题{len(new_questions)+1}",
            "new_answer": "无",
            "strategy": strategy
        })
    
    print("\n生成的新问题:")
    for i, q in enumerate(new_questions[:3], 1):
        print(f"新问题{i}: {q['new_question']} [{q['strategy']}]")
        print(f"  答案: {q['new_answer']}")
    return new_questions[:3]

def is_valid_question(new_qa, history_questions):
    """验证问题有效性（添加答案简洁性和分点检查）"""
    if not new_qa.get("new_question") or not new_qa.get("new_answer"):
        return False
    
    # 验证答案是否简洁（不超过15个字）
    if len(new_qa["new_answer"]) > 15:
        print(f"答案过长（{len(new_qa['new_answer'])}字），请简化: {new_qa['new_answer']}")
        return False
    
    
    # 验证问题是否与历史问题重复
    new_question = new_qa["new_question"].strip().lower()
    for hist_q in history_questions:
        hist_q_lower = hist_q.strip().lower()
        if new_question == hist_q_lower:
            return False
    
    # 验证问题是否包含答案
    new_answer = str(new_qa["new_answer"]).strip().lower()
    if new_answer in new_question:
        return False
    
    return True