import re
from api_call import call_cuc_deepseek
import json
import concurrent.futures
import os
import time
import logging
import requests

# 创建日志目录 
LOG_DIR = os.path.join("logs", "new_question_logs", "truth", "validator")

# 定义答案不变的策略集合
ANSWER_PRESERVING_STRATEGIES = {"同义改写", "角色转变", "场景转变"}

def setup_logger(qid):
    """为每个问题创建独立的日志文件"""
    logger = logging.getLogger(qid)
    
    if logger.handlers:
        return logger
    
    logger.setLevel(logging.INFO)
    logger.propagate = False
    
    log_file = os.path.join(LOG_DIR, f"{qid}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    
    return logger

def score_all_dimensions(new_question, new_answer, original_question, logger):
    """一次性评估所有维度，返回结构化评分"""
    prompt = f"""请作为专业评估员，对以下问题进行全面评估：

【评估指标及标准】
1. 相关性 (0-10分): 
   - 评估新问题与原问题({original_question})的知识领域关联程度
   - 10分: 完美关联（直接扩展原问题核心知识）
       例：原问题"世界第一高峰？" → 新问题"珠峰高度？"
   - 9分: 很强关联（直接对原问题进行拓展）
       例：原问题"世界第一高峰？" → 新问题"珠峰位于哪里？"
   - 7-8分: 强关联（相同知识领域，不同角度）
       例：原问题"世界第一高峰？" → 新问题"世界第二高峰？"
   - 5-6分: 中等关联（相关领域但有偏移）
       例：原问题"世界第一高峰？" → 新问题"世界第二高峰位置？"
   - 3-4分: 弱关联（同一大类但不同主题）
       例：原问题"世界第一高峰？" → 新问题"世界最大湖泊？"
   - 0-2分: 无关联（完全不同领域）
       例：原问题"世界第一高峰？" → 新问题"苹果颜色？"

2. 合理性 (0-10分):
   - 评估新问题本身是否合理、可理解
   - 10分: 问题清晰明确，符合常识和逻辑
   - 5分: 问题基本可理解，但有轻微歧义
   - 0分: 问题无法理解或明显不合理

3. 答案唯一性 (0-10分):
   - 评估答案是否明确、唯一且正确
   - 10分: 答案明确唯一，无争议
   - 5分: 答案基本正确，但有轻微模糊
   - 0分: 答案错误、模糊或有多种解释

【输入信息】
- 原问题: {original_question}
- 新问题: {new_question}
- 新答案: {new_answer}

【输出要求】
请严格按以下JSON格式返回结果：
{{
  "score1": 相关性分数,
  "reason1": "相关性理由(30字内)",
  "score2": 合理性分数,
  "reason2": "合理性理由(30字内)",
  "score3": 唯一性分数,
  "reason3": "唯一性理由(30字内)"
}}

请确保：
1. 所有评分均为0-10的整数
2. 理由简明扼要（不超过30字）
3. 不要包含任何其他内容
"""

    try:
        start_time = time.time()
        response = call_cuc_deepseek(
            prompt=prompt,
            temperature=0.0,
            max_tokens=1024
        )
        elapsed = time.time() - start_time
        
        logger.info(f"原始响应: {response.strip()}")
        
        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                scores = json.loads(json_match.group(0))
            else:
                scores = json.loads(response)
                
            for key in ['score1', 'score2', 'score3']:
                if key not in scores:
                    raise ValueError(f"缺少评分项: {key}")
                scores[key] = int(scores[key])
                if not (0 <= scores[key] <= 10):
                    raise ValueError(f"评分超出范围: {scores[key]}")
            
            logger.info(f"解析成功 | 耗时: {elapsed:.2f}s")
            return scores
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"解析失败: {str(e)}")
            return {
                "score1": 0, "reason1": "解析失败",
                "score2": 0, "reason2": "解析失败",
                "score3": 0, "reason3": "解析失败"
            }
    except Exception as e:
        logger.error(f"API调用出错: {e}")
        return {
            "score1": 0, "reason1": "API错误",
            "score2": 0, "reason2": "API错误",
            "score3": 0, "reason3": "API错误"
        }

def validate_answer_with_local_model(question, answer):
    """使用本地模型验证答案正确性，返回验证结果和原因"""
    prompt = f"""请判断以下答案是否是问题的正确回答，并严格按以下JSON格式返回结果：
{{
  "valid": "<是或否>",
  "reason": "<验证原因(20字内)>"
}}

要求：
1. "valid"字段只能是"是"或"否"
2. "reason"字段为验证原因（不超过20字）
3. 不要包含任何其他内容

【问题】{question}
【答案】{answer}"""

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "qwen3:32b",
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.0}
            },
            timeout=30
        )
        raw_response = response.json().get("response", "")
        
        # 提取JSON部分
        json_match = re.search(r'\{[\s\S]*\}', raw_response)
        if json_match:
            try:
                validation_result = json.loads(json_match.group(0))
                is_valid = validation_result.get("valid", "否") == "是"
                reason = validation_result.get("reason", "无原因")
                return is_valid, reason
            except json.JSONDecodeError:
                return False, "JSON解析失败"
        else:
            return False, "未找到有效JSON"
    except Exception as e:
        return False, f"验证异常: {str(e)}"

def validate_answer(new_question, new_answer, strategy, logger):
    """验证答案是否有效 - 使用双模型并行验证（仅知识拓展策略）"""
    # 答案保留策略直接通过
    if strategy in ANSWER_PRESERVING_STRATEGIES:
        reason = f"策略 '{strategy}' 跳过答案验证"
        logger.info(reason)
        return True, reason
    
    logger.info(f"双重验证: {new_question}")
    start_time = time.time()
    
    # 并行调用两个模型
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # 远程模型验证
        remote_future = executor.submit(
            self_validate_answer, 
            new_question, new_answer, logger
        )
        # 本地模型验证
        local_future = executor.submit(
            validate_answer_with_local_model,
            new_question, new_answer
        )
        
        remote_result, remote_reason = remote_future.result()
        local_result, local_reason = local_future.result()
    
    elapsed = time.time() - start_time
    is_valid = remote_result and local_result
    
    # 记录详细的验证结果和原因
    logger.info(f"模型1验证: {'通过' if remote_result else '不通过'} - {remote_reason}")
    logger.info(f"模型2验证: {'通过' if local_result else '不通过'} - {local_reason}")
    
    if is_valid:
        result_text = f"双重验证通过 | 耗时: {elapsed:.2f}s"
        logger.info(result_text)
        return True, result_text
    else:
        # 详细说明未通过的原因
        failure_reasons = []
        if not remote_result:
            failure_reasons.append(f"模型1: {remote_reason}")
        if not local_result:
            failure_reasons.append(f"模型2: {local_reason}")
        
        result_text = f"双重验证不通过 | 原因: {'; '.join(failure_reasons)} | 耗时: {elapsed:.2f}s"
        logger.info(result_text)
        return False, result_text

def self_validate_answer(question, answer, logger):
    """原始验证方法（供并行调用），返回验证结果和原因"""
    prompt = f"""请判断以下答案是否是问题的正确回答，并严格按以下JSON格式返回结果：
{{
  "valid": "<是或否>",
  "reason": "<验证原因(20字内)>"
}}

要求：
1. "valid"字段只能是"是"或"否"
2. "reason"字段为验证原因（不超过20字）
3. 不要包含任何其他内容

【问题】{question}
【答案】{answer}"""

    try:
        response = call_cuc_deepseek(
            prompt=prompt,
            temperature=0.0,
            max_tokens=200
        )
        
        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                validation_result = json.loads(json_match.group(0))
                is_valid = validation_result.get("valid", "否") == "是"
                reason = validation_result.get("reason", "无原因")
                return is_valid, reason
            else:
                return False, "未找到有效JSON"
        except (json.JSONDecodeError, ValueError) as e:
            cleaned = response.strip().lower()
            if "是" in cleaned:
                return True, "从原始响应中提取: 是"
            elif "否" in cleaned:
                return False, "从原始响应中提取: 否"
            else:
                return False, "无法判断验证结果"
    except Exception as e:
        return False, f"验证异常: {str(e)}"

def evaluate_question(new_qa, current_question, qid):
    """评估单个问题，添加唯一标识符"""
    logger = setup_logger(qid)
    
    new_question = new_qa["new_question"]
    new_answer = new_qa["new_answer"]
    strategy = new_qa["strategy"]
    
    logger.info("=" * 60)
    logger.info(f"开始评估问题: {new_question}")
    logger.info(f"使用策略: {strategy} | 原问题: {current_question}")
    logger.info("=" * 60)

    # 修改：根据策略类型进行验证
    is_valid, valid_reason = validate_answer(new_question, new_answer, strategy, logger)
    
    valid_status = "通过" if is_valid else "不通过"
    logger.info(f"答案有效性: {valid_status} - {valid_reason}")

    if not is_valid:
        logger.info("答案验证不通过，跳过其他维度评估")
        logger.info("=" * 60)
        logger.info(f"评估完成 | 总分: 0.00")
        logger.info("=" * 60)
        
        print(f"[评估进度] 问题 {qid} 评估完成 | 总分: 0.00 (答案无效)")
        
        return {
            **new_qa,
            "score": 0,
            "relevance_score": 0,
            "relevance_reason": "答案无效",
            "rationality_score": 0,
            "rationality_reason": "答案无效",
            "uniqueness_score": 0,
            "uniqueness_reason": "答案无效",
            "log_file": f"{LOG_DIR}/{qid}.log"
        }
    
    scores = score_all_dimensions(
        new_question, 
        new_answer, 
        current_question, 
        logger
    )
    
    rel_score = scores.get("score1", 0)
    rat_score = scores.get("score2", 0)
    uni_score = scores.get("score3", 0)
    
    score = (rel_score + rat_score + uni_score) / 3.0

    logger.info("维度评分:")
    logger.info(f"  相关性: {rel_score} - {scores.get('reason1', '')}")
    logger.info(f"  合理性: {rat_score} - {scores.get('reason2', '')}")
    logger.info(f"  答案唯一性: {uni_score} - {scores.get('reason3', '')}")
    logger.info(f"最终得分 (三维平均): {score:.2f}")

    new_qa["score"] = score
    new_qa.update({
        "relevance_score": rel_score,
        "relevance_reason": scores.get("reason1", "无理由"),
        "rationality_score": rat_score,
        "rationality_reason": scores.get("reason2", "无理由"),
        "uniqueness_score": uni_score,
        "uniqueness_reason": scores.get("reason3", "无理由"),
        "log_file": f"{LOG_DIR}/{qid}.log"
    })

    logger.info("=" * 60)
    logger.info(f"评估完成 | 总分: {score:.2f}")
    logger.info("=" * 60)
    
    print(f"[评估进度] 问题 {qid} 评估完成 | 总分: {score:.2f}")
    
    return new_qa

def validate_and_select_best_question(current_question, generate_new_questions_func, max_retries=3):
    """评估并选择评分最高的问题（增加重试机制）"""
    os.makedirs(LOG_DIR, exist_ok=True)
    
    for attempt in range(1, max_retries + 1):
        print(f"\n=== 评估批次 #{attempt} ===")
        print(f"开始评估问题，原问题: {current_question}")
        
        # 生成新问题
        new_questions = generate_new_questions_func()
        if not new_questions:
            print("生成器返回空列表，跳过评估")
            continue
            
        print(f"将评估 {len(new_questions)} 个问题，日志保存在 {LOG_DIR} 目录")
        
        best_question = None
        best_score = -1
        qid_counter = 1
        all_zero = True

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for new_qa in new_questions:
                qid = f"TQ{qid_counter}_{attempt}"
                qid_counter += 1
                futures.append(executor.submit(
                    evaluate_question, 
                    new_qa, 
                    current_question, 
                    qid
                ))
            
            for future in concurrent.futures.as_completed(futures):
                new_qa = future.result()
                score = new_qa["score"]
                
                if score > 0:
                    all_zero = False
                
                if score > best_score:
                    best_score = score
                    best_question = new_qa

        # 如果有有效问题，直接返回
        if not all_zero and best_question:
            print("\n=== 评估总结 ===")
            print("最佳问题详情:")
            print(f"  问题: {best_question['new_question']}")
            print(f"  答案: {best_question['new_answer']}")
            print(f"  策略: {best_question['strategy']}")
            print("  维度评分:")
            print(f"    相关性: {best_question['relevance_score']} - {best_question['relevance_reason']}")
            print(f"    合理性: {best_question['rationality_score']} - {best_question['rationality_reason']}")
            print(f"    答案唯一性: {best_question['uniqueness_score']} - {best_question['uniqueness_reason']}")
            print(f"  最终得分 (三维平均): {best_question['score']:.2f}")
            print(f"  详细日志: {best_question['log_file']}")
            return best_question
        
        print(f"本批次所有问题评估得分为0，将重试({attempt}/{max_retries})")
    
    print("\n!!! 所有批次评估失败，未找到有效问题 !!!")
    return None