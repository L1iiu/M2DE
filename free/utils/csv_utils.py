import os
import pandas as pd

def initialize_csv(output_path, dimension, has_reference):
    """为所有维度生成简化表头 - 多模型评估版本"""
    header = ["id", "subdimension"]
    
    # 初始轮次
    header.extend(["question0", "answer0"])
    if has_reference:
        header.append("reference_answer0")
    
    # 评估结果 - 各模型
    for model in ["deepseek", "gemma", "qwen"]:
        header.extend([f"eval_result0_{model}", f"score0_{model}"])
    
    # 投票得分
    header.append("score0_voted")
    
    # 结束轮次
    header.extend(["question_final", "answer_final"])
    if has_reference:
        header.append("reference_answer_final")
    
    # 评估结果 - 各模型
    for model in ["deepseek", "gemma", "qwen"]:
        header.extend([f"eval_result_final_{model}", f"score_final_{model}"])
    
    # 投票得分
    header.append("score_final_voted")
    
    # 攻破轮次
    header.append("break_round")
    
    if not os.path.exists(output_path):
        pd.DataFrame(columns=header).to_csv(output_path, index=False, encoding='utf-8')

def append_to_csv(output_path, record, dimension, has_reference):
    """统一追加记录格式"""
    df = pd.DataFrame([record])
    df.to_csv(output_path, mode='a', header=False, index=False, encoding='utf-8')

def update_stats(dimension, eval_data, DIMENSION_CONFIG):
    """统一统计更新逻辑 - 使用投票得分"""
    if "voted_score" in eval_data:
        score = str(eval_data["voted_score"])
        DIMENSION_CONFIG[dimension]["stats"][score] += 1