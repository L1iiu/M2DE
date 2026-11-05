import os
import pandas as pd
import json

def initialize_csv(output_path):
    """初始化CSV文件 - 简化版本"""
    header = [
        "data_id", "core_goal", "subdimension", "script_id",
        "final_question", "final_response",
        "eval_deepseek_score", "eval_deepseek_reason",
        "eval_gemma_score", "eval_gemma_reason", 
        "eval_qwen_score", "eval_qwen_reason",
        "score_final_voted", "broken_round"
    ]
    
    if not os.path.exists(output_path):
        pd.DataFrame(columns=header).to_csv(output_path, index=False, encoding='utf-8')

def append_to_csv(output_path, record):
    """追加记录到CSV - 简化版本"""
    import pandas as pd
    import json
    
    # 提取最后一轮对话信息和评估结果
    dialogue_history = record.get("dialogue_history", [])
    final_question = ""
    final_response = ""
    
    # 初始化评分字段
    eval_deepseek_score = ""
    eval_deepseek_reason = ""
    eval_gemma_score = ""
    eval_gemma_reason = ""
    eval_qwen_score = ""
    eval_qwen_reason = ""

    # ⭐ 修改点：score_final_voted 直接从顶层 record 读，而不是从 last_turn 里读
    score_final_voted = record.get("score_final_voted", "")
    
    if dialogue_history:
        last_turn = dialogue_history[-1]
        final_question = last_turn.get("question", "")
        final_response = last_turn.get("response", "")
        
        # 提取详细评估结果
        eval_results = last_turn.get("eval_results", {})
        if eval_results:
            # DeepSeek评估结果
            deepseek_result = eval_results.get("deepseek", "{}")
            try:
                deepseek_data = json.loads(deepseek_result)
                eval_deepseek_score = deepseek_data.get("score", "")
                eval_deepseek_reason = deepseek_data.get("reason", "")
            except:
                pass
            
            # Gemma评估结果
            gemma_result = eval_results.get("gemma", "{}")
            try:
                gemma_data = json.loads(gemma_result)
                eval_gemma_score = gemma_data.get("score", "")
                eval_gemma_reason = gemma_data.get("reason", "")
            except:
                pass
            
            # Qwen评估结果
            qwen_result = eval_results.get("qwen", "{}")
            try:
                qwen_data = json.loads(qwen_result)
                eval_qwen_score = qwen_data.get("score", "")
                eval_qwen_reason = qwen_data.get("reason", "")
            except:
                pass
        
        # ❌ 原来的这一行会覆盖顶层结果，而且 last_turn 里其实没有这个字段
        # score_final_voted = last_turn.get("score_final_voted", "")
    
    simplified_record = {
        "data_id": record.get("data_id", ""),
        "core_goal": record.get("core_goal", ""),
        "subdimension": record.get("subdimension", ""),
        "script_id": record.get("script_id", ""),
        "final_question": final_question,
        "final_response": final_response,
        "eval_deepseek_score": eval_deepseek_score,
        "eval_deepseek_reason": eval_deepseek_reason,
        "eval_gemma_score": eval_gemma_score,
        "eval_gemma_reason": eval_gemma_reason,
        "eval_qwen_score": eval_qwen_score,
        "eval_qwen_reason": eval_qwen_reason,
        "score_final_voted": score_final_voted,
        "broken_round": record.get("broken_round", "None")
    }
    
    df = pd.DataFrame([simplified_record])
    df.to_csv(output_path, mode='a', header=False, index=False, encoding='utf-8')