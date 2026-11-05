"""多维度对话评测主流程。"""

import os
import json
import time
from datetime import datetime
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Tuple
from typing import Optional
import pandas as pd
from tqdm import tqdm
from utils.logger_utils import DimensionLogger
from utils.csv_utils import initialize_csv
from utils.csv_utils import append_to_csv
from core.question_manager import generate_and_validate_question
from api_call import call_qwen_turbo, call_glm45, call_llama2_7b, call_llama2_13b, call_llama3_8b, call_mistral_7b, call_gpt
from utils.score_analysis import analyze_score_stats
from utils.score_analysis import analyze_score_stats
from utils.break_round_analysis import analyze_break_round_stats
from core.eval_utils import DIMENSION_CONFIG
from utils.stop_handler import should_stop


ModelCallFunc = Callable[[str], str]


# =============================================================================
# 工具函数
# =============================================================================

def _parse_ollama_model(model: str) -> Tuple[str, str, str]:
    """解析 Ollama 模型字符串，返回 (模型名, base_url, 标识符)。"""
    model_spec = model[7:]

    if ":" in model_spec and "://" in model_spec:
        parts = model_spec.split(":", 1)
        ollama_model_name = parts[0]
        ollama_base_url = parts[1]
    elif ":" in model_spec:
        ollama_model_name = model_spec
        ollama_base_url = "http://localhost:11434"
    else:
        ollama_model_name = model_spec
        ollama_base_url = "http://localhost:11434"

    model_identifier = f"ollama_{ollama_model_name.replace('-', '_').replace(':', '_')}"
    return ollama_model_name, ollama_base_url, model_identifier


def make_model_call_func(model: str) -> Tuple[ModelCallFunc, str]:
    """根据模型名称返回调用函数和模型标识符。"""

    if model.startswith("ollama:"):
        ollama_model_name, ollama_base_url, model_identifier = _parse_ollama_model(model)

        if ollama_model_name == "llama2:7b":
            model_call_func: ModelCallFunc = lambda p: call_llama2_7b(p, ollama_base_url)
        elif ollama_model_name == "llama2:13b":
            model_call_func = lambda p: call_llama2_13b(p, ollama_base_url)
        elif ollama_model_name == "llama3:8b":
            model_call_func = lambda p: call_llama3_8b(p, ollama_base_url)
        elif ollama_model_name == "mistral:7b":
            model_call_func = lambda p: call_mistral_7b(p, ollama_base_url)
        else:
            # 未知 Ollama 模型统一回落到 llama2:7b
            print(f"警告: 未知的Ollama模型 {ollama_model_name}，使用默认模型 llama2:7b")
            model_call_func = lambda p: call_llama2_7b(p, ollama_base_url)

        return model_call_func, model_identifier

    model_identifier = model

    mapping: Dict[str, ModelCallFunc] = {
        "qwen-turbo": lambda p: call_qwen_turbo(p),
        "glm-4.5": lambda p: call_glm45(p),
        "gpt-3.5-turbo": lambda p: call_gpt(p, model="gpt-3.5-turbo"),
        "gpt-4.1": lambda p: call_gpt(p, model="gpt-4.1"),
        "gpt-5": lambda p: call_gpt(p, model="gpt-5"),
    }
    return mapping[model], model_identifier


def parse_eval_results(raw_results: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """解析评估结果，兼容 JSON 字符串和 dict，失败时记为 0 分。"""
    scores: Dict[str, Any] = {}
    eval_data: Dict[str, Dict[str, Any]] = {}

    for model_name in ["deepseek", "gemma", "qwen"]:
        result_raw = raw_results.get(model_name, "")
        try:
            if isinstance(result_raw, str):
                data = json.loads(result_raw.strip())
            else:
                data = result_raw or {}
            eval_data[model_name] = data
            scores[model_name] = data.get("score", 0)
        except Exception:
            scores[model_name] = 0
            eval_data[model_name] = {"score": 0, "reason": "评估结果解析失败"}

    return scores, eval_data


# =============================================================================
# 主流程
# =============================================================================

def process_single_dimension(dimension: str, model: str) -> None:
    """处理单个维度的完整评测流程。"""

    print(f"\n开始处理维度: {dimension.upper()}，使用模型: {model}")
    config = DIMENSION_CONFIG[dimension].copy()

    # 输出目录
    os.makedirs(os.path.join("outputs", "result_csv"), exist_ok=True)
    os.makedirs(os.path.join("outputs", "score_stats"), exist_ok=True)
    os.makedirs(os.path.join("outputs", "break_round_stats"), exist_ok=True)

    # 模型调用函数 & 标识符
    model_call_func, model_identifier = make_model_call_func(model)

    # 结果文件路径
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"{dimension}_{model_identifier}_{timestamp}_results.csv"
    output_path = os.path.join("outputs", "result_csv", output_filename)
    config["output_path"] = output_path

    # 日志
    with DimensionLogger(dimension, model_identifier) as logger:
        logger.log_header()
        print(f"\n正在处理维度: {dimension.upper()}，使用模型: {model}")

        if model.startswith("ollama:"):
            ollama_model_name, ollama_base_url, _ = _parse_ollama_model(model)
            print(f"解析结果: 模型名称={ollama_model_name}, 基础URL={ollama_base_url}")

        # 数据集
        try:
            df = pd.read_csv(config["data_path"])
        except FileNotFoundError:
            print(f"数据集文件不存在: {config['data_path']}")
            return

        initialize_csv(output_path, dimension, config["has_reference"])

        # 逐样本处理
        process_dataset(df, model_call_func, config, dimension, logger, output_path)

        logger.write(f"\n{'=' * 50}\n")
        logger.write(f"{dimension.upper()}维度测试完成!\n")
        logger.write(f"结果已保存至: {output_path}\n")
        logger.write(f"详细日志已保存至: {logger.log_filename}\n")
        logger.write(f"{'=' * 50}\n")

        print(f"\n{dimension.upper()}维度测试完成!")
        print(f"结果已保存至: {output_path}")
        print(f"详细日志已保存至: {logger.log_filename}")

        analyze_results(model_identifier, dimension, output_path)


def process_dataset(
    df: pd.DataFrame,
    model_call_func: ModelCallFunc,
    config: Dict[str, Any],
    dimension: str,
    logger: DimensionLogger,
    output_path: str,
) -> None:
    """处理数据集中的每条记录。"""

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {dimension}"):
        if should_stop():
            print(f"\n收到停止信号，正在停止维度 {dimension} 的处理...")
            return

        logger.log_header(id=row["id"])

        subdimension = row.get("subdimension", "")
        record: Dict[str, Any] = {"id": row["id"], "subdimension": subdimension}

        all_rounds_data, break_round = run_dialogue_rounds(
            row=row,
            config=config,
            model_call_func=model_call_func,
            dimension=dimension,
            logger=logger,
        )

        save_results(record, all_rounds_data, break_round, config, output_path, dimension)

        time.sleep(1)


def run_dialogue_rounds(
    row: pd.Series,
    config: Dict[str, Any],
    model_call_func: ModelCallFunc,
    dimension: str,
    logger: DimensionLogger,
) -> Tuple[List[Dict[str, Any]], Optional[int]]:
    """执行多轮对话并收集结果。"""

    all_rounds_data: List[Dict[str, Any]] = []
    break_round: Optional[int] = None
    history_questions: List[str] = []
    new_qa_pairs: List[Tuple[str, str]] = []

    initial_question_col = "question" if dimension == "truth" else "prompt"

    initial_prompt = row[initial_question_col]
    history_questions.append(initial_prompt)
    initial_question = initial_prompt

    for round_idx in range(0, 11):
        if should_stop():
            print(f"\n收到停止信号，正在停止当前ID的处理...")
            break

        prompt = history_questions[-1]
        logger.write(f"=== 第 {round_idx} 轮 ===\n")
        logger.write(f"当前问题: {prompt}\n")

        round_data = process_single_round(
            round_idx=round_idx,
            history_questions=history_questions,
            config=config,
            model_call_func=model_call_func,
            reference_answer=row.get("reference_answer", ""),
            dimension=dimension,
            logger=logger,
            new_qa_pairs=new_qa_pairs,
        )

        if not round_data:
            logger.write(f"第{round_idx}轮未得到有效数据，终止本ID。\n")
            break

        all_rounds_data.append(round_data)

        try:
            eval_score = int(round_data.get("voted_score", 0))
        except ValueError:
            eval_score = 0

        if eval_score == -1:
            break_round = round_idx
            logger.write(f"ID {row['id']} 评测结束: 检测到违规回答（第{round_idx}轮）\n")
            break

        if round_idx == 10:
            logger.write(f"ID {row['id']} 评测结束: 达到最大轮次（第10轮）\n")
            break

        new_question = generate_next_question(
            dimension=dimension,
            initial_question=initial_question,
            current_question=history_questions[-1],
            generated_answer=round_data["answer"],
            reference_answer=round_data.get("reference_answer", ""),
            history_questions=history_questions,
            logger=logger,
        )
        if not new_question:
            logger.write("未能生成新问题，终止本ID。\n")
            break

        history_questions.append(new_question["new_question"])
        if dimension == "truth":
            new_qa_pairs.append((new_question["new_question"], new_question["new_answer"]))
        else:
            new_qa_pairs.append((new_question["new_question"], ""))

    return all_rounds_data, break_round


def process_single_round(
    round_idx: int,
    history_questions: List[str],
    config: Dict[str, Any],
    model_call_func: ModelCallFunc,
    reference_answer: str,
    dimension: str,
    logger: DimensionLogger,
    new_qa_pairs: List[Tuple[str, str]],
) -> Dict[str, Any]:
    """处理单轮对话：生成回答并评估。"""

    prompt = history_questions[-1]
    generated_answer = model_call_func(prompt) or "[无回答]"
    logger.log_round(round_idx, prompt, generated_answer)

    if config["has_reference"] and round_idx > 0:
        reference_answer = new_qa_pairs[-1][1] if new_qa_pairs else ""

    initial_question = history_questions[0] if history_questions else ""

    eval_results = config["eval_func"](
        context=initial_question,  
        question=prompt,
        answer_new=generated_answer,
        answer=reference_answer,
    )

    scores, eval_data = parse_eval_results(eval_results)

    score_values = list(scores.values())
    voted_score = max(set(score_values), key=score_values.count)

    logger.write("评估结果: ")
    for model_name in ["deepseek", "gemma", "qwen"]:
        logger.write(f"{model_name}: 得分={scores.get(model_name, 0)}")
        reason = eval_data[model_name].get("reason")
        if reason:
            logger.write(f", 理由: {reason[:50]}...")
        logger.write(" | ")
    logger.write(f"投票得分: {voted_score}\n")

    return {
        "question": prompt,
        "answer": generated_answer,
        "reference_answer": reference_answer,
        "eval_results": eval_results,
        "scores": scores,
        "voted_score": voted_score,
    }


def generate_next_question(
    dimension: str,
    initial_question: str,
    current_question: str,
    generated_answer: str,
    reference_answer: str,
    history_questions: List[str],
    logger: DimensionLogger,
) -> Optional[Dict[str, Any]]:
    """生成下一轮问题，内部封装异常处理。"""

    logger.write("\n准备生成新问题...\n")
    try:
        best_question = generate_and_validate_question(
            dimension=dimension,
            initial_question=initial_question,
            current_question=current_question,
            model_answer=generated_answer,
            reference_answer=reference_answer,
            history_questions=history_questions,
        )
    except Exception as e:
        logger.write(f"生成新问题时发生错误: {str(e)}\n")
        return None

    if best_question:
        logger.write(f"将使用新问题: {best_question['new_question']}\n")
        return best_question

    logger.write("未生成有效问题，终止评测\n")
    return None


def save_results(
    record: Dict[str, Any],
    all_rounds_data: List[Dict[str, Any]],
    break_round: Optional[int],
    config: Dict[str, Any],
    output_path: str,
    dimension: str,
) -> None:
    """保存结果到 CSV，字段结构与原实现一致。"""

    if all_rounds_data:
        round0 = all_rounds_data[0]
        record["question0"] = round0["question"]
        record["answer0"] = round0["answer"]
        if config["has_reference"]:
            record["reference_answer0"] = round0["reference_answer"]

        for model in ["deepseek", "gemma", "qwen"]:
            record[f"eval_result0_{model}"] = round0["eval_results"].get(model, "")
            record[f"score0_{model}"] = round0["scores"].get(model, 0)

        record["score0_voted"] = round0["voted_score"]

        final_round = all_rounds_data[-1]
        record["question_final"] = final_round["question"]
        record["answer_final"] = final_round["answer"]
        if config["has_reference"]:
            record["reference_answer_final"] = final_round["reference_answer"]

        for model in ["deepseek", "gemma", "qwen"]:
            record[f"eval_result_final_{model}"] = final_round["eval_results"].get(model, "")
            record[f"score_final_{model}"] = final_round["scores"].get(model, 0)

        record["score_final_voted"] = final_round["voted_score"]

    record["break_round"] = break_round if break_round is not None else "None"

    append_to_csv(output_path, record, dimension, config["has_reference"])


def analyze_results(model_name: str, dimension: str, output_path: str) -> None:
    """执行结果分析。"""

    base_filename = os.path.basename(output_path)
    file_identifier = base_filename.replace(".csv", "")

    score_stats_path = os.path.join("outputs", "score_stats", f"{file_identifier}_score_stats.json")
    analyze_score_stats(output_path, score_stats_path, score_column="score_final_voted")

    break_round_stats_path = os.path.join("outputs", "break_round_stats", f"{file_identifier}_break_round_stats.json")
    analyze_break_round_stats(output_path, break_round_stats_path)

    print(f"评分统计结果已保存至: {score_stats_path}")
    print(f"攻破轮次统计结果已保存至: {break_round_stats_path}")