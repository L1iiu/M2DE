import concurrent.futures
import json
import logging
import time
from typing import Any, Callable, Dict, List

from eval_scripts.safety.safety_eval_deepseek import evaluate_qa as safety_eval_deepseek
from eval_scripts.safety.safety_eval_gemma import evaluate_qa as safety_eval_gemma
from eval_scripts.safety.safety_eval_qwen import evaluate_qa as safety_eval_qwen

from eval_scripts.ethic.ethic_eval_deepseek import evaluate_qa as ethic_eval_deepseek
from eval_scripts.ethic.ethic_eval_gemma import evaluate_qa as ethic_eval_gemma
from eval_scripts.ethic.ethic_eval_qwen import evaluate_qa as ethic_eval_qwen

from eval_scripts.fairness.fairness_eval_deepseek import evaluate_qa as fairness_eval_deepseek
from eval_scripts.fairness.fairness_eval_gemma import evaluate_qa as fairness_eval_gemma
from eval_scripts.fairness.fairness_eval_qwen import evaluate_qa as fairness_eval_qwen

from eval_scripts.privacy.privacy_eval_deepseek import evaluate_qa as privacy_eval_deepseek
from eval_scripts.privacy.privacy_eval_gemma import evaluate_qa as privacy_eval_gemma
from eval_scripts.privacy.privacy_eval_qwen import evaluate_qa as privacy_eval_qwen

from eval_scripts.truth.truth_eval_deepseek import evaluate_qa as truth_eval_deepseek
from eval_scripts.truth.truth_eval_gemma import evaluate_qa as truth_eval_gemma
from eval_scripts.truth.truth_eval_qwen import evaluate_qa as truth_eval_qwen


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EvalFunc = Callable[[str, str, str, str], Any]


# =============================================================================
# 通用工具
# =============================================================================

def log_timestamp(func: EvalFunc) -> EvalFunc:
    """在评估函数调用前打印时间戳，不改变原函数行为。"""

    def wrapper(context: str, question: str, answer_new: str, answer: str) -> Any:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        logger.info("%s 开始执行，时间戳: %s", func.__name__, timestamp)
        return func(context, question, answer_new, answer)

    return wrapper  # type: ignore[return-value]


def _parallel_eval(
    eval_functions: List[EvalFunc],
    context: str,
    question: str,
    answer_new: str,
    answer: str,
) -> List[Any]:
    """并行执行多个评估函数，返回结果顺序与 eval_functions 一致。"""
    results: List[Any] = [None] * len(eval_functions)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_index = {
            executor.submit(func, context, question, answer_new, answer): idx
            for idx, func in enumerate(eval_functions)
        }

        for future in concurrent.futures.as_completed(future_to_index):
            idx = future_to_index[future]
            func = eval_functions[idx]
            try:
                results[idx] = future.result()
            except Exception as e:  # noqa: BLE001
                logger.error(
                    "评估函数 %s 执行出错: %s",
                    func.__name__,
                    str(e),
                    exc_info=True,
                )
                # 保持原行为：返回可被 json.loads 的字符串
                results[idx] = json.dumps(
                    {"score": 0, "reason": f"评估异常: {str(e)}"},
                    ensure_ascii=False,
                )

    return results


def _make_dimension_eval(
    deepseek_eval: EvalFunc,
    gemma_eval: EvalFunc,
    qwen_eval: EvalFunc,
) -> EvalFunc:
    """生成一个维度评估函数，返回 {'deepseek','gemma','qwen'} 的结果字典。"""
    deepseek_eval = log_timestamp(deepseek_eval)
    gemma_eval = log_timestamp(gemma_eval)
    qwen_eval = log_timestamp(qwen_eval)

    def eval_dimension(context: str, question: str, answer_new: str, answer: str) -> Dict[str, Any]:
        eval_functions = [deepseek_eval, gemma_eval, qwen_eval]
        results = _parallel_eval(eval_functions, context, question, answer_new, answer)
        return {
            "deepseek": results[0],
            "gemma": results[1],
            "qwen": results[2],
        }

    return eval_dimension


# =============================================================================
# 维度评估函数（对外接口）
# =============================================================================

safety_eval = _make_dimension_eval(
    safety_eval_deepseek, safety_eval_gemma, safety_eval_qwen
)
ethic_eval = _make_dimension_eval(
    ethic_eval_deepseek, ethic_eval_gemma, ethic_eval_qwen
)
fairness_eval = _make_dimension_eval(
    fairness_eval_deepseek, fairness_eval_gemma, fairness_eval_qwen
)
privacy_eval = _make_dimension_eval(
    privacy_eval_deepseek, privacy_eval_gemma, privacy_eval_qwen
)
truth_eval = _make_dimension_eval(
    truth_eval_deepseek, truth_eval_gemma, truth_eval_qwen
)


# =============================================================================
# 维度配置
# =============================================================================

DIMENSION_CONFIG = {
    "safety": {
        "data_path": "data/safety.csv",
        "eval_func": safety_eval,
        "output_path": "outputs/safety_results.csv",
        "stats": {"-1": 0, "0": 0, "1": 0},
        "has_reference": False,
    },
    "ethic": {
        "data_path": "data/ethic.csv",
        "eval_func": ethic_eval,
        "output_path": "outputs/ethic_results.csv",
        "stats": {"-1": 0, "0": 0, "1": 0},
        "has_reference": False,
    },
    "fairness": {
        "data_path": "data/fairness.csv",
        "eval_func": fairness_eval,
        "output_path": "outputs/fairness_results.csv",
        "stats": {"-1": 0, "0": 0, "1": 0},
        "has_reference": False,
    },
    "privacy": {
        "data_path": "data/privacy.csv",
        "eval_func": privacy_eval,
        "output_path": "outputs/privacy_results.csv",
        "stats": {"-1": 0, "0": 0, "1": 0},
        "has_reference": False,
    },
    "truth": {
        "data_path": "data/truth.csv",
        "eval_func": truth_eval,
        "output_path": "outputs/truth_results.csv",
        "stats": {"-1": 0, "0": 0, "1": 0},
        "has_reference": True,  # 真实性维度使用参考答案
    },
}
