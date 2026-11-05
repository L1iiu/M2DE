import concurrent.futures
import json
import logging
import time

# 导入各个维度的评估函数
from eval.privacy.privacy_eval_deepseek import evaluate_qa as privacy_eval_deepseek
from eval.privacy.privacy_eval_gemma import evaluate_qa as privacy_eval_gemma
from eval.privacy.privacy_eval_qwen import evaluate_qa as privacy_eval_qwen

from eval.safety.safety_eval_deepseek import evaluate_qa as safety_eval_deepseek
from eval.safety.safety_eval_gemma import evaluate_qa as safety_eval_gemma
from eval.safety.safety_eval_qwen import evaluate_qa as safety_eval_qwen

from eval.ethic.ethic_eval_deepseek import evaluate_qa as ethic_eval_deepseek
from eval.ethic.ethic_eval_gemma import evaluate_qa as ethic_eval_gemma
from eval.ethic.ethic_eval_qwen import evaluate_qa as ethic_eval_qwen

from eval.fairness.fairness_eval_deepseek import evaluate_qa as fairness_eval_deepseek
from eval.fairness.fairness_eval_gemma import evaluate_qa as fairness_eval_gemma
from eval.fairness.fairness_eval_qwen import evaluate_qa as fairness_eval_qwen

from eval.truth.truth_eval_deepseek import evaluate_qa as truth_eval_deepseek
from eval.truth.truth_eval_gemma import evaluate_qa as truth_eval_gemma
from eval.truth.truth_eval_qwen import evaluate_qa as truth_eval_qwen

logger = logging.getLogger(__name__)


def parallel_eval(eval_functions, context, question, answer_new, core_goal=None):
    """并行评估函数（支持传递 core_goal）"""
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # 把 core_goal 一并传给每个评估函数
        futures = [
            executor.submit(func, context, question, answer_new, core_goal)
            for func in eval_functions
        ]
        results = []
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(
                    f"评估函数 {eval_functions[i].__name__} 执行出错: {str(e)}",
                    exc_info=True
                )
                results.append(json.dumps({"score": 0, "reason": f"评估异常: {str(e)}"}))
    return results


def log_timestamp(func):
    """给评估函数套一层时间戳日志（兼容 core_goal 等额外参数）"""
    def wrapper(*args, **kwargs):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        logger.info(f"{func.__name__} 开始执行，时间戳: {timestamp}")
        return func(*args, **kwargs)
    return wrapper


# 为每个底层评估函数添加时间戳记录功能
privacy_eval_deepseek = log_timestamp(privacy_eval_deepseek)
privacy_eval_gemma = log_timestamp(privacy_eval_gemma)
privacy_eval_qwen = log_timestamp(privacy_eval_qwen)

safety_eval_deepseek = log_timestamp(safety_eval_deepseek)
safety_eval_gemma = log_timestamp(safety_eval_gemma)
safety_eval_qwen = log_timestamp(safety_eval_qwen)

ethic_eval_deepseek = log_timestamp(ethic_eval_deepseek)
ethic_eval_gemma = log_timestamp(ethic_eval_gemma)
ethic_eval_qwen = log_timestamp(ethic_eval_qwen)

fairness_eval_deepseek = log_timestamp(fairness_eval_deepseek)
fairness_eval_gemma = log_timestamp(fairness_eval_gemma)
fairness_eval_qwen = log_timestamp(fairness_eval_qwen)

truth_eval_deepseek = log_timestamp(truth_eval_deepseek)
truth_eval_gemma = log_timestamp(truth_eval_gemma)
truth_eval_qwen = log_timestamp(truth_eval_qwen)


# ================ 维度评估函数 =================
def privacy_eval(context, question, answer_new, core_goal=None):
    """隐私性评估"""
    eval_functions = [privacy_eval_deepseek, privacy_eval_gemma, privacy_eval_qwen]
    results = parallel_eval(eval_functions, context, question, answer_new, core_goal)
    return {"deepseek": results[0], "gemma": results[1], "qwen": results[2]}


def safety_eval(context, question, answer_new, core_goal=None):
    """安全性评估"""
    eval_functions = [safety_eval_deepseek, safety_eval_gemma, safety_eval_qwen]
    results = parallel_eval(eval_functions, context, question, answer_new, core_goal)
    return {"deepseek": results[0], "gemma": results[1], "qwen": results[2]}


def ethic_eval(context, question, answer_new, core_goal=None):
    """道德规范评估"""
    eval_functions = [ethic_eval_deepseek, ethic_eval_gemma, ethic_eval_qwen]
    results = parallel_eval(eval_functions, context, question, answer_new, core_goal)
    return {"deepseek": results[0], "gemma": results[1], "qwen": results[2]}


def fairness_eval(context, question, answer_new, core_goal=None):
    """公平性评估"""
    eval_functions = [fairness_eval_deepseek, fairness_eval_gemma, fairness_eval_qwen]
    results = parallel_eval(eval_functions, context, question, answer_new, core_goal)
    return {"deepseek": results[0], "gemma": results[1], "qwen": results[2]}


def truth_eval(context, question, answer_new, core_goal=None):
    """真实性评估"""
    eval_functions = [truth_eval_deepseek, truth_eval_gemma, truth_eval_qwen]
    results = parallel_eval(eval_functions, context, question, answer_new, core_goal)
    return {"deepseek": results[0], "gemma": results[1], "qwen": results[2]}


# 维度配置 - 简化版本
DIMENSION_CONFIG = {
    "privacy": {
        "eval_func": privacy_eval
    },
    "safety": {
        "eval_func": safety_eval
    },
    "ethic": {
        "eval_func": ethic_eval
    },
    "fairness": {
        "eval_func": fairness_eval
    },
    "truth": {
        "eval_func": truth_eval
    }
}
import concurrent.futures
import json
import logging
import time
from functools import partial

# 导入各个维度的评估函数
from eval.privacy.privacy_eval_deepseek import evaluate_qa as privacy_eval_deepseek
from eval.privacy.privacy_eval_gemma import evaluate_qa as privacy_eval_gemma
from eval.privacy.privacy_eval_qwen import evaluate_qa as privacy_eval_qwen

from eval.safety.safety_eval_deepseek import evaluate_qa as safety_eval_deepseek
from eval.safety.safety_eval_gemma import evaluate_qa as safety_eval_gemma
from eval.safety.safety_eval_qwen import evaluate_qa as safety_eval_qwen

from eval.ethic.ethic_eval_deepseek import evaluate_qa as ethic_eval_deepseek
from eval.ethic.ethic_eval_gemma import evaluate_qa as ethic_eval_gemma
from eval.ethic.ethic_eval_qwen import evaluate_qa as ethic_eval_qwen

from eval.fairness.fairness_eval_deepseek import evaluate_qa as fairness_eval_deepseek
from eval.fairness.fairness_eval_gemma import evaluate_qa as fairness_eval_gemma
from eval.fairness.fairness_eval_qwen import evaluate_qa as fairness_eval_qwen

from eval.truth.truth_eval_deepseek import evaluate_qa as truth_eval_deepseek
from eval.truth.truth_eval_gemma import evaluate_qa as truth_eval_gemma
from eval.truth.truth_eval_qwen import evaluate_qa as truth_eval_qwen

logger = logging.getLogger(__name__)


def parallel_eval(eval_functions, context, question, answer_new, core_goal=None):
    """并行评估函数（支持传递 core_goal）"""
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # 把 core_goal 一并传给每个评估函数
        futures = [
            executor.submit(func, context, question, answer_new, core_goal)
            for func in eval_functions
        ]
        results = []
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(
                    f"评估函数 {eval_functions[i].__name__} 执行出错: {str(e)}",
                    exc_info=True
                )
                results.append(json.dumps({"score": 0, "reason": f"评估异常: {str(e)}"}))
    return results


def log_timestamp(func):
    """给评估函数套一层时间戳日志（兼容 core_goal 等额外参数）"""
    def wrapper(*args, **kwargs):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        logger.info(f"{func.__name__} 开始执行，时间戳: {timestamp}")
        return func(*args, **kwargs)
    return wrapper


# 为每个底层评估函数添加时间戳记录功能
privacy_eval_deepseek = log_timestamp(privacy_eval_deepseek)
privacy_eval_gemma = log_timestamp(privacy_eval_gemma)
privacy_eval_qwen = log_timestamp(privacy_eval_qwen)

safety_eval_deepseek = log_timestamp(safety_eval_deepseek)
safety_eval_gemma = log_timestamp(safety_eval_gemma)
safety_eval_qwen = log_timestamp(safety_eval_qwen)

ethic_eval_deepseek = log_timestamp(ethic_eval_deepseek)
ethic_eval_gemma = log_timestamp(ethic_eval_gemma)
ethic_eval_qwen = log_timestamp(ethic_eval_qwen)

fairness_eval_deepseek = log_timestamp(fairness_eval_deepseek)
fairness_eval_gemma = log_timestamp(fairness_eval_gemma)
fairness_eval_qwen = log_timestamp(fairness_eval_qwen)

truth_eval_deepseek = log_timestamp(truth_eval_deepseek)
truth_eval_gemma = log_timestamp(truth_eval_gemma)
truth_eval_qwen = log_timestamp(truth_eval_qwen)


# ================ 维度评估函数 =================
def privacy_eval(context, question, answer_new, core_goal=None):
    """隐私性评估"""
    eval_functions = [privacy_eval_deepseek, privacy_eval_gemma, privacy_eval_qwen]
    results = parallel_eval(eval_functions, context, question, answer_new, core_goal)
    return {"deepseek": results[0], "gemma": results[1], "qwen": results[2]}


def safety_eval(context, question, answer_new, core_goal=None):
    """安全性评估"""
    eval_functions = [safety_eval_deepseek, safety_eval_gemma, safety_eval_qwen]
    results = parallel_eval(eval_functions, context, question, answer_new, core_goal)
    return {"deepseek": results[0], "gemma": results[1], "qwen": results[2]}


def ethic_eval(context, question, answer_new, core_goal=None):
    """道德规范评估"""
    eval_functions = [ethic_eval_deepseek, ethic_eval_gemma, ethic_eval_qwen]
    results = parallel_eval(eval_functions, context, question, answer_new, core_goal)
    return {"deepseek": results[0], "gemma": results[1], "qwen": results[2]}


def fairness_eval(context, question, answer_new, core_goal=None):
    """公平性评估"""
    eval_functions = [fairness_eval_deepseek, fairness_eval_gemma, fairness_eval_qwen]
    results = parallel_eval(eval_functions, context, question, answer_new, core_goal)
    return {"deepseek": results[0], "gemma": results[1], "qwen": results[2]}


def truth_eval(context, question, answer_new, core_goal=None):
    """真实性评估"""
    eval_functions = [truth_eval_deepseek, truth_eval_gemma, truth_eval_qwen]
    results = parallel_eval(eval_functions, context, question, answer_new, core_goal)
    return {"deepseek": results[0], "gemma": results[1], "qwen": results[2]}


# 维度配置 - 简化版本
DIMENSION_CONFIG = {
    "privacy": {
        "eval_func": privacy_eval
    },
    "safety": {
        "eval_func": safety_eval
    },
    "ethic": {
        "eval_func": ethic_eval
    },
    "fairness": {
        "eval_func": fairness_eval
    },
    "truth": {
        "eval_func": truth_eval
    }
}