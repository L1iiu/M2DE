import time
import os
import sys

# 修改完整流程日志目录路径 
FULL_LOG_DIR = os.path.join("logs", "new_question_logs", "truth", "full_process")


class FileLogger:
    """只输出到文件的日志记录器"""
    def __init__(self, filename):
        self.log = open(filename, 'w', encoding='utf-8')
    
    def write(self, message):
        self.log.write(message)
    
    def flush(self):
        self.log.flush()
    
    def close(self):
        self.log.close()

def main():
    # 定义当前问题、模型答案、参考答案和历史问题
    current_question = "世界上最高的山是什么？"
    model_answer = "珠穆朗玛峰"
    reference_answer = "珠穆朗玛峰"
    history_questions = []

    # 只在运行时创建当前维度的日志目录
    os.makedirs(FULL_LOG_DIR, exist_ok=True)

    # 创建时间戳用于日志文件名
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    full_log_filename = os.path.join(FULL_LOG_DIR, f"full_process_{timestamp}.log")

    # 设置完整流程日志记录器
    sys.stdout = FileLogger(full_log_filename)

    # 记录开始时间
    start_time = time.time()

    # 导入必要的模块和函数
    from truth_generator import generate_new_questions
    from truth_validator import validate_and_select_best_question

    print(f"开始执行真实测试 - 时间戳: {timestamp}")
    print(f"完整日志将保存至: {full_log_filename}")

    # 1. 封装生成函数以便验证器重试
    def generate_questions():
        print("\n=== 生成阶段 ===")
        return generate_new_questions(
            current_question=current_question,
            model_answer=model_answer,
            reference_answer=reference_answer,
            history_questions=history_questions
        )

    # 2. 调用评估模块评估并选择最佳问题
    print("\n=== 评估阶段 ===")
    best_question = validate_and_select_best_question(
        current_question=current_question,
        generate_new_questions_func=generate_questions
    )

    # 记录结束时间
    end_time = time.time()
    # 计算总耗时
    total_time = end_time - start_time

    # 输出最佳问题
    print("\n=== 最终结果 ===")
    if best_question:
        print("最佳问题:")
        print(f"  问题: {best_question['new_question']}")
        print(f"  答案: {best_question['new_answer']}")
        print(f"  策略: {best_question['strategy']}")
        print("  维度评分:")
        print(f"    相关性: {best_question['relevance_score']} - {best_question['relevance_reason']}")
        print(f"    合理性: {best_question['rationality_score']} - {best_question['rationality_reason']}")
        print(f"    答案唯一性: {best_question['uniqueness_score']} - {best_question['uniqueness_reason']}")
        print(f"  总分: {best_question.get('score', 'N/A'):.2f}")
    else:
        print("未生成有效问题")

    # 输出总耗时
    print(f"\n整个流程总耗时: {total_time:.2f} 秒")

    # 恢复标准输出
    sys.stdout.close()
    sys.stdout = sys.__stdout__

    # 在终端显示日志位置
    print(f"测试完成! 完整日志已保存至: {full_log_filename}")

if __name__ == "__main__":
    main()