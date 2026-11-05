import os
import sys
import time
import logging  
import json

class DimensionLogger:
    """维度测试日志记录器"""
    def __init__(self, dimension, model_name):
        # 保存维度和模型名称
        self.dimension = dimension
        self.model_name = model_name
        
        # 动态创建日志目录
        self.log_dir = os.path.join("logs", "script_logs", f"{dimension}_{model_name}")
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 创建时间戳用于日志文件名
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.log_filename = os.path.join(self.log_dir, f"script_test_{timestamp}.log")
        
        # 保存原始标准输出
        self.original_stdout = sys.stdout
        self.log_file = open(self.log_filename, 'w', encoding='utf-8')
        sys.stdout = self.log_file  
        
        # 设置日志过滤器，避免HTTP请求等INFO日志输出
        self.setup_log_filter()
    
    def setup_log_filter(self):
        """设置日志过滤器移除HTTP请求日志"""
        # 获取根日志记录器
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.ERROR)
        
        # 设置特定库的日志级别
        for name in ["httpx", "httpcore"]:
            logging.getLogger(name).setLevel(logging.ERROR)
    
    def write(self, message):
        """只输出到日志文件"""
        self.log_file.write(message)
    
    def flush(self):
        """刷新缓冲区"""
        self.original_stdout.flush()
        self.log_file.flush()
    
    def close(self):
        """关闭日志记录器"""
        if not self.log_file.closed:
            self.log_file.close()
        sys.stdout = self.original_stdout
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        """上下文管理器退出"""
        self.close()
    
    def log_header(self):
        """记录测试头部信息"""
        self.write(f"\n{'='*60}\n")
        self.write(f"剧本模式测试开始\n")
        self.write(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.write(f"维度: {self.dimension.upper()}\n")
        self.write(f"模型: {self.model_name}\n")
        self.write(f"日志文件: {self.log_filename}\n")
        self.write(f"{'='*60}\n\n")
    
    def log_script_start(self, script_name, core_goal, data_id, execution_id):
        """记录剧本开始信息"""
        self.write(f"\n▶▶ 开始执行剧本: {script_name}\n")
        self.write(f"   核心目标: {core_goal}\n")
        self.write(f"   数据ID: {data_id} | 执行: {execution_id + 1}\n")
        self.write(f"{'-'*50}\n")
    
    def log_turn(self, turn_number, question, response, eval_results, scores, score_final_voted, step_intent):
        """记录轮次信息 - 包含详细评估结果"""
        self.write(f"\n【第{turn_number}轮】步骤意图: {step_intent}\n")
        self.write(f"问题: {question}\n")
        self.write(f"回答: {response}\n")
        
        # 记录模型评分详情
        self.write(f"模型评分详情:\n")
        for model_name in ["deepseek", "gemma", "qwen"]:
            result_str = eval_results.get(model_name, "{}")
            try:
                eval_data = json.loads(result_str.strip())
                score = eval_data.get('score', 'N/A')
                reason = eval_data.get('reason', '无理由')
                self.write(f"  {model_name}: 分数={score}, 理由={reason}\n")
            except:
                self.write(f"  {model_name}: 解析失败\n")
        
        # 记录投票评分
        self.write(f"投票评分: {score_final_voted}\n")
    
    def log_history_update(self, history_length):
        """记录历史更新信息"""
        self.write(f"历史记录已更新，长度: {history_length} 字符\n")
    
    def log_violation(self, turn_number):
        """记录违规检测"""
        self.write(f"\n❌ 检测到违规! 轮次: {turn_number}\n")
    
    def log_script_end(self, script_name, violation_occurred, broken_round, total_turns):
        """记录剧本结束信息"""
        self.write(f"\n{'='*50}\n")
        self.write(f"剧本执行完成: {script_name}\n")
        if violation_occurred:
            self.write(f"结果: 已攻破 (第{broken_round}轮)\n")
        else:
            self.write(f"结果: 未攻破 (共{total_turns}轮)\n")
        self.write(f"{'='*50}\n\n")
    
    def log_error(self, error_message):
        """记录错误信息"""
        self.write(f"\n⚠️ 错误: {error_message}\n")
    
    def log_summary(self, total_tests, completed_tests):
        """记录测试摘要"""
        self.write(f"\n📊 测试摘要: 已完成 {completed_tests}/{total_tests}\n")