import os
import sys
import time
import logging  

class DimensionLogger:
    """维度测试日志记录器"""
    def __init__(self, dimension, model_name):
        # 保存维度和模型名称
        self.dimension = dimension
        self.model_name = model_name
        
        # 动态创建日志目录
        self.log_dir = os.path.join("logs", "main_logs", f"{dimension}_{model_name}")
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 创建时间戳用于日志文件名
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.log_filename = os.path.join(self.log_dir, f"{dimension}_test_{timestamp}.log")
        
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
    
    def log_header(self, id=None):
        """记录测试头部信息"""
        self.write(f"\n{'='*50}\n")
        self.write(f"开始 {self.dimension.upper()} 维度测试\n")
        self.write(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.write(f"使用模型: {self.model_name}\n")
        if id:
            self.write(f"测试ID: {id}\n")
        self.write(f"日志文件: {self.log_filename}\n")
        self.write(f"{'='*50}\n\n")
    
    def log_round(self, round_idx, question, answer, score=None):
        """记录轮次信息"""
        self.write(f"\n=== 第 {round_idx} 轮 ===")
        self.write(f"\n问题: {question}")
        self.write(f"\n模型回答: {answer}")
        if score is not None:
            self.write(f"\n评估结果: 得分={score}")
        self.write("\n")