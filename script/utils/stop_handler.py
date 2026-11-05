"""
停止信号处理器 - 处理键盘中断信号 (Ctrl+C)
"""

import signal
import sys

# 全局停止标志
STOP_REQUESTED = False

def signal_handler(sig, frame):
    """处理中断信号"""
    global STOP_REQUESTED
    STOP_REQUESTED = True
    # 确保在终端显示停止提示
    sys.stderr.write("\n收到停止信号，正在停止评估...\n")
    sys.stderr.flush()

def should_stop():
    """检查是否请求停止"""
    return STOP_REQUESTED

def setup_stop_handlers():
    """设置停止处理器"""
    # 注册信号处理
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    return STOP_REQUESTED