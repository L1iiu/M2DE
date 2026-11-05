import os
import sys
import argparse
import logging
from datetime import datetime
from core.dimension_processor import process_single_dimension
from core.eval_utils import DIMENSION_CONFIG
from utils.stop_handler import setup_stop_handlers
from utils.stop_handler import should_stop


# 支持的模型列表
CLOUD_MODELS = [
    "qwen-turbo",
    "glm-4.5",
    "gpt-3.5-turbo",
    "gpt-4.1",
    "gpt-5",
]

OLLAMA_MODELS = [
    "ollama:llama2:7b",
    "ollama:llama2:13b",
    "ollama:llama3:8b",
    "ollama:mistral:7b",
]


def setup_logger() -> tuple[logging.Logger, str]:
    """配置日志系统，只输出到文件，返回 logger 和日志文件路径。"""
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join("logs", f"evaluation_{timestamp}.log")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 清理可能已有的 handler，避免重复输出
    if logger.handlers:
        logger.handlers.clear()

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )

    logger.addHandler(file_handler)
    return logger, log_file


def get_api_key(model: str) -> str | None:
    """根据模型名称获取对应的 API 密钥。"""
    if model == "qwen-turbo":
        return os.getenv("DASHSCOPE_API_KEY")
    if model == "glm-4.5":
        return os.getenv("OPEN_BIGMODEL_API_KEY")
    if model.startswith("gpt"):
        return os.getenv("OPENAI_API_KEY")
    return None


def print_supported_models() -> None:
    """打印支持的模型列表。"""
    print("Supported models:")
    print("  - qwen-turbo, glm-4.5")
    print("  - gpt-3.5-turbo, gpt-4.1, gpt-5")
    print("  - ollama:llama2:7b, ollama:llama2:13b, ollama:llama3:8b, ollama:mistral:7b")


def parse_args() -> argparse.Namespace:
    """命令行参数解析。"""
    parser = argparse.ArgumentParser(description="大模型动态测评工具")

    parser.add_argument(
        "-d",
        "--dimensions",
        nargs="+",
        choices=list(DIMENSION_CONFIG.keys()),
        required=True,
        help="选择测评维度（可多个，用空格分隔）",
    )

    parser.add_argument(
        "-m",
        "--models",
        nargs="+",
        type=str,
        required=True,
        help=(
            "Select model(s) to use (multiple allowed). Supported: "
            "qwen-turbo, glm-4.5, "
            "gpt-3.5-turbo, gpt-4.1, gpt-5, "
            "ollama:llama2:7b, ollama:llama2:13b, "
            "ollama:llama3:8b, ollama:mistral:7b"
        ),
    )

    return parser.parse_args()


def validate_models(models: list[str]) -> bool:
    """检查模型是否受支持，打印错误信息，返回是否全部合法。"""
    unsupported_models = [
        m for m in models if (m not in CLOUD_MODELS and m not in OLLAMA_MODELS)
    ]

    if not unsupported_models:
        return True

    for m in unsupported_models:
        print(f"Error: Unsupported model '{m}'")
    print_supported_models()
    return False


def check_api_keys(models: list[str]) -> bool:
    """检查云端模型是否已配置所需 API Key。"""
    missing_key_models: list[str] = []

    for model in models:
        if model.startswith("ollama:"):
            continue
        api_key = get_api_key(model)
        if not api_key:
            missing_key_models.append(model)

    if not missing_key_models:
        return True

    print("错误: 以下模型需要设置相应的环境变量")
    for m in missing_key_models:
        print(f"  - {m}")
        if m == "qwen-turbo":
            print("    请设置 DASHSCOPE_API_KEY 环境变量")
        elif m == "glm-4.5":
            print("    请设置 OPEN_BIGMODEL_API_KEY 环境变量")
        elif m.startswith("gpt"):
            print("    请设置 OPENAI_API_KEY 环境变量")
    return False


def main() -> None:
    logger, log_file = setup_logger()
    start_time = datetime.now()
    logger.info("评估开始时间: %s", start_time)

    args = parse_args()

    # 模型合法性校验
    if not validate_models(args.models):
        sys.exit(1)

    # API Key 检查（仅云端模型）
    if not check_api_keys(args.models):
        sys.exit(1)

    # 处理 Ctrl+C / 终止信号
    setup_stop_handlers()

    logger.info("评估维度: %s", args.dimensions)
    logger.info("评估模型列表: %s", args.models)

    os.makedirs("outputs", exist_ok=True)

    try:
        total_dimensions = len(args.dimensions)
        for idx, dimension in enumerate(args.dimensions, start=1):
            if should_stop():
                logger.info("停止请求已收到，跳过剩余维度处理")
                break

            logger.info("开始处理维度 %d/%d: %s", idx, total_dimensions, dimension.upper())
            print(f"\n处理维度 {idx}/{total_dimensions}: {dimension.upper()}")

            try:
                dim_start = datetime.now()
                logger.info("开始处理维度: %s", dimension.upper())

                # 当前维度下依次评测所有模型
                for model in args.models:
                    logger.info("开始处理模型 %s 在维度 %s 上的评估", model, dimension.upper())
                    print(f"  -> 模型 {model} 测试中...")
                    process_single_dimension(dimension, model)

                dim_end = datetime.now()
                duration = dim_end - dim_start
                logger.info("维度 %s 处理完成, 耗时: %s", dimension.upper(), duration)
                print(f"维度 {dimension.upper()} 完成, 耗时: {duration}")

            except Exception as e:  # noqa: BLE001
                dim_end = datetime.now()
                duration = dim_end - dim_start
                logger.error(
                    "维度 %s 处理失败, 耗时: %s, 错误: %s",
                    dimension.upper(),
                    duration,
                    str(e),
                    exc_info=True,
                )
                print(f"维度 {dimension.upper()} 处理失败: {str(e)}")

                if idx < total_dimensions:
                    continue_input = input("是否继续处理剩余维度? (y/n): ").strip().lower()
                    if continue_input != "y":
                        logger.info("用户选择停止处理剩余维度")
                        break

    except Exception as e:  # noqa: BLE001
        logger.error("评估过程发生未知错误: %s", str(e), exc_info=True)
        print(f"评估过程发生错误: {str(e)}")

    finally:
        end_time = datetime.now()
        total_duration = end_time - start_time
        logger.info("评估结束时间: %s", end_time)
        logger.info("总耗时: %s", total_duration)
        logger.info("评估完成")
        logger.info("日志文件: %s", log_file)

        print("\n评估完成！结果保存在 outputs/ 目录")
        print(f"日志文件: {log_file}")


if __name__ == "__main__":
    main()