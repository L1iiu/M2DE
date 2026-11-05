import os
import pandas as pd
import json
from datetime import datetime
from tqdm import tqdm
from utils.logger_utils import DimensionLogger
from utils.csv_utils import initialize_csv, append_to_csv
from core.script_executor import ScriptExecutor
from core.dimension_manager import DimensionManager  # 从core目录导入
from utils.stop_handler import should_stop

def load_script_by_id(dimension, script_id):
    """根据剧本ID加载剧本 - 只从维度专用目录加载"""
    try:
        # 使用维度管理器获取剧本路径
        script_path = DimensionManager.get_script_path(dimension, script_id)
        
        if script_path and os.path.exists(script_path):
            print(f"📁 找到维度专用剧本: {script_path}")
            return load_json_script(script_path)
        
        print(f"❌ 未找到剧本文件: {script_id} 在维度 {dimension}")
        return None
        
    except Exception as e:
        print(f"❌ 加载剧本失败: {e}")
        return None

def load_json_script(json_path):
    """从JSON文件加载剧本"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            script_data = json.load(f)
            print(f"✅ 成功加载剧本: {script_data.get('name', '未知')}")
            return script_data
    except json.JSONDecodeError as e:
        print(f"❌ JSON格式错误: {e}")
        return None
    except Exception as e:
        print(f"❌ 加载JSON剧本失败: {e}")
        return None

def process_script_dimension(dimension, model, num_executions=1, model_identifier=None):
    """处理剧本模式的维度测试 - 完全使用维度专用模块"""
    print(f"\n🎬 开始剧本模式测试 - 维度: {dimension.upper()}，模型: {model}")
    print(f"📂 使用维度专用模块: {dimension}")
    
    # 读取核心目标数据
    data_path = f"data/{dimension}.csv"
    if not os.path.exists(data_path):
        print(f"❌ 错误: 维度 {dimension} 的数据文件不存在: {data_path}")
        return None
    
    try:
        # 修改这里：指定dtype，将script_ids列读取为字符串
        df = pd.read_csv(data_path, dtype={'script_ids': str})
        print(f"📊 加载了 {len(df)} 个核心目标")
    except Exception as e:
        print(f"❌ 读取数据文件失败: {e}")
        return None
    
    # 设置模型标识
    if not model_identifier:
        if model.startswith("ollama:"):
            model_spec = model[7:]
            if ":" in model_spec and "://" in model_spec:
                parts = model_spec.split(":", 1)
                ollama_model_name = parts[0]
                ollama_base_url = parts[1]
            else:
                ollama_model_name = model_spec
                ollama_base_url = "http://localhost:11434"
            model_identifier = f"ollama_{ollama_model_name.replace('-', '_').replace(':', '_')}"
        else:
            model_name = "glm45" if model == "glm-4.5" else "qwen" if model == "qwen-turbo" else "gpt" if model.startswith("gpt") else model
            model_identifier = model_name
    
    # 设置模型调用函数
    from api_call import get_model_call_func
    target_model_func = get_model_call_func(model)
    
    # 创建输出目录和文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"script_{dimension}_{model_identifier}_{timestamp}_results.csv"
    output_path = os.path.join("outputs", "result_csv", output_filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 初始化CSV
    initialize_csv(output_path)
    
    # 使用日志
    with DimensionLogger(f"script_{dimension}", model_identifier) as logger:
        logger.log_header()
        
        # 计算总测试数
        total_tests = 0
        test_plan = []
        
        for _, data_row in df.iterrows():
            data_id = data_row['id']
            core_goal = data_row['prompt']
            subdimension = data_row['subdimension']
            
            # 修改这里：正确处理script_ids，确保是字符串
            if pd.isna(data_row['script_ids']):
                script_ids = []
            else:
                # 直接使用字符串，保持原始格式
                script_ids_str = str(data_row['script_ids']).strip()
                script_ids = [script_id.strip() for script_id in script_ids_str.split(',')]
            
            print(f"📝 解析script_ids: '{data_row['script_ids']}' -> {script_ids}")  # 添加调试
            
            for script_id in script_ids:
                # 确保script_id是字符串并保持原样
                script_id = str(script_id).strip()
                for execution in range(num_executions):
                    test_plan.append({
                        'data_id': data_id,
                        'core_goal': core_goal,
                        'subdimension': subdimension,
                        'script_id': script_id,
                        'execution_id': execution
                    })
                    total_tests += 1
        
        print(f"📋 测试计划: {total_tests} 个测试用例")
        
        completed_tests = 0
        successful_tests = 0
        failed_tests = 0
        
        # 创建进度条
        pbar = tqdm(total=total_tests, desc="整体进度", position=0)
        
        # 执行测试计划
        for test in test_plan:
            if should_stop():
                logger.log_summary(total_tests, completed_tests)
                pbar.close()
                return output_path
            
            data_id = test['data_id']
            core_goal = test['core_goal']
            subdimension = test['subdimension']
            script_id = test['script_id']
            execution_id = test['execution_id']
            
            # 更新进度条描述
            pbar.set_description(f"测试 {data_id}-{script_id}")
            
            # 加载剧本（只从维度专用目录）
            script = load_script_by_id(dimension, script_id)
            if not script:
                print(f"⚠️ 警告: 剧本 {script_id} 加载失败，跳过")
                completed_tests += 1
                failed_tests += 1
                pbar.update(1)
                continue
            
            # 记录剧本开始
            logger.log_script_start(script.get('name', script_id), core_goal, data_id, execution_id)
            
            try:
                # 创建执行器（使用维度专用模块）
                executor = ScriptExecutor(dimension, target_model_func, logger)
                
                # 执行剧本
                result = executor.execute_script(script, core_goal, data_id, execution_id, subdimension)
                
                # 记录剧本结束
                logger.log_script_end(
                    script.get('name', script_id), 
                    result["violation_occurred"], 
                    result["broken_round"],
                    len(result["dialogue_history"])
                )
                
                # 保存结果
                append_to_csv(output_path, result)
                
                if result["violation_occurred"]:
                    successful_tests += 1
                    pbar.write(f"✅ 攻破: {script.get('name', script_id)} (数据ID: {data_id})")
                else:
                    pbar.write(f"➖ 未攻破: {script.get('name', script_id)} (数据ID: {data_id})")
                
            except Exception as e:
                failed_tests += 1
                pbar.write(f"❌ 执行异常: {script.get('name', script_id)} - {str(e)}")
                logger.log_error(f"剧本执行异常: {str(e)}")
            
            completed_tests += 1
            pbar.update(1)
            
            # 更新进度条后缀
            pbar.set_postfix({
                '成功': successful_tests,
                '失败': failed_tests,
                '完成率': f"{completed_tests/total_tests*100:.1f}%"
            })
        
        pbar.close()
        
        # 输出最终统计
        logger.log_summary(total_tests, completed_tests)
        print(f"\n📈 测试完成统计:")
        print(f"  总测试数: {total_tests}")
        print(f"  成功攻破: {successful_tests}")
        print(f"  未攻破: {total_tests - successful_tests - failed_tests}")
        print(f"  失败: {failed_tests}")
        print(f"  成功率: {successful_tests/total_tests*100:.1f}%" if total_tests > 0 else "  成功率: 0%")
        
        logger.write(f"\n🎉 剧本模式测试完成! 结果保存至: {output_path}\n")
        print(f"💾 结果保存至: {output_path}")
    
    return output_path