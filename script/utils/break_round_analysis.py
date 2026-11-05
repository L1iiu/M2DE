import pandas as pd
import json
import os
import logging
import numpy as np
import argparse

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_break_round_stats(csv_path, output_path):
    """分析攻破轮次统计数据并生成报告"""
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if not os.path.exists(csv_path):
        logger.warning(f"结果文件不存在: {csv_path}")
        return
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        logger.error(f"读取CSV文件失败: {e}", exc_info=True)
        return
    
    # 处理broken_round列
    df['broken_round'] = df['broken_round'].apply(
        lambda x: None if x == "None" or pd.isna(x) else int(float(x)) if isinstance(x, (int, float)) or (isinstance(x, str) and x.replace('.', '', 1).isdigit()) else None
    )
    
    # 1. 总体统计
    total = len(df)
    
    # 统计各轮次攻破数量和比例（修改为1-8）
    break_rounds = list(range(1, 9))  # 1到8轮
    break_rounds.append("None")  # 添加未攻破的情况
    
    round_stats = {}
    for rnd in break_rounds:
        if rnd == "None":
            count = df['broken_round'].isna().sum()
        else:
            count = (df['broken_round'] == rnd).sum()
        
        percentage = round(count / total * 100, 2) if total > 0 else 0
        round_stats[str(rnd)] = {
            "count": int(count),
            "percentage": percentage
        }
    
    # 计算平均攻破轮次（仅计算被攻破的记录）
    broken_df = df[df['broken_round'].notna()]
    avg_break_round = round(broken_df['broken_round'].mean(), 2) if not broken_df.empty else 0
    
    # 计算攻破率
    break_rate = round(len(broken_df) / total * 100, 2) if total > 0 else 0
    
    overall_stats = {
        "total": total,
        "break_rate": break_rate,
        "average_break_round": avg_break_round,
        "round_distribution": round_stats
    }
    
    # 2. 维度统计 - 修改为子维度统计
    dimension_stats = {}
    if "subdimension" in df.columns:
        dimensions = df['subdimension'].unique()
        for dim in dimensions:
            if pd.isna(dim) or dim == '':
                continue
                
            dim_df = df[df['subdimension'] == dim]
            dim_total = len(dim_df)
            
            # 维度的轮次分布
            dim_round_stats = {}
            for rnd in break_rounds:
                if rnd == "None":
                    count = dim_df['broken_round'].isna().sum()
                else:
                    count = (dim_df['broken_round'] == rnd).sum()
                
                percentage = round(count / dim_total * 100, 2) if dim_total > 0 else 0
                dim_round_stats[str(rnd)] = {
                    "count": int(count),
                    "percentage": percentage
                }
            
            # 维度的平均攻破轮次
            dim_broken_df = dim_df[dim_df['broken_round'].notna()]
            dim_avg_break_round = round(dim_broken_df['broken_round'].mean(), 2) if not dim_broken_df.empty else 0
            
            # 维度的攻破率
            dim_break_rate = round(len(dim_broken_df) / dim_total * 100, 2) if dim_total > 0 else 0
            
            dimension_stats[str(dim)] = {
                "total": dim_total,
                "break_rate": dim_break_rate,
                "average_break_round": dim_avg_break_round,
                "round_distribution": dim_round_stats
            }
    
    # 生成结果字典
    result = {
        "overall": overall_stats,
        "subdimensions": dimension_stats
    }
    
    # 保存统计结果
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logger.info(f"攻破轮次统计结果已保存至: {output_path}")
    except Exception as e:
        logger.error(f"保存攻破轮次统计结果失败: {str(e)}", exc_info=True)
    
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='分析攻破轮次统计数据')
    parser.add_argument('--input', required=True, help='输入CSV文件路径')
    parser.add_argument('--output', required=True, help='输出JSON文件路径')
    
    args = parser.parse_args()
    
    analyze_break_round_stats(
        csv_path=args.input,
        output_path=args.output
    )