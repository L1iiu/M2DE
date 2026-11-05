import pandas as pd
import json
import os
import logging
import numpy as np
import argparse  # 添加参数解析模块

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
    
    # 处理break_round列
    # 将"None"字符串转换为真正的None，数值转换为整数
    df['break_round'] = df['break_round'].apply(
        lambda x: None if x == "None" or pd.isna(x) else int(float(x)) if isinstance(x, (int, float)) or (isinstance(x, str) and x.replace('.', '', 1).isdigit()) else None
    )
    
    # 1. 一级维度统计
    total = len(df)
    
    # 统计各轮次攻破数量和比例
    break_rounds = list(range(0, 11))  # 0到10轮
    break_rounds.append("None")  # 添加未攻破的情况
    
    round_stats = {}
    for rnd in break_rounds:
        if rnd == "None":
            count = df['break_round'].isna().sum()
        else:
            count = (df['break_round'] == rnd).sum()
        
        percentage = round(count / total * 100, 2) if total > 0 else 0
        round_stats[str(rnd)] = {
            "count": int(count),
            "percentage": percentage
        }
    
    # 计算平均攻破轮次（仅计算被攻破的记录）
    broken_df = df[df['break_round'].notna()]
    avg_break_round = round(broken_df['break_round'].mean(), 2) if not broken_df.empty else 0
    
    overall_stats = {
        "total": total,
        "average_break_round": avg_break_round,
        "round_distribution": round_stats
    }
    
    # 2. 子维度统计
    sub_stats = {}
    if "subdimension" in df.columns:
        categories = df['subdimension'].unique()
        for cat in categories:
            if pd.isna(cat) or cat == '':
                continue
                
            cat_df = df[df['subdimension'] == cat]
            cat_total = len(cat_df)
            
            # 子维度的轮次分布
            cat_round_stats = {}
            for rnd in break_rounds:
                if rnd == "None":
                    count = cat_df['break_round'].isna().sum()
                else:
                    count = (cat_df['break_round'] == rnd).sum()
                
                percentage = round(count / cat_total * 100, 2) if cat_total > 0 else 0
                cat_round_stats[str(rnd)] = {
                    "count": int(count),
                    "percentage": percentage
                }
            
            # 子维度的平均攻破轮次
            cat_broken_df = cat_df[cat_df['break_round'].notna()]
            cat_avg_break_round = round(cat_broken_df['break_round'].mean(), 2) if not cat_broken_df.empty else 0
            
            sub_stats[str(cat)] = {
                "total": cat_total,
                "average_break_round": cat_avg_break_round,
                "round_distribution": cat_round_stats
            }
    
    # 生成结果字典
    result = {
        "overall": overall_stats,
        "subcategories": sub_stats
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