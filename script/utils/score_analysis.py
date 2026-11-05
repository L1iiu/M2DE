import pandas as pd
import json
import os
import logging
import argparse

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_score_stats(csv_path, output_path, score_column="score_final_voted"):
    """分析评分统计数据并生成报告"""
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
    
    # 确保score_final_voted是字符串类型
    df[score_column] = df[score_column].astype(str)
    
    # 统计
    total = len(df)
    overall_stats = {
        "total": total,
        "scores": {}
    }
    
    # 评分范围
    scores = ['-1', '0', '1']
    
    for score in scores:
        count = df[df[score_column] == score].shape[0]
        percentage = round(count / total * 100, 2) if total > 0 else 0
        overall_stats["scores"][score] = {
            "count": int(count),
            "percentage": percentage
        }
    
    # 维度统计 - 修改为子维度统计
    dimension_stats = {}
    if "subdimension" in df.columns:
        dimensions = df['subdimension'].unique()
        for dim in dimensions:
            if pd.isna(dim) or dim == '':
                continue
                
            dim_df = df[df['subdimension'] == dim]
            dim_total = len(dim_df)
            
            dim_score_stats = {}
            for score in scores:
                count = dim_df[dim_df[score_column] == score].shape[0]
                percentage = round(count / dim_total * 100, 2) if dim_total > 0 else 0
                dim_score_stats[score] = {
                    "count": int(count),
                    "percentage": percentage
                }
            
            dimension_stats[str(dim)] = {
                "total": dim_total,
                "scores": dim_score_stats
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
        logger.info(f"评分统计结果已保存至: {output_path}")
    except Exception as e:
        logger.error(f"保存评分统计结果失败: {str(e)}", exc_info=True)
    
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='分析评分统计数据')
    parser.add_argument('--input', required=True, help='输入CSV文件路径')
    parser.add_argument('--output', required=True, help='输出JSON文件路径')
    parser.add_argument('--score_column', default='score_final_voted', 
                        help='得分列名称 (默认: score_final_voted)')
    
    args = parser.parse_args()
    
    analyze_score_stats(
        csv_path=args.input,
        output_path=args.output,
        score_column=args.score_column
    )