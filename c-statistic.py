from scipy.stats import friedmanchisquare, rankdata
import scikit_posthocs as sp
import pandas as pd
import numpy as np

metadata=np.array(
    [
[33.96,	7.34,	3.42,	5.23,	2.34],
[33.96,	7.34,	3.42,	5.23,	2.34],
[33.96,	7.34,	3.42,	5.23,	2.34],


# [96.23,81.45,62.26,54.35,96.23,],
# [97.3,81.45,62.26,54.35,91.23,],
# [98.23,81.45,62.26,54.35,86.23,],


]


)
# metadata=metadata.T


# for i in range(metadata.shape[0]):
# 示例数据：假设第一个算法是"Ours"，其他为Baseline
    # data = np.array([
    #     [0.9, 0.8, 0.85, 0.7],   # 数据集1
    #     [0.85, 0.8, 0.8, 0.75],  # 数据集2
    #     [0.8, 0.75, 0.7, 0.65],  # 数据集3
    #     [0.85, 0.7, 0.75, 0.6],  # 数据集4
    #     [0.9, 0.85, 0.8, 0.7]    # 数据集5
    # ])
data=metadata
# 自定义算法名称（假设第一个是"Ours"）
algo_names = ['Ours']+["Baseline"+str(i) for i in range(1, 5)]

# Friedman检验
stat, p_value = friedmanchisquare(*data.T)
print(f"Friedman检验统计量: {stat}, p值: {p_value}\n")

if p_value < 0.05:
    # 计算排名（确保数值越大排名越高）
    ranks = np.apply_along_axis(lambda x: rankdata(-x), 1, data)
    mean_ranks = np.mean(ranks, axis=0)
    
    # 转换为长格式数据
    long_data = []
    for dataset_idx in range(data.shape[0]):
        for algo_idx in range(data.shape[1]):
            long_data.append({
                'Algorithm': algo_names[algo_idx],
                'Value': data[dataset_idx, algo_idx]
            })
    df = pd.DataFrame(long_data)
    
    # Conover检验获取p值矩阵
    p_matrix = sp.posthoc_conover(df, val_col='Value', group_col='Algorithm', sort=True)
    
    # 提取"Ours"与其他算法的p值
    p_values = []
    for algo in algo_names[1:]:  # 跳过"Ours"自身
        p = p_matrix.loc['Ours', algo]
        p_values.append(p)
    
    # 构建结果表格
    result_df = pd.DataFrame({
        'Algorithm': algo_names,
        'Average Rank': mean_ranks,
        'p-value (vs Ours)': ['-'] + [f"{p:.10f}" for p in p_values]
    })
    
    # 打印表格
    print("统计结果表格:")
    print(result_df.to_string(index=False))