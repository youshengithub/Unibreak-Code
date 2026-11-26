import os
import sys
import pickle
import yaml
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch

from GA_based_methods.Model_manager import model_manager
from GA_based_methods.Fitness_calcution import fitness_calcution
from GA_based_methods.Adv_libraray import adv_library

def process_directory(dir_path, csv_path, n_goals=10, n_adv=10, m_queries=5):
    """处理单个目录的函数"""
    config_path = os.path.join(dir_path, "attack_config.yaml")
    adv_library_path = os.path.join(dir_path, "adv_library.pkl")
    save_path = os.path.join(dir_path, "jailbroken_rates_heatmap.csv")
    
    # 检查必要文件是否存在
    if not os.path.exists(config_path):
        print(f"Config file not found in {dir_path}, skipping...")
        return None
    if not os.path.exists(adv_library_path):
        print(f"Adv library file not found in {dir_path}, skipping...")
        return None
    
    # 加载配置文件
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 初始化模型
    model = model_manager(
        config['model_path'], 
        config['model_type'], 
        config['device'], 
        config['do_sample']
    )
    
    # 加载对抗后缀库
    vocab_size = len(model.tokenizer)  # 使用模型的分词器获取词汇表大小
    enable_library = config.get('enable_library', True)
    device = config['device']
    
    adv_lib = adv_library(
        adv_library_path, 
        vocab_size, 
        model.tokenizer,  # 使用模型的分词器
        enable_library, 
        device
    )
    
    # 读取无害行为CSV文件
    df = pd.read_csv(csv_path)
    
    # 获取前n个目标
    goals = df['goal'].head(n_goals).tolist()
    
    # 获取前n个对抗后缀字符串
    adv_strings = [i["adv_str"] for i in adv_lib.adv_list[:n_adv]]
    
    # 创建结果矩阵 - 行对应adv，列对应goal
    results = np.zeros((n_adv, n_goals))
    
    # 对每个对抗后缀和目标组合进行查询
    for i, adv_str in enumerate(tqdm(adv_strings, desc=f"Processing adv strings in {os.path.basename(dir_path)}")):
        for j, goal in enumerate(tqdm(goals, desc="Processing goals", leave=False)):
            jailbroken_count = 0
            
            # 构造输入文本
            input_text = goal + " " + adv_str
            input_ids = model.tokenizer.encode(input_text, return_tensors="pt").to(device)
            
            # 查询模型m次
            for _ in range(m_queries):
                response = model.get_output_one_by_one(input_ids, max_length=300)
                
                # 检查是否攻击成功
                if fitness_calcution.check_for_attack_success_for_text(response):
                    jailbroken_count += 1
            
            # 计算jailbroken率
            jailbroken_rate = jailbroken_count / m_queries
            results[i, j] = jailbroken_rate
    
            # 创建结果DataFrame - 行对应adv，列对应goal
            result_df = pd.DataFrame(
                results,
                index=[f"Adv_{i+1}" for i in range(n_adv)],
                columns=[f"Goal_{j+1}" for j in range(n_goals)]
            )
            
            # 添加对抗后缀字符串作为额外列
            #result_df['Adv_Text'] = adv_strings
            
            # 保存结果到CSV
            result_df.to_csv(save_path, index=False)
    print(f"Results saved to {save_path}")
    
    return result_df

def main():
    # 参数设置
    base_dir = "results/hotmap/"  # 替换为你的基础目录
    csv_path = os.getenv("UNHARMFUL_CSV", "dependency/data/advbench/unharmful_behaviors.csv")
    
    # 设置处理参数
    n_goals = 10
    n_adv = 10
    m_queries = 5
    
    # 获取所有子目录
    sub_dirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) 
                if os.path.isdir(os.path.join(base_dir, d))]
    
    # 处理每个子目录
    all_results = {}
    for dir_path in sub_dirs:
        print(f"\nProcessing directory: {dir_path}")
        result = process_directory(dir_path, csv_path, n_goals, n_adv, m_queries)
        if result is not None:
            all_results[os.path.basename(dir_path)] = result
    
    return all_results

if __name__ == "__main__":
    all_results = main()