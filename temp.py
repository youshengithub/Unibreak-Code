import pandas as pd
import re
import csv
import os

def extract_training_data(input_file, output_file=None):
    """
    从原始文件中提取训练数据
    
    参数:
        input_file: 输入文件路径
        output_file: 输出CSV文件路径（可选）
    
    返回:
        DataFrame: 包含训练数据的数据框
    """
    # 读取输入文件
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 使用正则表达式匹配每个记录
    pattern = r"Flag: (True|False)\s+Input: (.*?)\s+Ans: (.*?)(?=Flag:|pattern:|\Z)"
    matches = re.findall(pattern, content, re.DOTALL)
    
    # 处理所有记录
    training_data = []
    for match in matches:
        flag, query, answer = match
        
        # 清理文本
        query = clean_query(query.strip())
        answer = answer.strip()
        
        # 构造对话格式
        conversation_text = f"Human: {query}\nAssistant: {answer}"
        
        # 设置标签：True表示回答了问题(标签为1)，False表示拒绝回答(标签为0)
        label = 1 if flag == "True" else 0
        
        training_data.append({
            "text": conversation_text,
            "label": label,
            "query": query,
            "answer": answer,
            "flag": flag
        })
    
    # 创建DataFrame
    df = pd.DataFrame(training_data)
    
    # 保存到CSV文件
    if output_file:
        df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"成功生成训练数据集，共 {len(df)} 条记录，保存到 {output_file}")
    
    return df

def clean_query(query):
    """
    清理查询文本，移除乱码
    
    参数:
        query: 原始查询文本
    
    返回:
        str: 清理后的查询文本
    """
    # 移除常见的乱码模式
    query = re.sub(r'\]==-paperefa\(-.*', '', query)
    query = re.sub(r'\(id bells freaking.*', '', query)
    query = re.sub(r'scre_\)commercial.*', '', query)
    query = re.sub(r'\(\)\s*\d+.*', '', query)
    
    # 标准化空白字符
    query = re.sub(r'\s+', ' ', query)
    
    return query.strip()

def extract_harmful_only(input_file, output_file):
    """
    只提取有害查询（Flag为True的记录）
    
    参数:
        input_file: 输入文件路径
        output_file: 输出CSV文件路径
    """
    # 读取输入文件
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 使用正则表达式匹配每个记录
    pattern = r"Flag: (True|False)\s+Input: (.*?)\s+Ans: (.*?)(?=Flag:|pattern:|\Z)"
    matches = re.findall(pattern, content, re.DOTALL)
    
    # 提取Flag为True的记录
    harmful_data = []
    for match in matches:
        flag, query, answer = match
        if flag == "True":
            # 清理查询文本
            query = clean_query(query.strip())
            answer = answer.strip()
            
            # 构造对话格式
            conversation_text = f"Human: {query}\nAssistant: {answer}"
            
            harmful_data.append({
                "text": conversation_text,
                "label": 1,  # 有害查询标签为1
                "query": query,
                "answer": answer
            })
    
    # 保存到CSV文件
    df = pd.DataFrame(harmful_data)
    df.to_csv(output_file, index=False, encoding='utf-8')
    
    print(f"成功提取 {len(harmful_data)} 个有害查询到 {output_file}")
    return df

def analyze_extracted_data(df):
    """
    分析提取的数据
    
    参数:
        df: 数据集DataFrame
    """
    print("\n数据集分析:")
    print(f"总记录数: {len(df)}")
    
    if 'label' in df.columns:
        print(f"标签分布:")
        label_counts = df['label'].value_counts()
        for label, count in label_counts.items():
            label_name = "有害回答" if label == 1 else "拒绝回答"
            print(f"  - {label_name} (label={label}): {count} ({count/len(df)*100:.1f}%)")
    
    print(f"\n前3个样本:")
    print("="*60)
    for i, row in df.head(3).iterrows():
        print(f"样本 {i+1}:")
        if 'label' in row:
            print(f"标签: {row['label']}")
        print(f"查询: {row['query'][:100]}...")
        print(f"回答: {row['answer'][:100]}...")
        print("-"*40)

# 使用示例
if __name__ == "__main__":
    import sys
    input_file = sys.argv[1] if len(sys.argv) > 1 else os.getenv("INPUT_LOG", "results/export/log/z_details_output.log")  # 你的原始数据文件
    
    # 方法1: 提取所有数据（包含True和False的记录）
    print("提取所有训练数据...")
    all_data = extract_training_data(input_file, "complete_training_dataset.csv")
    analyze_extracted_data(all_data)
    
