import pandas as pd
import random
import re

# 文件路径
input_file = "dependency/data/advbench/test-00000-of-00001.parquet"  # 替换为你的输入文件路径
output_file = "regular.csv"    # 输出文件路径

# 读取文件
df = pd.read_parquet(input_file)

# 提取需要的字段 question（提问）和 answer（回答）
qa_pairs = df[['question', 'answer']]

# 定义一个函数，用于移除字符串中的所有符号
def remove_symbols(text):
    # 正则表达式：只保留字母、数字和空格
    return re.sub(r'[^a-zA-Z0-9\s]', '', text)

# 对 question 和 answer 应用清理
qa_pairs['question'] = qa_pairs['question'].apply(remove_symbols)
qa_pairs['answer'] = qa_pairs['answer'].apply(remove_symbols)

# 随机打乱数据
qa_pairs = qa_pairs.sample(frac=1, random_state=42).reset_index(drop=True)

# 选取前 100 个问答对
qa_pairs = qa_pairs.head(100)

# 转换为目标格式：列名为 goal 和 target，确保格式正确
qa_pairs['goal'] = qa_pairs['question']
qa_pairs['target'] = qa_pairs['answer'].apply(lambda x: f'"{x}"')

# 只保留目标列并重命名
final_data = qa_pairs[['goal', 'target']]

# 保存为新的 CSV 文件
final_data.to_csv(output_file, index=False)

print(f"处理完成！已将文件保存为 {output_file}")