import pandas as pd
import pickle,os
path="results/hyper/2025-05-30_15-17-34_53/adv_library.pkl.csv"
adv_library="results/hyper/2025-05-30_15-17-34_53/adv_library.pkl"
from dependency.llm_attacks.minimal_gcg.opt_utils import *
directory=os.getenv("ADV_LIBRARY_DIR", "./adv_library")
libs_all_file=[adv_library]

all_files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

# 过滤掉CSV文件
# non_csv_files = [f for f in all_files if not f.lower().endswith('.csv')]
# # 将非CSV文件逐个送入处理函数
# for file_name in non_csv_files:
#     file_path = os.path.join(directory, file_name)
#     libs_all_file.append(file_path)

# 读取CSV文件，从第二行开始（跳过第1行）
df = pd.read_csv(path)
data_list = df.values.tolist()
def calc_res(data):
    #print(f"数据条数{len(data)}")
    number_of_ac=0
    number_of_testing_ok=0
    number_of_attacking_ok=0
    run_iters=0
    ac_num=0
    for current,i in enumerate(data):
        #if(current<=200): continue
        if(i[2]==True):
            number_of_ac+=1
            if i[6]==True:
                number_of_testing_ok+=1
        if(i[7]!=0 and i[6]==False and i[2]==True):
            number_of_attacking_ok+=1
        if(i[7]!=0):
            run_iters+=i[7]
            ac_num+=1
        
    return [number_of_ac/len(data),number_of_testing_ok/len(data),(number_of_attacking_ok/ac_num),run_iters/ac_num]

def split_and_process(data, num_chunks=20):
    total_length = len(data)
    chunk_size = total_length // num_chunks
    remainder = total_length % num_chunks
    
    start = 0
    for i in range(num_chunks):
        # 确保最后一块包含剩余的元素（如果有）
        end = start + chunk_size + (1 if i < remainder else 0)
        chunk = data[start:end]
        
        # 调用ana()函数处理当前块
        ans=calc_res(chunk)
        for j in ans:
            print(j,end=" ")
        
        start = end
        print("")
# 分割数据并调用ana()函数处理每个块
res=[]
split_and_process(data_list[:], num_chunks=10)

for file in libs_all_file:
    adv_library=file
    f=open(adv_library,"rb")
    adv_list=pickle.load(f)
    avg_tae=0
    data_len=len(data_list)
    top_tae=len(adv_list[0]["configs"])/520
    model_path=adv_list[0]["configs"][0]["model_path"]
    model,tokenizer=load_model_and_tokenizer(model_path)
    token_ana={}
    for i in adv_list:
        avg_tae+=len(i["configs"])
        for j in i["adv_token"]:
            if j.item() not in token_ana:
                token_ana[j.item()]=len(i["configs"])
            else:
                token_ana[j.item()]+=len(i["configs"])
    sorted_token_ana = sorted(token_ana.items(), key=lambda item: item[1], reverse=True)
    avg_tae/=((data_len)*len(adv_list))
    print(top_tae,avg_tae)
    for token,frequency in sorted_token_ana[:20]:
        print(tokenizer.decode(token).strip(),frequency)