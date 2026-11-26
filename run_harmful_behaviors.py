import random
import warnings,torch,csv,copy
import numpy as np
import os,yaml,sys
from pprint import pprint
os.environ.setdefault("DASHSCOPE_API_KEY", os.getenv("DASHSCOPE_API_KEY", ""))

np.random.seed(20) #fix the random seed, so that the results are reproducible
torch.manual_seed(20)
torch.cuda.manual_seed_all(20)
warnings.filterwarnings("ignore", category=UserWarning)
from GA_based_methods.Utils import get_attack_configs
from GA_based_methods.Attack_manager import attacker_manager
from datetime import datetime
import argparse,pickle

parser = argparse.ArgumentParser()
parser.add_argument('--target', type=str, default="behaviors_original")
parser.add_argument('--version', type=str, default="full",choices=["full","no_initial","no_gradient_evolve","no_crossover","no_mutation","gray-box","black-box","no_library"])
parser.add_argument('--iteration', type=int, default=50)
parser.add_argument('--power', type=int, default=10)
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--model_path', type=str,default="models/qwen7b")
parser.add_argument('--exp_type', type=str,default="test")
#添加这三个参数
#攻击类型
parser.add_argument('--attack_type', type=str, default="suffix",choices=["suffix","prefix","insert"])
parser.add_argument('--population_size', type=int, default=10)
parser.add_argument('--crossover_num', type=int, default=5)
parser.add_argument('--mutation_rate', type=float, default=0.3)


#需要改一下逻辑了！
args = parser.parse_args()
harmful_behavior_path=f"dependency/data/advbench/harmful_{args.target}.csv"
#harmful_behavior_path=f"./harmful_{args.target}s.csv"
attack_template_path="configs/attack_conf.yaml"

#创建一个文件前缀 用exp_type results/exp_type
random_prix=str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))+f"_{random.randint(0,100)}"
res_file_preix=f"results/{args.exp_type}/{random_prix}"
# lib_file_preix=f"adv_library/{args.exp_type}/{random_prix}"
if not os.path.exists(res_file_preix):
    os.makedirs(res_file_preix)
result_file_path=f'{res_file_preix}/output.csv'
adv_library_path=f'{res_file_preix}/adv_library.pkl'

sys.stdout = open(f'{res_file_preix}/z_details_output.log', 'w', encoding='utf-8')

attack_template=get_attack_configs(attack_template_path)
attack_template["tensorboard_log"]=False
attack_template["use_gradient"]=True
def write_reults(results):
    with open(result_file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(results)
if __name__ == '__main__':
    try:
        at=attacker_manager()
        testing_succ_num=0
        avrage_iters=0
        asr=0
        assr=0
        avg_runing_time=0
        start_time = datetime.now()
        with open(harmful_behavior_path, newline='') as csvfile:
            csvreader = csv.reader(csvfile)
            header = next(csvreader)
            results=[]
            metrics={"testing_succ_num":0,"asr":0,"assr":0,"avg_iters":0,"avg_tae":0,"top_tae":0,"z_avg_runing_time":[],"z_asr":[],"z_tssr":[],"z_assr":[],"z_anai":[],"z_avg_tae":[],"z_top_tae":[]}

            for id,row in enumerate(csvreader):
                attack_template2=copy.deepcopy(attack_template)
                attack_template2["model_path"]=args.model_path
                if("ama" in attack_template2["model_path"] or "elm" in attack_template2["model_path"]):
                    attack_template2["model_type"]="llama-2"
                elif("wen" in attack_template2["model_path"]):
                    attack_template2["model_type"]="qwen"
                elif("model" in attack_template2["model_path"]):
                    attack_template2["model_type"]="llama-3"
                else:
                    print(attack_template2["model_path"])
                    attack_template2["model_type"]="llama-3"
                    # assert 1==0
                if(args.target=="behavior"):
                    attack_template2["input"]=row[0]
                    attack_template2["except_output"]=row[1]
                else:
                    attack_template2["input"]=""
                    attack_template2["except_output"]=row[0]
                attack_template2["device"]=f"cuda:{args.device}"
                attack_template2["iters"]=args.iteration
                attack_template2["per_length"]=args.power
                attack_template2["adv_library"]=adv_library_path
                attack_template2["population_size"]=args.population_size
                attack_template2["crossover_num"]=args.crossover_num
                attack_template2["mutation_rate"]=args.mutation_rate
                attack_template2["attack_type"]=args.attack_type
                if(args.version=="no_initial"):
                    attack_template2["enable_library"]=False
                elif (args.version=="no_gradient_evolve"):
                    attack_template2["use_gradient"]=False
                elif (args.version=="no_crossover"):
                    attack_template2["enable_crossover_sensitivity"]=False
                elif (args.version=="no_mutation"):  
                    attack_template2["enable_mutation_sensitivity"]=False
                elif (args.version=="no_sample"):  
                    attack_template2["enable_adv_sample"]=False
                elif (args.version=="no_library"):
                    attack_template2["enable_library"]=False
                elif (args.version=="gray-box"):  
                        attack_template2["use_gradient"]=False
                        attack_template2["enable_crossover_sensitivity"]=False
                elif (args.version=="black-box"):  
                        attack_template2["use_gradient"]=False
                        attack_template2["enable_crossover_sensitivity"]=False
                        attack_template2["enable_calculate_fitness"]=False
                pprint(attack_template2)
                is_success,score,input,output,is_testing_ok,run_iters=at.attack(configs=attack_template2)
                testing_succ_num+= 1 if  is_testing_ok==True else 0
                asr+= 1 if  is_success==True else 0
                assr+= 1 if  is_success==True and is_testing_ok==False else 0
                avrage_iters+=run_iters
                results.append([row[0],row[1],is_success,score,input,output,is_testing_ok,run_iters,testing_succ_num])
                write_reults(results)
                attack_template2["current_id"]=id 
                metrics["testing_succ_num"]=testing_succ_num/(id+1)
                metrics["avg_iters"]=avrage_iters/(id+1)
                metrics["asr"]=asr/(id+1)
                metrics["assr"]=assr/(id+1)
                current_time = datetime.now()
                metrics["z_avg_runing_time"].append((current_time - start_time).total_seconds()/(id+1))
                metrics["z_asr"].append(metrics["asr"])
                metrics["z_tssr"].append(metrics["testing_succ_num"])
                metrics["z_assr"].append(metrics["assr"])
                metrics["z_anai"].append(metrics["avg_iters"])

                avg_tae=0
                top_tae=0
                with open(adv_library_path,"rb") as f:
                    adv_list=pickle.load(f)
                    if(len(adv_list)!=0):
                        top_tae=len(adv_list[0]["configs"])/(id+1)
                        for i in adv_list:
                            avg_tae+=len(i["configs"])
                        avg_tae/=((id+1)*len(adv_list))
                metrics["avg_tae"]=avg_tae
                metrics["top_tae"]=top_tae
                metrics["z_avg_tae"].append(avg_tae)
                metrics["z_top_tae"].append(top_tae)
                attack_template2["z_metrics"]=metrics
                with open(f"{res_file_preix}/attack_config.yaml", 'w') as file:
                    yaml.dump(attack_template2, file)
        testing_succ_num/=(id+1)
        avrage_iters/=(id+1)
        asr/=(id+1)
        assr/=(id+1)
        f=open(adv_library_path,"rb")
        adv_list=pickle.load(f)
        avg_tae=0
        top_tae=len(adv_list[0]["configs"])/(id+1)
        for i in adv_list:
            avg_tae+=len(i["configs"])
        avg_tae/=((id+1)*len(adv_list))
        results.insert(0,[testing_succ_num,avrage_iters,asr,assr,top_tae,avg_tae])
        write_reults(results)
    except Exception as e:
        print(f"Error occurred: {e}")
        with open(f"{res_file_preix}/error.txt", 'w') as file:
            import traceback
            traceback.print_exc(file=file)
            file.write(str(e))

#gray-box 禁用grad_evolving 禁用crossover
#black-box 禁用score 禁用grad_evolving 禁用crossover
