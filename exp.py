from random import randint
from time import sleep
import warnings,torch,csv,copy
import numpy as np
np.random.seed(20) #fix the random seed, so that the results are reproducible
torch.manual_seed(20)
torch.cuda.manual_seed_all(20)
warnings.filterwarnings("ignore", category=UserWarning)
from GA_based_methods.Utils import get_attack_configs
from GA_based_methods.Attack_manager import attacker_manager
from datetime import datetime
import argparse
import subprocess
import multiprocessing,concurrent
parser = argparse.ArgumentParser()
parser.add_argument('--exp_type', type=str, default="enable_lib")
parser.add_argument('--model_path', type=str, default="models/qwen1.5b")
args = parser.parse_args()
#harmful_behavior_path=f"dependency/data/advbench/harmful_{args.target}s.csv"
attack_template_path="configs/attack_conf.yaml"
result_file_path=f'results/Exp-{args.exp_type}-{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.csv'
attack_template=get_attack_configs(attack_template_path)
attack_template["tensorboard_log"]=False
def run_test_script(arg,id_queue):
    id = id_queue.get(timeout=1)
    if "population_size" not in arg:
        arg["population_size"] = 10
    if "crossover_num" not in arg:
        arg["crossover_num"] = 5
    if "mutation_rate" not in arg:
        arg["mutation_rate"] = 0.2
    if "target" not in arg:
        arg["target"] = "behaviors"
    if "attack_type" not in arg:
        arg["attack_type"] = "suffix"
    command_line=['python', 'run_harmful_behaviors.py', '--version', str(arg["version"]), '--iteration', str(arg["iteration"]), '--power', str(arg["power"]), '--device', str(id),"--model_path",str(arg["model_path"]),"--population_size",str(arg["population_size"]),"--crossover_num",str(arg["crossover_num"]),"--mutation_rate",str(arg["mutation_rate"]),"--exp_type",str(arg["exp_type"]),"--target",str(arg["target"]),"--attack_type",str(arg["attack_type"])]
    print(command_line)
    subprocess.run(command_line)
    id_queue.put(id)
def write_reults(results):
    with open(result_file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(results)
def ablation_exp():
    task=[
        {"model_path":"models/qwen1.5b","version":"full","iteration":50,"power":10,"population_size":30,"exp_type":"ab"},
        # {"model_path":"models/qwen1.5b","version":"no_initial","iteration":50,"power":10,"population_size":30,"exp_type":"ab"},
        # {"model_path":"models/qwen1.5b","version":"no_gradient_evolve","iteration":50,"power":10,"population_size":30,"exp_type":"ab"},
        # {"model_path":"models/qwen1.5b","version":"no_crossover","iteration":50,"power":10,"population_size":30,"exp_type":"ab"},
        # {"model_path":"models/qwen1.5b","version":"no_mutation","iteration":50,"power":10,"population_size":30,"exp_type":"ab"},
        # {"model_path":"models/qwen3b","version":"full","iteration":50,"power":10,"population_size":30,"exp_type":"ab"},
        # {"model_path":"models/qwen3b","version":"no_initial","iteration":50,"power":10,"population_size":30,"exp_type":"ab"},
        # {"model_path":"models/qwen3b","version":"no_gradient_evolve","iteration":50,"power":10,"population_size":30,"exp_type":"ab"},
        # {"model_path":"models/qwen3b","version":"no_crossover","iteration":50,"power":10,"population_size":30,"exp_type":"ab"},
        # {"model_path":"models/qwen3b","version":"no_mutation","iteration":50,"power":10,"population_size":30,"exp_type":"ab"},
        
        #{"model_path":"models/qwen1.5b","version":"full","iteration":50,"power":4},
        #{"model_path":"models/qwen1.5b","version":"full","iteration":50,"power":16},
        # {"model_path":"models/qwen1.5b","version":"full","iteration":50,"power":10,},
        

        # {"model_path":"models/openelm1b","version":"full","iteration":50,"power":4,"population_size":30,"exp_type":"openelm1b"},
        # {"model_path":"models/openelm1b","version":"full","iteration":50,"power":10,"population_size":30,"exp_type":"openelm1b"},
        # {"model_path":"models/openelm1b","version":"full","iteration":50,"power":16,"population_size":30,"exp_type":"openelm1b"},
        
        # {"model_path":"models/llama1B","version":"full","iteration":50,"power":4},
        # {"model_path":"models/llama1B","version":"full","iteration":50,"power":10},
        # {"model_path":"models/llama1B","version":"full","iteration":50,"power":16},
        
        # {"model_path":"models/qwen1.5b","version":"gray-box","iteration":50,"power":11,"population_size":30,"exp_type":"gray-box"},
        # {"model_path":"models/qwen1.5b","version":"black-box","iteration":50,"power":11,"population_size":30,"exp_type":"black-box"},
        # {"model_path":"models/llama","version":"gray-box","iteration":50,"power":11,"population_size":30,"exp_type":"gray-box"},
        # {"model_path":"models/llama","version":"black-box","iteration":50,"power":11,"population_size":30,"exp_type":"gray-box"},
        # {"model_path":"models/openelm1b2","version":"gray-box","iteration":50,"power":11,"population_size":30,"exp_type":"gray-box"},
        # {"model_path":"models/openelm1b2","version":"black-box","iteration":50,"power":11,"population_size":30,"exp_type":"gray-box"},
        
        
        #hyper实验,设置超参数 种群大小，交叉数，变异率 分别变化5次 还有迭代次数
        # {"model_path":"models/qwen1.5b","version":"full","iteration":50,"power":10,"population_size":5,"crossover_num":2,"mutation_rate":0.1,"exp_type":"hyper"},
        # {"model_path":"models/qwen1.5b","version":"full","iteration":50,"power":10,"population_size":10,"crossover_num":2,"mutation_rate":0.1,"exp_type":"hyper"},
        # {"model_path":"models/qwen1.5b","version":"full","iteration":50,"power":10,"population_size":15,"crossover_num":2,"mutation_rate":0.1,"exp_type":"hyper"},
        # {"model_path":"models/qwen1.5b","version":"full","iteration":50,"power":10,"population_size":20,"crossover_num":2,"mutation_rate":0.1,"exp_type":"hyper"},
        # {"model_path":"models/qwen1.5b","version":"full","iteration":50,"power":10,"population_size":45,"crossover_num":2,"mutation_rate":0.1,"exp_type":"hyper"},
        
        # {"model_path":"models/qwen1.5b","version":"full","iteration":50,"power":10,"population_size":10,"crossover_num":3,"mutation_rate":0.1,"exp_type":"hyper"},
        # {"model_path":"models/qwen1.5b","version":"full","iteration":50,"power":10,"population_size":10,"crossover_num":4,"mutation_rate":0.1,"exp_type":"hyper"},
        # {"model_path":"models/qwen1.5b","version":"full","iteration":50,"power":10,"population_size":10,"crossover_num":5,"mutation_rate":0.1,"exp_type":"hyper"},
        # {"model_path":"models/qwen1.5b","version":"full","iteration":50,"power":10,"population_size":10,"crossover_num":6,"mutation_rate":0.1,"exp_type":"hyper"},
        # {"model_path":"models/qwen1.5b","version":"full","iteration":50,"power":10,"population_size":10,"crossover_num":7,"mutation_rate":0.1,"exp_type":"hyper"},

        # {"model_path":"models/qwen1.5b","version":"full","iteration":50,"power":10,"population_size":10,"crossover_num":2,"mutation_rate":0.05,"exp_type":"hyper"},
        # {"model_path":"models/qwen1.5b","version":"full","iteration":50,"power":10,"population_size":10,"crossover_num":2,"mutation_rate":0.1,"exp_type":"hyper"},
        # {"model_path":"models/qwen1.5b","version":"full","iteration":50,"power":10,"population_size":10,"crossover_num":2,"mutation_rate":0.15,"exp_type":"hyper"},
        # {"model_path":"models/qwen1.5b","version":"full","iteration":50,"power":10,"population_size":10,"crossover_num":2,"mutation_rate":0.2,"exp_type":"hyper"},
        # {"model_path":"models/qwen1.5b","version":"full","iteration":50,"power":10,"population_size":10,"crossover_num":2,"mutation_rate":0.25,"exp_type":"hyper"},
        # # #迭代次数 40-80
        # {"model_path":"models/qwen1.5b","version":"full","iteration":40,"power":10,"population_size":10,"crossover_num":2,"mutation_rate ":0.1,"exp_type":"hyper"},
        # {"model_path":"models/qwen1.5b","version":"full","iteration":50,"power":10,"population_size":10,"crossover_num":2,"mutation_rate ":0.1,"exp_type":"hyper"},
        # {"model_path":"models/qwen1.5b","version":"full","iteration":60,"power":10,"population_size":10,"crossover_num":2,"mutation_rate ":0.1,"exp_type":"hyper"},
        # {"model_path":"models/qwen1.5b","version":"full","iteration":70,"power":10,"population_size":10,"crossover_num":2,"mutation_rate ":0.1,"exp_type":"hyper"},
        # {"model_path":"models/qwen1.5b","version":"full","iteration":80,"power":10,"population_size":10,"crossover_num":2,"mutation_rate ":0.1,"exp_type":"hyper"},
        

        # {"model_path":"models/qwen7b","version":"full","iteration":50,"power":4,"exp_type":"para_size"},
        # # {"model_path":"models/qwen7b","version":"full","iteration":50,"power":10,"exp_type":"para_size"},
        # {"model_path":"models/qwen7b","version":"full","iteration":50,"power":16,"exp_type":"para_size"},
        # # {"model_path":"models/vicuna7b","version":"full","iteration":50,"power":4,"exp_type":"para_size"},
        # # {"model_path":"models/vicuna7b","version":"full","iteration":50,"power":10,"exp_type":"para_size"},
        # # {"model_path":"models/vicuna7b","version":"full","iteration":50,"power":16,"exp_type":"para_size",
        # # {"model_path":"models/llama7b","version":"full","iteration":50,"power":4,"exp_type":"para_size"},
        # # {"model_path":"models/llama7b","version":"full","iteration":50,"power":4,"exp_type":"para_size"},
        # # {"model_path":"models/llama7b","version":"full","iteration":50,"power":16,"exp_type":"para_size"},



        # {"model_path":"models/qwen3b","version":"full","iteration":50,"power":4,"exp_type":"para_size"},
        # {"model_path":"models/qwen7b","version":"black-box","iteration":50,"power":10,"population_size":30,"exp_type":"para_size"},
        # {"model_path":"models/qwen1.5b","version":"full","iteration":50,"power":10,"exp_type":"para_size"},
        # # {"model_path":"models/openelm3b2","version":"full","iteration":50,"power":4,"exp_type":"para_size"},
        # #{"model_path":"models/openelm3b2","version":"full","iteration":50,"power":10,"exp_type":"para_size"},
        # # {"model_path":"models/openelm3b2","version":"full","iteration":50,"power":16,"exp_type":"para_size",
        # # {"model_path":"models/llama3b","version":"full","iteration":50,"power":4,"exp_type":"para_size"},
        # #{"model_path":"models/llama3b","version":"full","iteration":50,"power":10,"exp_type":"para_size"},
        # # {"model_path":"models/llama3b","version":"full","iteration":50,"power":16,"exp_type":"para_size"},
        # # {"model_path":"models/llama3b","version":"full","iteration":50,"power":4,"exp_type":"para_size"},
        ##{"model_path":"deepseek-v3","version":"black-box","iteration":50,"power":10,"exp_type":"commercial_model"}  
        #{"model_path":"qwen-turbo-2025-07-15","version":"black-box","iteration":50,"power":10,"exp_type":"commercial_model"}
        #{"model_path":"qwen-turbo-2025-02-11","version":"black-box","iteration":50,"power":10,"exp_type":"commercial_model"}, #Qwen3
        #{"model_path":"qwen-turbo-0919","version":"black-box","iteration":50,"power":10,"exp_type":"commercial_model"} #qwen2
        #{"model_path":"Doubao-lite-4k","version":"black-box","iteration":50,"power":10,"exp_type":"commercial_model"}
        #{"model_path":"gpt-3.5-turbo","version":"black-box","iteration":50,"power":10,"exp_type":"commercial_model"},
        #{"model_path":"grok-3-mini","version":"black-box","iteration":50,"power":10,"exp_type":"commercial_model"},
        #{"model_path":"gpt-5-nano","version":"black-box","iteration":50,"power":10,"exp_type":"commercial_model"},
        # {"model_path":"ernie-4.0-8k","version":"black-box","iteration":50,"power":10,"exp_type":"commercial_model"},
        # {"model_path":"kimi-k2-turbo-preview","version":"black-box","iteration":50,"power":10,"exp_type":"commercial_model"}
        
        # {"model_path":"models/qwen3b","version":"full","iteration":50,"power":4,"population_size":30,"exp_type":"language","target":"behaviors_chinese"},
        # {"model_path":"models/qwen3b","version":"full","iteration":50,"power":4,"population_size":30,"exp_type":"language","target":"behaviors_franch"},
        # {"model_path":"models/qwen3b","version":"full","iteration":50,"power":4,"population_size":30,"exp_type":"language", "target":"behaviors_japanese"},
        # {"model_path":"models/qwen3b","version":"full","iteration":50,"power":4,"population_size":30,"exp_type":"attack_type","attack_type":"insert"},
        # {"model_path":"models/qwen3b","version":"full","iteration":50,"power":10,"population_size":30,"exp_type":"attack_type","attack_type":"insert"},
        # {"model_path":"models/qwen3b","version":"full","iteration":50,"power":16,"population_size":30,"exp_type":"attack_type", "attack_type":"insert"},
        # {"model_path":"../DELMAN/models/qwen7b","version":"full","iteration":50,"power":10,"population_size":10,"exp_type":"export","target":"converted"},
    ]
    id_queue = multiprocessing.Manager().Queue()
    #获取当前可用的GPUid
    gpus = [i for i in range(torch.cuda.device_count())]
    gpus=[1] #每个cpu开启三个进程
    for i in gpus: id_queue.put(i)
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=len(gpus)) as executor:
        futures = {executor.submit(run_test_script, arg_value, id_queue): arg_value for arg_value in task}
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                print(f'Task generated an exception: {exc}')
def comparasion_exp(): #其他的实验数据
    pass
if __name__ == '__main__':
    if args.exp_type=="enable_lib":
        ablation_exp()
    else:
        comparasion_exp()