import os
os.environ.setdefault("DASHSCOPE_API_KEY", os.getenv("DASHSCOPE_API_KEY", ""))

import warnings,torch
import numpy as np
np.random.seed(20) #fix the random seed, so that the results are reproducible
torch.manual_seed(20)
torch.cuda.manual_seed_all(20)

warnings.filterwarnings("ignore", category=UserWarning)
from GA_based_methods.Attack_manager import attacker_manager

if __name__ == '__main__':
    ak=attacker_manager()
    is_success,score,input,output,is_testing_ok,run_iters=ak.attack("configs/attack_conf_debug.yaml")
    print(f"is_success:{is_success} score:{score}\ninput:\n{input}\noutput:\n{output}\nis_testing_ok:\n{is_testing_ok}\nrun_iters:\n{run_iters}\n")