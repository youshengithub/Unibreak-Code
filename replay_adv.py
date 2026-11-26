import warnings,torch
import numpy as np
np.random.seed(20) #fix the random seed, so that the results are reproducible
torch.manual_seed(20)
torch.cuda.manual_seed_all(20)

warnings.filterwarnings("ignore", category=UserWarning)
from GA_based_methods.Attack_manager import attacker_manager

if __name__ == '__main__':
    ak=attacker_manager()
    find_adv=""
    find_adv=""
    output=ak.replay_attack("configs/attack_conf_debug.yaml",None,find_adv,max_lenght=300)
    print(f"output:\n{output}")