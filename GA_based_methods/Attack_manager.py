from .Model_manager import model_manager
from .Utils import get_attack_configs
from .GA import GA
from .Fitness_calcution import fitness_calcution
from dependency import get_nonascii_toks
from .Adv_libraray import adv_library,is_config_equal
import gc,torch

class attacker_manager:
    def __init__(self) -> None:
        self.model_list={}
        self.attacked_list=[]
        pass
    def replay_attack(self,config_path,configs,adv,model=None,max_lenght=300): 
        if(config_path!=None):
            config=get_attack_configs(config_path)
        else:  
            config=configs
        model_path=config["model_path"]
        model_type=config["model_type"]
        device=config["device"]
        prompt=config["prompt"]
        input=config["input"]
        except_output=config["except_output"]   
        do_sample=config["do_sample"]
        if(model==None):
            model = model_manager(model_path,model_type,device,do_sample)
        if(not isinstance(adv,str)):
            adv_ids=adv.to(model.device).int()
        else:
            adv_ids=torch.tensor(model.tokenizer(adv,add_special_tokens=False)["input_ids"],device=model.device)
        if model.is_online_model:
            return model.model_query(input+" "+model.tokenizer.decode(adv_ids),max_tokens=max_lenght)
        # Build the prompt exactly as the search does: goal tokens followed by the
        # perturbation tokens.  (The bundled SuffixManager relies on older fastchat
        # conversation templates, so it is deliberately not used here.)
        goal_ids=model.text2token(input)[0]
        input_ids=torch.cat([goal_ids,adv_ids],dim=-1)
        control_slice=slice(len(goal_ids),len(input_ids))
        output_logits=model.generate(input_ids,control_slice,max_length=max_lenght)
        if(isinstance(output_logits,str)):
            return output_logits
        return model.tokenizer.decode(output_logits)
        
    def apply_to_configs(self,config):
        #获取model  
        # model_path=config["model_path"]
        # if model_path not in self.model_list:
        #     model = model_manager(model_path,model_type,device,do_sample)
        #     self.model_list[model_path]=model
        # else:
        #     model=self.model_list[model_path]
        # #获取libraray
        # adv_lib=adv_library(config["adv_library"],None,None,config["enable_library"],config["device"])
        # for adv in adv_lib.adv_list:
        #     is_in=False
        #     for cf in adv["configs"]:
        #         if(is_config_equal(config,cf)):
        #             is_in=True
        #             break
        #     if(is_in==False):
        #         text=self.replay_attack(None,config,record["adv_token"],model,150)
        #         if(fitness.check_for_attack_success_for_text(text)):
        #             print(text,end="\n\n\n")
        #             adv_lib.add(record["adv_token"],config)
        #             adv_lib.save()
        pass
        
    def attack(self,config_path=None,configs=None):
        if(config_path==None and configs==None):
            raise ValueError("config_path and configs cannot be both None")
        if(configs==None):
            config=get_attack_configs(config_path)
        else:
            config=configs
        model_path=config["model_path"]
        model_type=config["model_type"]
        device=config["device"]
        prompt=config["prompt"]
        input=config["input"]
        except_output=config["except_output"]
        iters=config["iters"]
        per_length=config["per_length"]
        population_size=config["population_size"]
        crossover_num=config["crossover_num"]
        mutation_rate=config["mutation_rate"]
        no_ascii=config["no_ascii"]
        use_gradient=config["use_gradient"]
        tensorboadr_log=config["tensorboard_log"]   
        do_sample=config["do_sample"]
        adv_library_=config["adv_library"]
        enable_library=config["enable_library"]
        enable_adv_sample=config["enable_adv_sample"]
        enable_mutation_sensitivity=config["enable_mutation_sensitivity"]
        enable_crossover_sensitivity=config["enable_crossover_sensitivity"]
        enable_calculate_fitness=config["enable_calculate_fitness"]
        attack_type=config["attack_type"]
        #get model
        if model_path not in self.model_list:
            model = model_manager(model_path,model_type,device,do_sample)
            self.model_list[model_path]=model
        else:
            model=self.model_list[model_path]
        #get adv_library
        adv_lib=adv_library(adv_library_,len(model.tokenizer),model.tokenizer,enable_library,device)
        # adv_lib.remove_allconfigs()
        # exit(0)
        #get fitness
        fitness=fitness_calcution(model,prompt,input,except_output)
        
        #直接去尝试
        # find_flag=adv_lib.find_config(config)
        # if(find_flag!=None):
        #     text=self.replay_attack(None,config,find_flag["adv_token"],model,80)
        #     return True,-1,find_flag["adv_str"],text,text
        is_testing_ok=False
        for record in adv_lib.adv_list:
            if(config in record["configs"]): continue
            text=self.replay_attack(None,config,record["adv_token"],model,500)
            if(fitness.check_for_attack_success_for_text(text)):
                print(text,end="\n\n\n")
                is_testing_ok=True
                adv_lib.add(record["adv_token"],config)
                adv_lib.save()
                break

        if(no_ascii==True):
            non_as=get_nonascii_toks(model.tokenizer,device)
        else:
            non_as=None
        ga=GA(per_length,len(model.tokenizer),non_as,population_size,crossover_num,mutation_rate,use_gradient,attack_type,tensorboadr_log,adv_lib,enable_adv_sample,enable_mutation_sensitivity,enable_crossover_sensitivity,enable_calculate_fitness,device) #还需要指定编码的方案
        #solve
        best_per,run_iters=ga.solve(iters,fitness)
        #送去检查
        #fitness.score([best_per],attack_type,decode_best=True)
        #print(f"score:{best_per.score}\ninput:\n{best_per.input}\noutput:\n{best_per.output}")
        return_flag=False
        if(best_per.is_success==True):
            for record in adv_lib.adv_list:
                if( all(record["adv_token"]==best_per.per)):#有重复了就不再添加了
                    return_flag=True   
            if(return_flag==False): 
                for record in self.attacked_list:
                    if(config["model_path"]==record["model_path"]):
                        #大概率会实现 不用更换模型
                        text=self.replay_attack(None,record,best_per.per,model)
                    else:
                        text=self.replay_attack(None,record,best_per.per)
                    if(fitness.check_for_attack_success_for_text(text)):
                        adv_lib.add(best_per.per,record)
                adv_lib.add(best_per.per,config)
                adv_lib.save()
        self.attacked_list.append(config) 
        adv_lib.save()
        return best_per.is_success,best_per.score,best_per.input,best_per.output,is_testing_ok,run_iters
        
        