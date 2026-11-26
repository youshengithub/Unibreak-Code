import math
from .Fitness_calcution import attack_coder,fitness_calcution
from .Encoder import attack_coder

import torch,random,copy
from tqdm import tqdm
import numpy as np
import os
class GA: #evloutionary algorithm
    def __init__(self,per_length=10,max_v=10,non_as=None,population_size=10,crossover_num=4,mutaion_tario=0.05,use_gradient=False,attack_type="suffix",tensorboadr_log=False,adv_lib=None,enable_adv_sample=True,enable_mutation_sensitivity=False,enable_crossover_sensitivity=False,enable_calculate_fitness=True,device='cpu') -> None:
        self.population_size=population_size
        self.mutation_ratio=mutaion_tario
        self.per_length=per_length
        self.non_as=non_as
        self.max_v=max_v
        self.crossover_num=crossover_num
        self.device=device
        self.use_gradient=use_gradient 
        self.attack_type=attack_type 
        self.tensorboadr_log=tensorboadr_log
        self.adv_lib=adv_lib
        self.enable_adv_sample=enable_adv_sample
        self.enable_mutation_sensitivity=enable_mutation_sensitivity
        self.enable_crossover_sensitivity=enable_crossover_sensitivity
        self.enable_calculate_fitness=enable_calculate_fitness
        if(self.enable_calculate_fitness==False):
            assert self.enable_crossover_sensitivity==False and self.use_gradient==False
        if(self.enable_mutation_sensitivity==True):
            attack_coder.token_sensitivity=self.adv_lib.token_statistics()
        
    def initial(self): #从库里面加载
        self.select_stastic=[]
        tokens=self.adv_lib.get_topk(self.population_size)
        self.pers=[ attack_coder(self.per_length,self.max_v,self.non_as,self.device) for i in range(self.population_size)]
        for id,token in enumerate(tokens):
            self.pers[id].per=token
    def solve(self,iter_num,finess_class: fitness_calcution):
        self.best_per=None
        self.select_stastic=[]
        self.initial()
        data_bar=tqdm(range(iter_num))   # 将data设置为进度条
        for  i in data_bar:
            #计算适应度
            grads=finess_class.score(self.pers,self.enable_calculate_fitness,self.attack_type)
            self.pers=sorted(self.pers,key=lambda x:x.score)
            if(i==iter_num-1):
                break
            ###############log code
            gradient_generates=[]
            for per in self.pers:
                if(per.is_success==True):
                    self.best_per=per
                    return self.best_per,i
            data_bar.set_description(f"Score:{self.pers[0].score}") 
            if(self.best_per==None or self.best_per.score>self.pers[0].score):
                self.best_per=copy.deepcopy(self.pers[0])
            ################
            if(self.use_gradient==True):
                gradient_generates=self.gradient_evoluation(grads)
                # continue
            crossover_generates=self.crossover(grads)
            selected=self.selection(gradient_generates,crossover_generates)
            del self.pers
            self.pers=selected
            self.mutation()
        if(self.best_per==None or self.best_per.score>self.pers[0].score):
            return self.pers[0],i
        print(self.select_stastic)
        return self.best_per,i
    def gradient_evoluation(self,grads): #utilize the grads to evolve
        evoluated_pers=[]
        for id,grad in enumerate(grads): #这里也使用sample
            if self.non_as is not None:
                grad[:, self.non_as] = np.infty
            if(self.enable_adv_sample==True):
                value,index=torch.topk(-grad,k=256)
                probs = torch.nn.functional.softmax(value, dim=1)
                sampled_indices = torch.multinomial(probs, num_samples=1)
                selected_index = torch.gather(index, 1, sampled_indices)
                selected_index = selected_index.squeeze(1)
                index=selected_index
            else:
                value,index=torch.topk(-grad,k=1) 
                
            copyed=copy.deepcopy(self.pers[id])
            if(self.attack_type=="suffix"):
                copyed.per=index[-self.per_length:].squeeze()
            elif(self.attack_type=="prefix"):
                copyed.per=index[:self.per_length].squeeze()
            elif(self.attack_type=="suffix&prefix"):
                copyed.per=torch.cat([index[:self.per_length//2],index[-self.per_length//2:]])
            else:
                raise Exception("attack_type must be suffix or prefix")
            evoluated_pers.append(copyed)
            #self.pers.append(copyed)
        return evoluated_pers
    def crossover(self,grads): #不要random_cross
        new_pers=[]
        if(self.enable_crossover_sensitivity==False):
            choose_times=math.ceil((1+math.pow(1+8*self.population_size,0.5))/2)
        else:
            choose_times=self.population_size
        for _ in range(choose_times):
            id1,id2=random.randint(0,len(self.pers)-1),random.randint(0,len(self.pers)-1)
            per1=copy.deepcopy(self.pers[id1])
            per2=copy.deepcopy(self.pers[id2])
            if(self.enable_crossover_sensitivity==False):
                for i in range(self.crossover_num): 
                    cross_point=random.randint(0,per1.length-1)
                    new_per=attack_coder(per1.length,self.max_v,self.non_as,self.device)
                    new_per.per=torch.cat([per1.per[:cross_point],per2.per[cross_point:]])
                    new_pers.append(new_per)
            else:
                score=0
                weight1=torch.gather(grads[id1], 1, per1.per.long().unsqueeze(1)).squeeze(1)
                weight2=torch.gather(grads[id2], 1, per2.per.long().unsqueeze(1)).squeeze(1)
                new_per=attack_coder(per1.length,self.max_v,self.non_as,self.device)
                for i in range(per1.length):
                    if(weight1[i]<weight2[i]): #越小越好！
                        score+=weight1[i]
                        new_per.per[i]=per1.per[i]
                    else:
                        score+=weight2[i]
                        new_per.per[i]=per2.per[i]
                new_per.score=score
                new_pers.append(new_per)
        new_pers.sort(key=lambda x:x.score)
        return new_pers #new_pers
    def selection(self,gradient_generate,crossover_genrate):#
        #gradient_genrate(依概率) 和crossover_genrate(依概率随机)中选择
        assert(len(gradient_generate)+len(crossover_genrate)>=self.population_size)
        all_new_pers=[]
        if(self.use_gradient==False):
            gradient_generate=crossover_genrate[:len(crossover_genrate)//2]
            crossover_genrate=crossover_genrate[len(crossover_genrate)//2:]
    
        for alternative_set in [gradient_generate,crossover_genrate]: 
            new_pers=[]
            
            need_chose=self.population_size//2
            delta=1/len(alternative_set)
            selected_ids=[]
            for i in range(len(alternative_set)):
                calc_x=i*delta
                if(torch.rand(1).item()<((-(calc_x*2-1)*(2*calc_x**0.75-1)*(calc_x*2-1)+1)/2)): #运用的三次方函数  可以有选择的加入一些点
                    new_pers.append(alternative_set[i])
                    selected_ids.append(i)
                    if(len(new_pers)==need_chose): #够了就不要选了
                        break
            #不够的话就随机选取
            while(len(new_pers)<need_chose):
                id=random.randint(0,len(alternative_set)-1)
                if(id not in selected_ids):
                    selected_ids.append(id)
                    new_pers.append(alternative_set[id])
            all_new_pers+=new_pers
        selected_ids.sort()
        self.select_stastic.append(selected_ids)
        return all_new_pers        

    def mutation(self): #搞定!!
        for id,per in enumerate(self.pers):
            for i in range(per.length):
                if(torch.rand(1).item()<self.mutation_ratio):
                    per.per[i]=per.sample()
