import os,pickle
import torch

def is_config_equal(config1,config2): #判断两个config是否相等
    if config1["model_path"]!=config2["model_path"]: return False
    if config1["input"]!=config2["input"]: return False
    if config1["attack_type"]!=config2["attack_type"]: return False
    if config1["per_length"]!=config2["per_length"]: return False
    return True

class adv_library:
    def __init__(self,path,vocab_size,tokenizer,enable_library,device) -> None:
        self.path=path
        self.adv_list=[]
        self.device=device
        self.vocab_size=vocab_size
        self.tokenizer=tokenizer
        self.enable_library=enable_library
        self.load()
    def load(self):
        if(os.path.exists(self.path)):
            with open(self.path,"rb") as f:
                self.adv_list=pickle.load(f)
        else:
            self.adv_list=[]
        for i in self.adv_list: 
            i["adv_token"]=torch.tensor(i["adv_token"]).to(self.device).int()
    def save(self):
        save_ans=[]
        for i in self.adv_list: 
            save_ans.append({"adv_token":i["adv_token"].cpu().numpy().tolist(),"configs":i["configs"],"adv_str":i["adv_str"],"adv_token":i["adv_token"]})
        with open(self.path,"wb") as f:
            pickle.dump(save_ans,f)
        self.export_csv()
    def add(self,adv,config):
        if(self.enable_library==False):
            return 
        for i in self.adv_list:
            if( all(i["adv_token"]==adv)):
                for t_config in i["configs"]: #去重
                    if( is_config_equal(t_config,config) ):
                        return
                i["configs"].append(config)
                return
        self.adv_list.append({"adv_token":adv.int(),"adv_str":self.tokenizer.decode(adv.int()),"configs":[config]})
    def get_topk(self,k): #返回依据len(input)排序的前k个 
        if self.enable_library==False: return []
        self.adv_list=sorted(self.adv_list,key=lambda x:len(x["configs"]),reverse=True)
        return [ i["adv_token"] for i in self.adv_list[:k]]
    def token_statistics(self):#统计adv里面token出现的次数,返回 token和token的次数
        tokens=torch.ones(self.vocab_size).to(self.device)#这里可能会出错！
        if self.enable_library==False: return tokens
        for adv in self.adv_list:
            for token in adv["adv_token"]:
                    tokens[token]+=len(adv["configs"])
        return tokens
    def export_csv(self):
        import pandas as pd
        data=[]
        for adv in self.adv_list:
            dicta=dict()
            dicta["adv_token"]=adv["adv_token"].cpu().numpy().tolist()
            dicta["adv_str"]=adv["adv_str"]
            for id,i in enumerate(adv["configs"]):
                dicta[f"configs:{id}"]=i
            data.append(dicta)
        df=pd.DataFrame(data)
        df.to_csv(self.path+".csv",index=False)
    def remove_allconfigs(self):    
        for i in self.adv_list:
            i["configs"]=[]
        self.save()
    #去重
    def remove_duplicate(self):
        seen = set()  # 用于跟踪已遇到的 adv_str
        result = []   # 存储结果
        for item in self.adv_list:
            adv_str = item.get("adv_str")
            if adv_str is not None and adv_str not in seen:
                seen.add(adv_str)
                result.append(item)
        self.adv_list=result
        self.save()
    def find_config(self,config):
        for record in self.adv_list:
            if(config in record["configs"]): return record
        return None