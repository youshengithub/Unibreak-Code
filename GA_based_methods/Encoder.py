import torch

class attack_coder: # A encoded  perturbation
    token_sensitivity=None
    def __init__(self,length,max_v,non_ascii=None,device='cpu'):
        self.non_ascii=non_ascii
        self.device=device
        self.per=torch.randint_like(torch.zeros(length),max_v).to(device)
        self.length=length
        self.max_v=max_v
        self.score=0
        self.fix_per()
    def sample(self):
        find=(self.non_ascii==None)
        while find==False:
            if(self.token_sensitivity==None):
                value = torch.randint(self.max_v,(1,)).to(self.device)
            else:
                value = torch.multinomial(self.token_sensitivity, num_samples=1, replacement=True).to(self.device)
            if(value not in self.non_ascii):
                break
        assert(value<self.max_v)
        return value
    def fix_per(self):
        if(self.non_ascii!=None):
            for i in range(self.per.shape[0]):
                if(self.per[i] in self.non_ascii):
                    self.per[i]=self.sample()