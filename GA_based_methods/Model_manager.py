# import sys,copy

# import dashscope
# sys.path.append("..")
# from dependency.llm_attacks.minimal_gcg.opt_utils import *
# from dependency.llm_attacks.minimal_gcg.string_utils import load_conversation_template,SuffixManager
# from GA_based_methods.Utils import *
# from langchain_community.llms import Tongyi
# from langchain_core.language_models.base import get_tokenizer
# import os
# from openai import OpenAI
# os.environ["DASHSCOPE_API_KEY"]=os.getenv("DASHSCOPE_API_KEY", "")

# #支持 的模型有：
# #deepseek 系列  deepseek-r1,deepseek-v3,deepseek-r1-distill-qwen-1.5b,deepseek-r1-distill-qwen-7b,deepseek-r1-distill-qwen-14b,deepseek-r1-distill-qwen-32b,deepseek-r1-distill-llama-8b,deepseek-r1-distill-llama-70b
# #通义系列 qwen-plus
# #minmax 使用dashscope abab6.5g-chat abab6.5s-chat 申请中
# #llama 系列 llama3.3-70b-instruct llama3.2-3b-instruct lama3.1-8b-instruct llama3.1-405b-instruct
# #yi yi-large yi-medium yi-large-rag(联网) yi-large-turbo 申请中
# #
# api_version={
#    "deepseek":"openai",
#     "qwen":"openai",
#     "llama":"dashscope", 
#     "abab":"dashscope",
#     "yi":"dashscope"
# }
# support_online_model=["deepseek","qwen","llama","abab","yi"]
# bacth_support=["deepseek-r1","deepseek-v3"]


# class model_manager:
#     def __init__(self,model_path,model_type,device,do_sample) -> None:
#         self.model_name=model_path
#         if("model" in model_path):
#             self.model,self.tokenizer=load_model_and_tokenizer(model_path,low_cpu_mem_usage=True, use_cache=False,device=device)
#             #device已经不再管用了 
#             self.is_online_model=False
#         elif( any(i in model_path for i in support_online_model)):
#             self.is_online_model=True
#             self.tokenizer=get_tokenizer()
#             for i in support_online_model:
#                 if i in model_path:
#                     self.used_api_version=api_version[i]
#                     break
#             if self.used_api_version=="openai":
#                 self.client = OpenAI(
#                     api_key=os.getenv("DASHSCOPE_API_KEY"),  
#                     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
#                 )
#             else:
#                 pass
#         else:
#             raise ValueError("model not support")
#         # if(model_path!="close-source"):
#         #     self.model,self.tokenizer=load_model_and_tokenizer(model_path,low_cpu_mem_usage=True, use_cache=False,device=device)
#         # else:
#         #     self.model=Tongyi(top_p=1)
#         #     self.tokenizer=get_tokenizer()
#         self.model_type=model_type
#         self.model_path=model_path
#         self.device=device
#         self.do_sample=do_sample
#         #清理cache
#         torch.cuda.empty_cache()
#     def text2token(self,text):
#         return self.tokenizer([text], return_tensors="pt").to(self.device)["input_ids"]
#     def token2text(self,token):
#         return self.tokenizer.decode(token)
#     def generate(self, input_ids, assistant_role_slice, gen_config=None,max_length=32):
#         if self.is_online_model:
#             input_ids = input_ids[:assistant_role_slice.stop].to(self.device).unsqueeze(0)
#             text=self.tokenizer.decode(input_ids[0],skip_special_tokens=True)
#             return self.model_query(text,max_tokens=max_length)
#         if gen_config is None:
#             gen_config = self.model.generation_config
#             gen_config.max_new_tokens = max_length
#             gen_config.do_sample=self.do_sample #关闭采样！ 这样保证输出的稳定性
#         #if gen_config.max_new_tokens > 50:
#         #    print('WARNING: max_new_tokens > 32 may cause testing to slow down.')
#         input_ids = input_ids[:assistant_role_slice.stop].to(self.device).unsqueeze(0)
#         attn_masks = torch.ones_like(input_ids).to(self.device)
#         output_ids = self.model.generate(input_ids, 
#                                     attention_mask=attn_masks, 
#                                     generation_config=gen_config,
#                                     pad_token_id=self.tokenizer.pad_token_id)[0]
#         return output_ids[assistant_role_slice.stop:]
#     def get_gradient_loss_logits(self,input_ids,input_slice,target_slice,loss_slice,positive_tokens,negetive_tokens): #对函数进行封装
#         input_ids=input_ids.to(self.device)
#         embed_weights = get_embedding_matrix(self.model)
#         one_hot = torch.zeros(
#             input_ids[input_slice].shape[0],
#             embed_weights.shape[0],
#             device=self.model.device,
#             dtype=embed_weights.dtype
#         )
#         one_hot.scatter_(
#             1, 
#             input_ids[input_slice].unsqueeze(1),
#             torch.ones(one_hot.shape[0], 1, device=self.model.device, dtype=embed_weights.dtype)
#         )
#         one_hot.requires_grad_()
#         input_embeds = (one_hot @ embed_weights).unsqueeze(0)
        
#         # now stitch it together with the rest of the embeddings
#         embeds = get_embeddings(self.model, input_ids.unsqueeze(0)).detach()
#         full_embeds = torch.cat(
#             [
#                 embeds[:,:input_slice.start,:], 
#                 input_embeds, 
#                 embeds[:,input_slice.stop:,:]
#             ], 
#             dim=1)
        
#         logits = self.model(inputs_embeds=full_embeds).logits
#         targets = input_ids[target_slice]
#         #loss = nn.CrossEntropyLoss()(logits[0,loss_slice,:], targets)
#         loss = logits[0,-1,negetive_tokens].sum()-logits[0,-1,positive_tokens].sum()
#         #loss=nn.CrossEntropyLoss()(logits[0,-1,:], torch.tensor(40914,device=model.device)) #19643
#         loss.backward()
#         grad = one_hot.grad.clone()
#         grad = grad / grad.norm(dim=-1, keepdim=True)
#         return grad,loss,logits

#     def get_output_one_by_one(self,inputids,max_length=300):
#         input_ids=copy.deepcopy(inputids)
#         original_len=input_ids.shape[1]
#         for _ in range(max_length):
#             outputs=self.model(input_ids)
#             logits = outputs.logits[:,-1,:]
#             ids=torch.argmax(logits, dim=-1)
#             #print(self.tokenizer.decode(ids,skip_special_tokens=True),outputs.logits.shape)
#             if ids[0]==self.tokenizer.eos_token_id:
#                 break
#             input_ids=torch.cat([input_ids,ids.unsqueeze(-1)],dim=-1)
#         response = self.tokenizer.decode(input_ids[0][original_len:],skip_special_tokens=True)
#         return response
#     def model_query(self,input_text,max_tokens=200):
#         try:
#             print("input_text",input_text,"\n\n")
#             prompt="Insturction: You are a helpful assistant. You will be given an input and you need to provide the output based on the input. Please follow the instruction carefully."
#             text = f"{prompt} Input: {input_text} Output:"
#             text=input_text
#             if self.is_online_model:
#                 if self.used_api_version=="openai":
#                     if "deepseek" in self.model_name:
#                         completion = self.client.chat.completions.create(
#                             model=self.model_name,  # 此处以 deepseek-r1 为例，可按需更换模型名称。
#                             messages=[
#                                 {'role': 'user', 'content': text}
#                             ]
#                         )
#                     elif "qwen" in self.model_name:
#                         completion = self.client.chat.completions.create(
#                             # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
#                             model=self.model_name,  # 此处以 qwen-7b 为例，可按需更换模型名称。
#                             messages=[
#                                 {"role": "system", "content": prompt},
#                                 {"role": "user", "content": input_text},
#                             ],
#                             # Qwen3模型通过enable_thinking参数控制思考过程（开源版默认True，商业版默认False）
#                             # 使用Qwen3开源版模型时，若未启用流式输出，请将下行取消注释，否则会报错
#                             extra_body={"enable_thinking": False},
#                             max_tokens=max_tokens
#                         )
                    
#                     out_text = completion.choices[0].message.content
#                     # if out_text == "":
#                     #     raise ValueError("output is empty")
#                     return out_text
#                 elif self.used_api_version=="dashscope":
#                     messages = [{"role": "system", "content": prompt},
#                                 {"role": "user", "content": input_text}]
#                     response = dashscope.Generation.call(
#                         # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
#                         api_key=os.getenv('DASHSCOPE_API_KEY'),
#                         model=self.model_name,  # 此处以abab6.5s-chat为例，可按需更换模型名称。模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
#                         messages=messages,
#                         max_tokens=max_tokens,
#                     )
#                     if(self.model_name in ["abab6.5g-chat"]):
#                         return response["output"].choices[0].message.content 
#                     return response["output"]["text"]
#             raise ValueError("model not support")
#         except Exception as e:
#             return " " #思考链太长
#             #raise ValueError(f"Error in model_query: {e},error")


import sys, copy

import dashscope
sys.path.append("..")
from dependency.llm_attacks.minimal_gcg.opt_utils import *
from dependency.llm_attacks.minimal_gcg.string_utils import load_conversation_template,SuffixManager
from GA_based_methods.Utils import *
from langchain_community.llms import Tongyi
from langchain_core.language_models.base import get_tokenizer
import os
from openai import OpenAI
# API keys are read from environment variables. Set DASHSCOPE_API_KEY before
# running, e.g. export DASHSCOPE_API_KEY="sk-xxx". Never hardcode secrets.
os.environ.setdefault("DASHSCOPE_API_KEY", os.getenv("DASHSCOPE_API_KEY", ""))

# 支持的模型有：
# deepseek 系列  deepseek-r1,deepseek-v3,deepseek-r1-distill-qwen-1.5b,deepseek-r1-distill-qwen-7b,deepseek-r1-distill-qwen-14b,deepseek-r1-distill-qwen-32b,deepseek-r1-distill-llama-8b,deepseek-r1-distill-llama-70b
# 通义系列 qwen-plus
# minmax 使用dashscope abab6.5g-chat abab6.5s-chat 申请中
# llama 系列 llama3.3-70b-instruct llama3.2-3b-instruct lama3.1-8b-instruct llama3.1-405b-instruct
# yi yi-large yi-medium yi-large-rag(联网) yi-large-turbo 申请中
# OpenAI兼容接口 支持所有OpenAI兼容的模型
#
api_version={
   "deepseek":"openai",
    "qwen":"openai",
    "llama":"dashscope", 
    "abab":"dashscope",
    "yi":"dashscope",
    "openai_compatible": "openai",  # 新增OpenAI兼容接口
    "gpt":"openai",
    "grok":"openai",
    "doubao":"openai",
    "Doubao":"openai",
    "ernie":"openai",
    "kimi":"openai",
    "gemini":"openai"
    
}
qzz_support_models=["gpt","grok","doubao","Doubao","ernie","kimi","gemini"]  # 支持智增增的模型列表
qzz_api_base_urls="https://api.zhizengzeng.com/v1"
# Read from env to avoid hardcoding secrets.
qzz_key=os.getenv("QZZ_API_KEY", "")
support_online_model=["deepseek","qwen","llama","abab","yi", "openai_compatible","gpt","grok","doubao","Doubao","ernie","kimi","gemini"]  # 添加openai_compatible
bacth_support=["deepseek-r1","deepseek-v3"]

from contextlib import contextmanager

@contextmanager
def no_weight_grad(model):
    """上下文管理器：临时禁用模型参数梯度"""
    original_states = {}
    for name, param in model.named_parameters():
        original_states[name] = param.requires_grad
        param.requires_grad_(False)
    
    try:
        yield
    finally:
        for name, param in model.named_parameters():
            param.requires_grad_(original_states[name])
class model_manager:
    def __init__(self, model_path, model_type, device, do_sample, openai_api_key=None, openai_base_url=None) -> None:  # 添加OpenAI参数
        self.model_name = model_path
        if("model" in model_path):
            self.model, self.tokenizer = load_model_and_tokenizer(model_path, low_cpu_mem_usage=True, use_cache=False, device=device)
            # device已经不再管用了 
            self.is_online_model = False
        elif(any(i in model_path for i in support_online_model)):
            self.is_online_model = True
            self.tokenizer = get_tokenizer()
            for i in support_online_model:
                if i in model_path:
                    self.used_api_version = api_version[i]
                    self.provider = i  # 记录提供商类型
                    break
            
            if self.used_api_version == "openai":
                # 如果是OpenAI兼容接口，使用自定义的API密钥和基础URL
                if any([i in model_path for i in qzz_support_models]): 
                    self.client = OpenAI(
                        api_key=qzz_key,
                        base_url=qzz_api_base_urls
                    )
                else:
                    # 原有的Dashscope OpenAI兼容模式
                    self.client = OpenAI(
                        api_key=os.getenv("DASHSCOPE_API_KEY"),  
                        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
                    )
            else:
                pass  # 保持原有的dashscope处理
        else:
            raise ValueError("model not support")
        
        self.model_type = model_type
        self.model_path = model_path
        self.device = device
        self.do_sample = do_sample
        # 清理cache
        torch.cuda.empty_cache()
    
    def text2token(self, text):
        return self.tokenizer([text], return_tensors="pt").to(self.device)["input_ids"]
    
    def token2text(self, token):
        return self.tokenizer.decode(token)
    
    def generate(self, input_ids, assistant_role_slice, gen_config=None, max_length=32):
        if self.is_online_model:
            input_ids = input_ids[:assistant_role_slice.stop].to(self.device).unsqueeze(0)
            text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
            return self.model_query(text, max_tokens=max_length)
        
        if gen_config is None:
            gen_config = self.model.generation_config
            gen_config.max_new_tokens = max_length
            gen_config.do_sample = self.do_sample  # 关闭采样！这样保证输出的稳定性
        
        input_ids = input_ids[:assistant_role_slice.stop].to(self.device).unsqueeze(0)
        attn_masks = torch.ones_like(input_ids).to(self.device)
        output_ids = self.model.generate(input_ids, 
                                    attention_mask=attn_masks, 
                                    generation_config=gen_config,
                                    pad_token_id=self.tokenizer.pad_token_id)[0]
        return output_ids[assistant_role_slice.stop:]
    


    def get_gradient_loss_logits(self, input_ids, input_slice, positive_tokens, negetive_tokens):
        """改造后的函数：只计算输入梯度，不计算模型权重梯度"""
        input_ids = input_ids.to(self.device)
        embed_weights = get_embedding_matrix(self.model)
        
        # 创建one_hot张量
        one_hot = torch.zeros(
            input_ids[input_slice].shape[0],
            embed_weights.shape[0],
            device=self.model.device,
            dtype=embed_weights.dtype
        )
        one_hot.scatter_(
            1, 
            input_ids[input_slice].unsqueeze(1),
            torch.ones(one_hot.shape[0], 1, device=self.model.device, dtype=embed_weights.dtype)
        )
        
        # 使用上下文管理器禁用模型参数梯度
        with no_weight_grad(self.model):
            # 确保只有one_hot需要梯度
            one_hot.requires_grad_(True)
            
            # 计算输入嵌入
            input_embeds = (one_hot @ embed_weights).unsqueeze(0)
            
            # 获取其他部分的嵌入（detach确保不需要梯度）
            embeds = get_embeddings(self.model, input_ids.unsqueeze(0)).detach()
            
            # 拼接完整的嵌入
            full_embeds = torch.cat([
                embeds[:, :input_slice.start, :], 
                input_embeds, 
                embeds[:, input_slice.stop:, :]
            ], dim=1)
            
            # 前向传播
            logits = self.model(inputs_embeds=full_embeds).logits
            
            # 计算损失
            loss = logits[0, -1, negetive_tokens].sum() - logits[0, -1, positive_tokens].sum()
            
            # 反向传播（只计算one_hot的梯度）
            loss.backward()
        
        # 获取并处理梯度
        grad = one_hot.grad.clone()
        grad = grad / grad.norm(dim=-1, keepdim=True)
        
        # 清理梯度引用
        one_hot.grad = None
        
        return grad, loss, logits

    def get_output_one_by_one(self, inputids, max_length=300, temperature=0.7):
        input_ids = copy.deepcopy(inputids)
        original_len = input_ids.shape[1]
        
        for _ in range(max_length):
            outputs = self.model(input_ids)
            logits = outputs.logits[:, -1, :]
            
            # 使用多项式采样而不是贪婪解码
            # 应用温度缩放
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            
            # 从多项式分布中采样
            ids = torch.multinomial(probs, num_samples=1)
            
            if ids[0] == self.tokenizer.eos_token_id:
                break
                
            input_ids = torch.cat([input_ids, ids], dim=-1)
        
        response = self.tokenizer.decode(input_ids[0][original_len:], skip_special_tokens=True)
        return response
    
    def model_query(self, input_text, max_tokens=200):
        try:
            prompt = "Insturction: You are a helpful assistant. You will be given an input and you need to provide the output based on the input. Please follow the instruction carefully."
            text = f"{prompt} Input: {input_text} Output:"
            text = input_text
            
            if self.is_online_model:
                if self.used_api_version == "openai":
                    # OpenAI兼容接口处理
                    if any([i in self.model_path for i in qzz_support_models]):
                        completion = self.client.chat.completions.create(
                            model=self.model_name,
                            messages=[
                                {"role": "system", "content": prompt},
                                {"role": "user", "content": input_text},
                            ],
                            max_tokens=max_tokens,
                            
                        )
                    elif "deepseek" in self.model_name:
                        completion = self.client.chat.completions.create(
                            model=self.model_name,  # 此处以 deepseek-r1 为例，可按需更换模型名称。
                            messages=[
                                {'role': 'user', 'content': text},
                            ],
                            extra_body={"enable_thinking": False},
                        )
                    elif "qwen" in self.model_name:
                        completion = self.client.chat.completions.create(
                            # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
                            model=self.model_name,  # 此处以 qwen-7b 为例，可按需更换模型名称。
                            messages=[
                                {"role": "system", "content": prompt},
                                {"role": "user", "content": input_text},
                            ],
                            # Qwen3模型通过enable_thinking参数控制思考过程（开源版默认True，商业版默认False）
                            # 使用Qwen3开源版模型时，若未启用流式输出，请将下行取消注释，否则会报错
                            extra_body={"enable_thinking": False},
                            max_tokens=max_tokens
                        )
                    try:
                        out_text = completion.choices[0].message.content
                    except:
                        out_text = " "
                    # if out_text == "":
                    #     raise ValueError("output is empty")
                    return out_text
                elif self.used_api_version == "dashscope":
                    messages = [{"role": "system", "content": prompt},
                                {"role": "user", "content": input_text}]
                    response = dashscope.Generation.call(
                        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
                        api_key=os.getenv('DASHSCOPE_API_KEY'),
                        model=self.model_name,  # 此处以abab6.5s-chat为例，可按需更换模型名称。模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
                        messages=messages,
                        max_tokens=max_tokens,
                    )
                    if(self.model_name in ["abab6.5g-chat"]):
                        return response["output"].choices[0].message.content 
                    return response["output"]["text"]
            raise ValueError("model not support")
        except Exception as e:
            return " "  # 思考链太长
            # raise ValueError(f"Error in model_query: {e},error")