import json

import torch
from tqdm import trange
import gc

from llm.openai_api import openai_call


system_promt = (
    "你是一个女装直播字幕分段助手。请根据当前句子和上下文，判断当前句子是否为「产品介绍段落的分界点」。规则如下：\n"
    "1. 分界点通常是主播结束当前产品、切换新产品的过渡句（如“过掉了”“再上一个”）。\n"
    "2. 分界句后的内容会开始描述新产品的特征（如颜色、材质、价格）。\n"
    "3. 注意排除重复口语（如“这个过掉了，过掉了哈”可能只有第二句是分界点）。\n"
    "输入格式：\n"
    "- 当前句子：[句子序号] [句子内容]\n"
    "- 上下文（最近几句）：[句子内容...]\n"
    "请用 JSON 格式回答：\n"
    "{\n"
    "  \"is_boundary\": true/false,\n"
    "  \"reason\": \"分界理由或排除理由\"\n"
    "}\n"
    "示例输入1：\n"
    "当前句子：过掉了哈"
    "上下文：够这个不多了，\n这个过掉了，\n过掉了哈，\n然后再给大家上一个刚出完货的，"
    "示例输出1："
    "{\n"
    "  \"is_boundary\": true,\n"
    "  \"reason\": \"主播明确用‘过掉了哈’结束当前产品，且后文开始新产品的介绍（如‘再给大家上一个’）\"\n"
    "}\n"
    "示例输入2：\n"
    "当前句子：只有 s 码了啊，"
    "上下文：嗯，啊，只有 s 码了啊，少女的妈妈也可以，这这裙子真的很优雅，就你感觉你要给自己一个机会去感受一下。"
    "示例输出2："
    "{\n"
    "  \"is_boundary\": false,\n"
    "  \"reason\": \"这只是推销的话术\"\n"
    "}\n"
)


class _Caller:
    
    def __call__(self, system_prompt, user_prompt):
        raise NotImplementedError
    
    def __exit__(self):
        pass
    
    def __enter__(self):
        return self


class APICaller(_Caller):

    def __init__(self, apikey, model):
        self.apikey = apikey
        self.model = model

    def __call__(self, system_prompt, user_prompt):
        llm_response = openai_call(
            apikey=self.apikey,
            model=self.model,
            user_content=user_prompt,
            system_content=system_prompt,
            is_json=True
        )
        return llm_response


class LocalCaller:
    
    def __init__(self, model="Qwen/Qwen2.5-7B-Instruct", device='cuda:0', torch_dtype=torch.bfloat16):
        self.model = model
        self.device = device
        self.torch_dtype = torch_dtype
        self.pipe = None

    def __enter__(self):
        from transformers import pipeline
        self.pipe = pipeline("text-generation", model=self.model, device=self.device, torch_dtype=self.torch_dtype)
        return self

    def __call__(self, system_prompt, user_prompt):
        if system_prompt is not None and len(system_prompt.strip()):
            messages = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
        ]
        else:
            messages = [
                {'role': 'user', 'content': user_prompt}
        ]
        llm_response = self.pipe(messages, max_new_tokens=500)[0]['generated_text'][-1]['content']
        return llm_response
    
    def __exit__(self):
        del self.pipe
        gc.collect()
        torch.cuda.empty_cache()


def local_model_pipeline(model="Qwen/Qwen2.5-7B-Instruct", device='cuda:0', torch_dtype=torch.bfloat16):
    from transformers import pipeline
    pipe = pipeline("text-generation", model=model, device=device, torch_dtype=torch_dtype)

    def caller(system_prompt, user_prompt):
        if system_prompt is not None and len(system_prompt.strip()):
            messages = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
        ]
        else:
            messages = [
                {'role': 'user', 'content': user_prompt}
        ]
        llm_response = pipe(messages, max_new_tokens=500)[0]['generated_text'][-1]['content']
        return llm_response
    
    return caller


def sw_split(
    text_clips, slide_window_size=9,
    caller=None
):
    bounds = []
    half_window_size = slide_window_size // 2
    skip_until = 0
    for i in trange(len(text_clips)):
        if i < half_window_size:
            continue
        if i >= len(text_clips) - half_window_size:
            break
        if i < skip_until:
            continue
        user_prompt = (
            f"当前句子：{text_clips[i][0]}\n"
            f"上下文：{''.join([ele[0] for ele in text_clips[i-slide_window_size:i+half_window_size+1]])}\n"
        )
        output = caller(system_promt, user_prompt)
        is_boundary = json.loads(output)['is_boundary']
        if is_boundary:
            bounds.append(i)
            skip_until = i + half_window_size
    return bounds