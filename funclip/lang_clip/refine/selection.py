from typing import List, Optional
import logging
import json

from openai import OpenAI
from llm.openai_api import openai_call


text_filter_system_prompt = """你是女装直播剪辑专家。请分析以下按时间顺序排列的字幕句子列表，提取出介绍女装的字幕。

# 介绍女装重点特征
0. **吸引人的话术**：比如“这个很好看”、“这个很适合你”，“穿起来很有气质”等
1. **衣服细节**：尺码、颜色、材质、款式等
2. **价格信息**：价格、折扣、优惠等
2. **剔除语气词**：忽略重复性口语填充词（如"嗯""啊"）
3. **剔除无关内容**：聊天内容，无关女装的内容，问候语等

# 输出要求
- 直接返回可供json.load的字符串：{"clips": [分界句序号列表]}。省略Markdown的```json```的格式。

# 示例分析
输入：
53. 男方也骂我们
54. 我是不是那样的人给你看一下
55. 我现在不敢骂人了
56. 我去年骂人被封号了
57. 这条裙子就是四月份的
58. 穿在身上很有质感的那种
59. 四月份的一个款式哦
60. 四月不同班 janda 家的叉 s 码码长度八十七
61. 腰围六十六到十十 SM 码
62. 长度十二
63. 腰围六十八到七十六 s 码
64. 长度九十五
65. 腰围七十到七十八叉 s 穿到一百一 s 码
66. 穿到一百一十五 m 码
67. 穿到一百二十斤
68. 二码四十九上车
69. 你这帅出来这条裙子
70. 我觉得很适合直播间
71. 很多当个妈妈的姐妹
72. 如果你们是一个比较精致的宝妈
73. 我推荐你们去试穿这种裙子
74. 这种裙子就是其他同学的妈妈
75. 她看到会觉得哇某某同学你的妈妈嗯好有品味
76. 穿的好精致的那种裙子

输出：{"clips": [59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76]} 
解释：53-58是聊天内容，不是介绍女装的内容；59-68是介绍女装的内容。69-76是推销内容。"""

text_filter_user_prompt_template = """请分析以下直播字幕的女装介绍内容
输入：
{}

输出（按JSON格式返回结果）：
"""


def selection_openai_call(
    apikey:str, 
    字幕列表:List[str], 
    model="deepseek-reasoner",
    system_content=text_filter_system_prompt,
    user_content:Optional[str]=None,
):
    subtitle_list = "\n".join([f"{i}. {text}" for i, text in enumerate(字幕列表)])

    user_content = user_content if user_content else text_filter_user_prompt_template.format(subtitle_list)
    logging.info("Openai model inference done.")
    output = openai_call(apikey, model, user_content, system_content, is_json=True)
    logging.info(output)
    json_output = json.loads(output)
    return json_output['clips']

__all__ = ["selection_openai_call"] 