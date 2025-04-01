import os
import logging
import json

from llm.openai_api import openai_call

text_seg_system_prompt = """你是女装直播字幕分段专家。请分析以下按时间顺序排列的字幕句子列表，精确识别所有产品介绍段落的分界点。

# 分界点特征
1. **过渡标志**：使用明确结束当前产品的词汇（如"过掉了""换款""下一个"）
2. **新品引入**：分界后的句子会描述新品属性（颜色/尺码/材质/价格）;还有一些特殊的引入词，比如“准备一下”，“给大家上一个”“再给你们上”表示介绍新品
4. **排除干扰**：忽略重复性口语填充词（如"嗯""啊"）和未切换产品的数量说明（如"只剩3件"）

# 输出要求
- array of json objects: ```json [{"index": 10, topic: "短裙"}, ...,{"index": n-1, topic: "连衣裙"}]```
- 序号从0开始，仅包含明确符合上述条件的句子编号。n-1是永远会出现的最后一个分界点。

# 方法介绍
1. 先找出所有过渡标志，然后再找出新品引入的句子
2. 通过上下文分析是否为分界点
3. 再通过每一段的主题来区分是否还可以再分割

# 示例分析
输入：
19. 够这个裙子不多了，
20. 这个过掉了，
21. 过掉了哈，
22. 然后再给大家上一个刚出完货的

输出：[{"index": 21, topic: "裙子"}, ...] 
解释：句子20是口语过渡，不是分界点；句子21是明确的切换过渡，后续开始介绍新品。"""


text_seg_user_prompt_template = """请分析以下直播字幕的分界点
输入：
{}

输出：
"""


def text_segment_openai_call(
    apikey, 
    字幕列表, 
    model="deepseek-reasoner",
    system_prompt=text_seg_system_prompt,
    user_content=None,
):
    if not user_content:
        subtitle_list = "\n".join([f"{i}. {text}" for i, text in enumerate(字幕列表)])
        user_content = text_seg_user_prompt_template.format(subtitle_list)
    logging.info("Openai model inference done.")
    output = openai_call(apikey, model, user_content, system_prompt, is_json=True)
    logging.info(output)
    json_output = json.loads(output)
    return json_output