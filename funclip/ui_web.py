import argparse
import os
from typing import List
import json

import gradio as gr

from agents.asr_agent import ASRAgent
from states.video_state import VideoState
from utils.time_utils import SentenceSRT
from llm.openai_api import openai_call_reasoning, deepseek_call_balance
from utils.io import merge_clips, pick_clips

from logger import logger


parser = argparse.ArgumentParser(description='argparse testing')
parser.add_argument('--share', '-s', action='store_true', help="if to establish gradio share link")
parser.add_argument('--port', '-p', type=int, default=7860, help='port number')
args = parser.parse_args()


# def create_video_player(video_state, start_time, end_time):
#     with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmpfile:
#         clip = video_state.subclip(start_time, end_time)
#         clip.write_videofile(tmpfile.name)
#         return tmpfile.name

text_list = []

theme = gr.Theme.load("funclip/utils/theme.json")
with gr.Blocks(theme=theme) as funclip_service:
    gr.Markdown("# 深度剪辑")

    with gr.Tabs():
        # with gr.TabItem("混剪"):

        with gr.TabItem("混剪"):
            video_state = gr.State()
            with gr.Row():
                with gr.Column():
                    video_path_input = gr.Text(label="⏪ 视频文件地址", interactive=True)
                    audio_btn = gr.Button("加载视频⏪音频👂", variant="primary")
                    output_dir = gr.Text(label="输出目录", interactive=True)
                    
                    gr.Markdown("### 分割视频")
                    # split_bounds = gr.TextArea()
                    number_of_segments = gr.Number(label="分割数量", value=0, interactive=True)
                    calibrate_btn = gr.Checkbox(label="校准分割点", value=False, interactive=True)
                    split_btn = gr.Button("✂️ 开始分割")
                    

                # Right Column
                with gr.Column():
                    
                    keywords = gr.Text(label="关键词", interactive=True)
                    with gr.Row():
                        start_slider = gr.Slider(
                                minimum=0, maximum=100, value=1, step=1, label="开始", scale=3
                        )
                        end_slider = gr.Slider(
                                minimum=0, maximum=120.0, value=0, step=0.1, label="Audio Split Point(s)", scale=3
                        )
                    # search_btn = gr.Button("搜索", visible=True, scale=1)

                    # with gr.Row():
                    #     index_slider = gr.Slider(
                    #             minimum=0, maximum=100, value=1, step=1, label="Index", scale=3
                    #     )
                    #     splitpoint_slider = gr.Slider(
                    #             minimum=0, maximum=120.0, value=0, step=0.1, label="Audio Split Point(s)", scale=3
                    #     )
                    #     btn_audio_split = gr.Button("Split Audio", scale=1)
                    #     btn_save_json = gr.Button("Save File", visible=True, scale=1)
                    #     btn_invert_selection = gr.Button("Invert Selection", scale=1)

                    # with gr.Row():
                               
                    @gr.render(inputs=[video_state, keywords])
                    def show_split(video_state:VideoState, key):
                        if video_state is None or len(video_state.get_srts()) == 0:
                            gr.Markdown("## No Input Provided")
                        else:
                            srts = video_state.get_srts()
                            if key:
                                rows = [x for x in srts if key in x.text]
                            for audio_state in rows:
                                with gr.Row():
                                    gr.Textbox(audio_state.start, show_label=False)
                                    gr.Textbox(audio_state.end, show_label=False)
                                    gr.Textbox(audio_state.text, show_label=False)
                                    gr.Checkbox(
                                        label="分界点",
                                        value=audio_state.is_selected,
                                    )
                    # blocks_output = gr.Dataframe(
                    #     headers=["Index", "Start", "End", "Text"],
                    #     interactive=False,
                    #     visible=True
                    # )

        with gr.TabItem("精剪"):
            video_state2 = gr.State()
            audios_state2 = gr.State([])
            with gr.Row():
                with gr.Column():
                    video_input = gr.Video(label="视频输入")
                with gr.Column():
                    with gr.Row():
                        audio_btn2 = gr.Button("生成字幕⏪", variant="primary")
                        update_btn = gr.Button("更新字幕", variant="primary")
                    output_dir2 = gr.Text(label="输出目录", interactive=True)
                    
                    gr.Markdown("### 字幕列表")
                    @gr.render(inputs=[video_state2])
                    def show_split2(video_state:VideoState):
                        if video_state is None:
                            return
                        text_list.clear()
                        if video_state is not None and len(video_state.get_srts()) > 0:
                            for i,audio_state in enumerate(video_state.get_srts()):
                                with gr.Row():
                                    gr.Textbox(str(i+1), show_label=False, scale=1)
                                    gr.Textbox(audio_state.start[3:-1], show_label=False, scale=2)
                                    gr.Textbox(audio_state.end[3:-1], show_label=False, scale=2)
                                    tb = gr.Textbox(audio_state.text, show_label=False, scale=4, interactive=True)
                                    text_list.append(tb)

                # Right Column
                with gr.Column():
                    split_btn2 = gr.Button("✂️ 开始分割")
                    video_output = gr.Video(label="视频输出", visible=False)
        
        with gr.TabItem("参数"):
            # split_bounds = gr.TextArea()
                    system_prompt_input = gr.TextArea(
                        label="Prompt System (按需更改，最好不要变动主体和要求)",
                        value="""你是女装直播剪辑专家。请分析以下按时间顺序排列的字幕句子列表，提取出介绍女装**精彩**的字幕。

# 输出要求
1. **开头**剪辑一些**吸引人**的片段：比如“卖点”，“名牌”，“好看”等（可以将后面的片段提前）
2. 再放一些**衣服细节**：尺码、颜色、材质、款式等
3. （可选）再放一些**价格信息**：价格、折扣、优惠等
4. **剔除语气词**：忽略口语填充词（如"嗯""啊"）
5. **剔除无关内容**：聊天内容，无关女装的内容，问候语，上链接等销售术语
6. 每一段保持**精简**，几段话就可以
7. 直接返回可供json.load的字符串：{"clips": [分界句序号列表], "title": ""}。省略Markdown的```json```的格式。

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

输出：{"clips": [72,73,59,60,61,62,63,64,65,66,67], "title": "宝妈精选"} 
解释：72,73 是卖点，53-58是聊天内容，不是介绍女装的内容；59-68是介绍女装的内容。69-76是推销内容。""",
                        interactive=True,
                    )
                    user_prompt_input = gr.Textbox(
                        label="Prompt User（不需要修改，会自动拼接左下角的srt字幕）",
                        value=("请分析以下直播字幕的女装介绍内容\n输入："),
                        interactive=True,
                    )
                    llm_model = gr.Dropdown(
                        choices=[
                            "deepseek-reasoner",
                            "deepseek-chat",
                        ],
                        value="deepseek-reasoner",
                        label="大模型算法",
                        allow_custom_value=True,
                        interactive=True,
                    )
                    apikey_input = gr.Textbox(label="APIKEY", interactive=True, value="sk-e03ec4e356c04d548cd1895cc3452ef7")
                    temperature_input = gr.Slider(
                        minimum=0, maximum=2, value=0.5, step=0.1, label="温度", scale=3,
                        interactive=True
                    )
                    with gr.Row():
                        balance_btn = gr.Button("余额查询", variant="primary")
                        balance_area = gr.Textbox(label="余额", interactive=False)

    @balance_btn.click(
        inputs=[apikey_input],
        outputs=[balance_area],
    )
    def get_balance(apikey:str):
        balance = deepseek_call_balance(apikey)
        return balance

    @audio_btn.click(inputs=video_path_input, outputs=[video_state, output_dir])
    @audio_btn2.click(inputs=video_input, outputs=[video_state2, output_dir2])
    def transcribe(video_path):
        # get directiory 
        dir_path = os.path.dirname(video_path)

        video_state = VideoState(video_path)
        audio_state = video_state.get_audio_binary()

        with ASRAgent() as agent:
            srts = agent(audio_state)
            video_state.set_srts(srts)
        return video_state, gr.Text(dir_path)
    
    @split_btn2.click(
        inputs=[video_state2, apikey_input, llm_model, system_prompt_input, user_prompt_input, output_dir2, temperature_input],
        outputs=video_output,
    )
    def selection_openai_call(
        video_state:VideoState,
        apikey:str, 
        model,
        system_content,
        user_content,
        output_dir:str,
        temperature:float
    ):
        subtitle_list = "\n".join([f"{i}. {text}" for i, text in enumerate(video_state.get_srts())])

        user_content = user_content + subtitle_list + "\n输出："
        logger.info("Calling openai model.")
        output, r_output = openai_call_reasoning(apikey, model, user_content, system_content, temperature=temperature)
        logger.info("Openai model inference done.")
        logger.info("Output: ", output)
        logger.debug("Reasoning output: ", r_output)
        json_output = json.loads(output)
        selections = json_output['clips']
        
        subclips = pick_clips(video_state.clip, video_state.get_srts(), selections)
        # subclips = list(map(lambda x: speed_up_video(*x), subclips))
        merged_clip, _ = merge_clips(subclips)
        output_path = os.path.join(output_dir, "output.mp4")
        merged_clip.write_videofile(output_path)

        merged_clip.close()

        return gr.Video(value=output_path, visible=True)

    @update_btn.click(
        inputs=[video_state2],
        outputs=[video_state2, *text_list],
    )
    def update_subtitle(video_state:VideoState):
        logger.info(text_list)
        for i, tb in enumerate(text_list):
            srt = video_state.get_srts()[i]
            if tb.value != srt.text:
                logger.info(f"Updating subtitle {i}: {srt.text} -> {tb.value}")
                srt.text = tb.value
        return video_state, text_list
        
    # @blocks_output.change(
    #     inputs=[blocks_output, bounds_state],
    #     outputs=[blocks_output, bounds_state]
    # )
    # def update_table(blocks, bs):
    #     new_values = []
    #     for i,value in enumerate(blocks.values):
    #         a, b, c, split = value
    #         if bs[i] != split:
    #             bs[i] = split
    #         new_values.append([a, b, c, split])
    #     # logger.info(sum(bs))
    #     return gr.DataFrame(new_values), bs
        
    # @split_btn.click(inputs=[video_state, split_bounds, output_dir], outputs=[video_state, output_dir, outputs_dirs_frame])
    # def split_into_blocks(video_state:VideoState, split_bounds:str, output_dir:str):

    #     new_bounds = [int(x) for x in split_bounds.split(",")]
    #     if len(new_bounds) == 0:
    #         # add warning
    #         logger.warning("No segments selected for splitting.")
    #         return video_state, output_dir, gr.DataFrame([], visible=False)
    #     new_clips = video_state.split_by_segments(new_bounds)
    #     file_name = os.path.basename(video_state.clip.filename)
    #     file_base_names = os.path.splitext(file_name)
    #     new_paths = []
    #     for i,new_clip in enumerate(new_clips):
    #         file_path = os.path.join(output_dir, f"{file_base_names[0]}_{i}{file_base_names[1]}")
    #         new_paths.append(file_path)
    #         new_clip.clip.write_videofile(file_path, codec="libx264", audio_codec="aac")
    #     for new_clip in new_clips:
    #         new_clip.clip.close()
    #     video_state.clip.close()
    #     return None, None, gr.Dataframe(
    #         value=new_paths,
    #         visible=True,
    #     )

if __name__ == "__main__":
    funclip_service.launch()
