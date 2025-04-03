import argparse
import os
from typing import List
import tempfile

import gradio as gr

from agents.asr_agent import ASRAgent
from states.video_state import VideoState
from utils.time_utils import SentenceSRT
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

theme = gr.Theme.load("funclip/utils/theme.json")
with gr.Blocks(theme=theme) as funclip_service:
    gr.Markdown("# Deep Clip")
    video_state = gr.State()
    with gr.Row():
        with gr.Column():
            video_input = gr.Text(label="⏪ 视频输入 | Video Input")
            audio_btn = gr.Button(" 加载视频⏪音频👂 | ASR", variant="primary")
            output_dir = gr.Text(label="输出目录 | Output Directory")
            
            gr.Markdown("### 分割视频 | Split Video")
            split_bounds = gr.TextArea()
            split_btn = gr.Button("✂️ 裁剪 | Clip")
            

        # Right Column
        with gr.Column():
            blocks_output = gr.Dataframe(
                headers=["Index", "Start", "End", "Text"],
                interactive=False,
                visible=True
            )
            outputs_dirs_frame = gr.Dataframe(
                headers=["Outputs"],
                datatype=["str"],
                interactive=False,
                visible=False
            )

    @audio_btn.click(inputs=video_input, outputs=[video_state, output_dir, blocks_output])
    def transcribe(video_path):
        # get directiory 
        dir_path = os.path.dirname(video_path)

        video_state = VideoState(video_path)
        audio_state = video_state.get_audio_binary()

        with ASRAgent() as agent:
            srts = agent(audio_state)
            video_state.set_srts(srts)
        return video_state, gr.Text(dir_path), gr.Dataframe(
            value=[[i, srt.start, srt.end, srt.text] for i,srt in enumerate(srts)],
            interactive=False,
        )
        
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
        
    @split_btn.click(inputs=[video_state, split_bounds, output_dir], outputs=[video_state, output_dir, outputs_dirs_frame])
    def split_into_blocks(video_state:VideoState, split_bounds:str, output_dir:str):

        new_bounds = [int(x) for x in split_bounds.split(",")]
        if len(new_bounds) == 0:
            # add warning
            logger.warning("No segments selected for splitting.")
            return video_state, output_dir, gr.DataFrame([], visible=False)
        new_clips = video_state.split_by_segments(new_bounds)
        file_name = os.path.basename(video_state.clip.filename)
        file_base_names = os.path.splitext(file_name)
        new_paths = []
        for i,new_clip in enumerate(new_clips):
            file_path = os.path.join(output_dir, f"{file_base_names[0]}_{i}{file_base_names[1]}")
            new_paths.append(file_path)
            new_clip.clip.write_videofile(file_path, codec="libx264", audio_codec="aac")
        for new_clip in new_clips:
            new_clip.clip.close()
        video_state.clip.close()
        return None, None, gr.Dataframe(
            value=new_paths,
            visible=True,
        )

if __name__ == "__main__":
    funclip_service.launch()
