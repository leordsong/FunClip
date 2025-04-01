import gradio as gr
from moviepy import VideoFileClip
import librosa
from funasr import AutoModel
import json
from os.path import join, exists
from os import remove
import numpy as np
from utils.subtitle_utils import time_convert

# 初始化模型（放在全局避免重复加载）
funasr_model = AutoModel(
    model="iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
    vad_model="damo/speech_fsmn_vad_zh-cn-16k-common-pytorch",
    punc_model="damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
    device='cuda:0'
)

def process_video(video_path, progress=gr.Progress()):
    """处理视频并返回字幕数据"""
    progress(0.2, "提取音频中...")
    clip = VideoFileClip(video_path)
    
    # 提取音频
    audio_path = "temp_audio.wav"
    try:
        clip.audio.write_audiofile(audio_path, logger=None)
        wav, _ = librosa.load(audio_path, sr=16000)
    finally:
        if exists(audio_path):
            remove(audio_path)
    
    # 语音识别
    progress(0.5, "语音识别中...")
    rec_result = funasr_model.generate(
        wav, 
        return_spk_res=False, 
        sentence_timestamp=True,
        output_dir="outputs"
    )
    
    # 结构化数据
    sentences = [
        {
            "start": s["start"],
            "end": s["end"],
            "text": s["text"],
            "is_boundary": False,
            "is_distorted": False
        } for s in rec_result[0]["sentence_info"]
    ]
    
    return {
        "video_path": video_path,
        "sentences": sentences,
        "current_idx": 0
    }, gr.update(visible=True)  # 显示标注界面

def update_subtitles(video_state, current_time):
    """根据当前时间更新字幕显示"""
    if not video_state:
        return "", []
    
    current_time = current_time * 1000  # 转换到毫秒

    # update timestamp_box
    global timestamp
    timestamp = time_convert(current_time) if current_time else "00:00"

    active_sentences = []
    
    # 查找当前有效句子
    for idx, sent in enumerate(video_state["sentences"]):
        if sent["start"] <= current_time <= sent["end"]:
            active_sentences.append(f"{sent['text']} ({sent['start']//1000}-{sent['end']//1000}s)")
            video_state["current_idx"] = idx
    
    return "\n".join(active_sentences), video_state, timestamp

def save_annotations(video_state):
    """保存标注结果"""
    if not video_state:
        return
    
    output_path = "annotations.json"
    with open(output_path, "w") as f:
        json.dump(video_state["sentences"], f, ensure_ascii=False, indent=2)
    
    return output_path

with gr.Blocks(title="视频标注工具") as demo:
    video_state = gr.State()
    
    with gr.Row():
        with gr.Column(scale=2):
            video_input = gr.Video(label="上传视频", sources=["upload"])
            progress_bar = gr.Progress()
            process_btn = gr.Button("开始处理", progress_bar)
        
        with gr.Column(scale=3, visible=False) as annotate_col:
            video_output = gr.Video(label="播放视频", interactive=False)
            time_slider = gr.Slider(0, 100, label="时间进度", interactive=True)
            # with gr.Row():
                # time_slider = gr.Slider(0, 100, label="时间进度", interactive=True)
            timestamp_display = gr.Textbox(
                label="当前时间",
                value="00:00",
                interactive=False,
                elem_classes=["timestamp"]
            )

            subtitles_box = gr.Textbox(label="当前字幕", interactive=False)
            
            with gr.Row():
                boundary_check = gr.Checkbox(label="是分界点")
                distortion_check = gr.Checkbox(label="画面扭曲")
                save_btn = gr.Button("保存标注")
            
            download = gr.File(label="下载标注")

    # 处理视频
    process_btn.click(
        process_video,
        inputs=[video_input],
        outputs=[video_state, annotate_col]
    ).then(
        lambda x: (x["video_path"], x["video_path"]),
        inputs=[video_state],
        outputs=[video_output, time_slider]
    )

    # 时间轴变化时更新
    time_slider.change(
        update_subtitles,
        inputs=[video_state, time_slider],
        outputs=[subtitles_box, video_state, timestamp_display]
    )

    # 标注交互
    boundary_check.change(
        lambda x, vs: vs["sentences"][vs["current_idx"]].update(is_boundary=x),
        inputs=[boundary_check, video_state],
        outputs=[]
    )
    distortion_check.change(
        lambda x, vs: vs["sentences"][vs["current_idx"]].update(is_distorted=x),
        inputs=[distortion_check, video_state],
        outputs=[]
    )

    # 保存结果
    save_btn.click(
        save_annotations,
        inputs=[video_state],
        outputs=[download]
    )

if __name__ == "__main__":
    demo.launch()