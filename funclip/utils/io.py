from os.path import join
from typing import List, Tuple
import copy
import base64
from io import BytesIO
from PIL import Image

import numpy as np
from moviepy import VideoFileClip, VideoClip, AudioClip
from moviepy import concatenate_videoclips
from moviepy.video.fx import AccelDecel

from asr import SentenceSRT


def convert_pcm_to_float(data):
    if data.dtype == np.float64:
        return data
    elif data.dtype == np.float32:
        return data.astype(np.float64)
    elif data.dtype == np.int16:
        bit_depth = 16
    elif data.dtype == np.int32:
        bit_depth = 32
    elif data.dtype == np.int8:
        bit_depth = 8
    else:
        raise ValueError("Unsupported audio data type")
    
    # Now handle the integer types
    max_int_value = float(2 ** (bit_depth - 1))
    if bit_depth == 8:
        data = data - 128
    return (data.astype(np.float64) / max_int_value)
    
def get_video(vidoe_path) -> VideoFileClip:
    return VideoFileClip(vidoe_path)

def get_audio_binary(clip: VideoClip, sr=16_000) -> np.ndarray:
    # audio: AudioClip = clip.audio
    # return audio.to_soundarray(fps=16000)
    import tempfile, librosa
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = join(temp_dir, "audio.wav")
        audio:AudioClip = clip.audio
        audio.write_audiofile(file_path)
        wav, _ = librosa.load(file_path, sr=sr)
    return wav

def split_by_segments(video:VideoClip, text_clips:List[SentenceSRT], preds:List[int]) -> List[Tuple[VideoClip, List[SentenceSRT]]]:

    before = 0
    preds.append(len(text_clips)-1)

    clips = []
    for pred in preds:
        if pred <= before:
            continue
        start = text_clips[before].start
        start_time = text_clips[before]._duration.start
        end = text_clips[pred].end
        subvideo = video.subclipped(start, end)
        new_srts = []
        for srt in text_clips[before:pred+1]:
            new_srt = copy.deepcopy(srt)
            new_srt.shift(start_time)
            new_srts.append(new_srt)
        clips.append((copy.copy(subvideo), new_srts))
        before = pred + 1

    return clips

def pick_clips(video:VideoClip, text_clips:List[SentenceSRT], preds:List[int]) -> List[Tuple[VideoClip, SentenceSRT]]:

    clips = []
    for pred in preds:
        start = text_clips[pred].start
        end = text_clips[pred].end

        subvideo = video.subclipped(start, end)
        srt = copy.deepcopy(text_clips[pred])
        srt.shift(text_clips[pred]._duration.start)
        clips.append((copy.copy(subvideo), srt))

    return clips

def merge_clips(clips:List[Tuple[VideoClip, SentenceSRT]]) -> Tuple[VideoClip, List[SentenceSRT]]:
    if len(clips) == 0:
        raise ValueError("No clips to merge")
    if len(clips) == 1:
        return clips[0][0], [clips[0][1]]
    videos = list(map(lambda x: x[0], clips))
    prev_end = None
    srts = list(map(lambda x: x[1], clips))
    for srt in srts:
        if prev_end:
            srt.shift(prev_end, True)
        prev_end = copy.deepcopy(srt._duration.end)
    return concatenate_videoclips(videos), srts

def speed_up_video(video:VideoClip, srt:SentenceSRT, max_seconds_per_token=.178) -> Tuple[VideoClip, SentenceSRT]:
    num_of_tokens = len(srt.text)
    duration_in_secs = srt.duration.to_milliseconds() / 1000
    max_duration = max_seconds_per_token * num_of_tokens

    if duration_in_secs <= max_duration:
        return video, srt
    else:
        acc = AccelDecel(max_duration)
        return acc.apply(video), srt

def image_np_to_base64(image:np.ndarray) -> str:
    img = Image.fromarray(image)
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str