from typing import List, Tuple
import os

from utils.io import get_audio_binary, get_video, split_by_segments, pick_clips, merge_clips, speed_up_video
from asr import asr, SentenceSRT
from lang_clip.segmentation.full_split import text_segment_openai_call
from lang_clip.refine.selection import selection_openai_call


if __name__ == "__main__":

    video_path = "C:\\videos\\扬哥回放\\2025-03-01 23-31-43.mp4"
    output_dir = "./outputs/2025-03-01"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    sampling_rate = 16000
    api_key = os.getenv("DS_API_KEY")
    assert api_key is not None, "Please set DS_API_KEY in your environment variables"

    with get_video(video_path) as clip:
        wav = get_audio_binary(clip, sr=sampling_rate)
        sentences:List[SentenceSRT] = asr(wav)

        boundaries = text_segment_openai_call(
            apikey=api_key,
            字幕列表=[ele.text for ele in sentences],
        )
        segments = split_by_segments(clip, sentences, boundaries)

        for i, (subclip, srts) in enumerate(segments):
            selections = selection_openai_call(
                apikey=api_key,
                字幕列表=[ele.text for ele in srts]
            )
            subclips = pick_clips(subclip, srts, selections)
            # subclips = list(map(lambda x: speed_up_video(*x), subclips))
            merged_clip, merged_srts = merge_clips(subclips)
            merged_clip.write_videofile(os.path.join(output_dir, f"subclip_{i}.mp4"))