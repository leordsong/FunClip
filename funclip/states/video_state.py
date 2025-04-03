from os.path import join
from typing import List, Tuple, Optional
import copy

from moviepy import VideoFileClip, AudioClip, VideoClip
import numpy as np
import tempfile, librosa

from logger import logger
from utils.time_utils import SentenceSRT


class VideoState:

    def __init__(self, video_path: Optional[str], video_clip: Optional[VideoClip] = None, srts:Optional[List[SentenceSRT]]=None) -> None:
        assert video_path or video_clip, "Either video_path or video_clip must be provided"
        self.clip = VideoFileClip(video_path) if video_path else video_clip
        self.current_frame = 0
        self.srts = srts

    def get_frame(self, frame_number: int) -> np.ndarray:
        return self.clip.get_frame(frame_number)

    def get_audio_binary(self, sr=16000) -> np.ndarray:
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = join(temp_dir, "audio.wav")
            audio:AudioClip = self.clip.audio
            audio.write_audiofile(file_path)
            wav, _ = librosa.load(file_path, sr=sr)
            logger.info(f"Audio file loaded into memory")
        return wav
    
    def set_srts(self, srts: List[SentenceSRT]) -> None:
        self.srts = srts

    def get_srts(self) -> List[SentenceSRT]:
        return self.srts
    
    def split_by_segments(self, preds:List[int]) -> List['VideoState']:

        before = 0
        preds.append(len(self.srts)-1)

        clips = []
        for pred in preds:
            if pred <= before:
                continue
            start = self.srts[before].start
            start_time = self.srts[before]._duration.start
            end = self.srts[pred].end
            subvideo:VideoClip = self.clip.subclipped(start, end)
            new_srts = []
            for srt in self.srts[before:pred+1]:
                new_srt = copy.deepcopy(srt)
                new_srt.move_forward(start_time)
                new_srts.append(new_srt)
            clips.append(VideoState(None, subvideo.copy(), new_srts))
            before = pred + 1

        return clips