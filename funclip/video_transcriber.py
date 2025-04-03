import os
import sys
import subprocess
import argparse
from dataclasses import dataclass
import re
from typing import List

from funasr import AutoModel
from funasr.register import tables


@dataclass
class Timestamp:
    hour: int
    minute: int
    second: int
    millisecond: int

    def to_milliseconds(self) -> int:
        return (self.hour * 3600 + self.minute * 60 + self.second) * 1000 + self.millisecond
    
    def to_str(self) -> str:
        return f"{self.hour:02}:{self.minute:02}:{self.second:02}.{self.millisecond:03}"
    
    def __sub__(self, other:'Timestamp'):
        end = self.to_milliseconds()
        start = other.to_milliseconds()
        assert end >= start
        return Timestamp.from_milliseconds(end - start)
    
    def __add__(self, other:'Timestamp'):
        return Timestamp.from_milliseconds(self.to_milliseconds() + other.to_milliseconds())

    @staticmethod
    def from_milliseconds(ms: int) -> 'Timestamp':
        assert ms >= 0
        s, ms = divmod(ms, 1000)
        m, s = divmod(s, 60)
        h, m = divmod(m, 60)
        return Timestamp(h, m, s, ms)
    
    @staticmethod
    def from_str(s: str) -> 'Timestamp':
        assert re.match(r'\d{2}:\d{2}:\d{2}\.\d{1,3}', s)
        h, m, s = map(int, s.split(':'))
        s, ms = s.split('.')
        return Timestamp(h, m, s, ms)
    
    def __str__(self):
        return self.to_str()
    

@dataclass
class Duration:
    start: Timestamp
    end: Timestamp

    @staticmethod
    def from_milliseconds(start_ms: int, end_ms: int):
        assert start_ms < end_ms
        return Duration(Timestamp.from_milliseconds(start_ms), Timestamp.from_milliseconds(end_ms))


@dataclass
class SentenceSRT:
    text: str
    _duration: Duration
    token_timestamps: List[Duration]

    def __str__(self):
        return f"{self._duration.start} --> {self._duration.end}\n{self.text}"
    
    @property
    def start(self) -> str:
        return str(self._duration.start)
    
    @property
    def end(self) -> str:
        return str(self._duration.end)
    
    @property
    def duration(self) -> Timestamp:
        return self._duration.end - self._duration.start
    
    def move_forward(self, ts:Timestamp):
        self._duration.start += ts
        self._duration.end += ts
        for dura in self.token_timestamps:
            dura.start += ts
            dura.end += ts

    def move_backward(self, ts:Timestamp):
        self._duration.start -= ts
        self._duration.end -= ts
        for dura in self.token_timestamps:
            dura.start -= ts
            dura.end -= ts
    
    def shift(self, ts:Timestamp, forward=False):
        if forward:
            self._duration.start += ts
            self._duration.end += ts
            for dura in self.token_timestamps:
                dura.start += ts
                dura.end += ts
        else:
            self._duration.start -= ts
            self._duration.end -= ts
            for dura in self.token_timestamps:
                dura.start -= ts
                dura.end -= ts
    
    @staticmethod
    def from_dict(d: dict) -> 'SentenceSRT':
        return SentenceSRT(
            d['raw_text'],
            Duration.from_milliseconds(d['start'], d['end']),
            [Duration.from_milliseconds(*ts) for ts in d['timestamp']]
        )
    
    def to_dict(self) -> dict:
        return {
            'raw_text': self.text,
            'start': self.start,
            'end': self.end,
            'timestamp': [(ts.start.to_milliseconds(), ts.end.to_milliseconds()) for ts in self.token_timestamps]
        }

def get_resource_path(relative_path):
    """获取打包后或开发环境的资源路径"""
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

class VideoTranscriber:
    def __init__(self):
        # 初始化路径
        self.bin_dir = get_resource_path("bin")
        base_model_path = get_resource_path("models/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch")
        vad_model_path = get_resource_path("models/speech_fsmn_vad_zh-cn-16k-common-pytorch")
        punc_model_path = get_resource_path("models/punc_ct-transformer_zh-cn-common-vocab272727-pytorch")
        tables.model_classes[base_model_path] = tables.model_classes.get('paraformer-zh', None)
        tables.model_classes[vad_model_path] = tables.model_classes.get('fsmn-vad', None)
        tables.model_classes[punc_model_path] = tables.model_classes.get('ct-punc-c', None)
        self.model = AutoModel(
            model=base_model_path, # Non-ar asr
            vad_model=vad_model_path, # support any length
            punc_model=punc_model_path, # split to sentences
            disable_update=True
        )

    def get_ffmpeg_path(self, tool_name):
        """获取ffmpeg/ffprobe可执行文件路径"""
        if sys.platform == "win32":
            tool_name += ".exe"
        return os.path.join(self.bin_dir, tool_name)

    def video_to_audio(self, video_path, output_dir):
        """使用ffmpeg转换视频为wav音频"""
        ffmpeg_path = self.get_ffmpeg_path("ffmpeg")
        output_path = os.path.join(output_dir, "output_audio.wav")
        
        cmd = [
            ffmpeg_path,
            "-y",  # 覆盖输出文件
            "-i", video_path,
            "-vn",  # 禁用视频
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            output_path
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            return output_path
        except subprocess.CalledProcessError as e:
            print(f"FFmpeg error: {e.stderr.decode()}")
            return None

    def transcribe_audio(self, audio_path):
        """使用funasr进行语音识别"""
        rec_result = self.model.generate(
            audio_path,
            sentence_timestamp=True, 
            return_raw_text=True, 
            # is_final=True, 
            # hotword="",
            # output_dir=None,
            # cache={}
        )
        return [SentenceSRT.from_dict(sent) for sent in rec_result[0]['sentence_info']]

    def process(self, video_path, output_dir):
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 转换视频到音频
        audio_path = self.video_to_audio(video_path, output_dir)
        if not audio_path:
            return False
        
        # 语音识别
        result = self.transcribe_audio(audio_path)
        
        # 保存结果
        text_path = os.path.join(output_dir, "transcription.txt")
        with open(text_path, "w", encoding="utf-8") as f:
            for seg in result[0]["sentence_info"]:
                f.write(f"{seg['start']}->{seg['end']}: {seg['text']}\n")
        
        return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="视频转文字工具")
    parser.add_argument("input", help="输入视频文件路径")
    parser.add_argument("-o", "--output", default="./output", help="输出目录")
    args = parser.parse_args()

    transcriber = VideoTranscriber()
    if transcriber.process(args.input, args.output):
        print("转换成功！结果保存在：", args.output)
    else:
        print("转换失败")