from dataclasses import dataclass
import re
from typing import List

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