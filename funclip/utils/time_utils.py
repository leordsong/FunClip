from dataclasses import dataclass
import re

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
