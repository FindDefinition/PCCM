
import bisect
import dataclasses
from typing import List, Tuple
from pathlib import Path 


@dataclasses.dataclass
class Replace:
    content: str 
    start: int 
    end: int # [start:end], [], not[)

def interval_remain(interval_sorted: List[Tuple[int, int]], start: int, end: int):
    before = start 
    res: List[Tuple[int, int]] = []
    for x, y in interval_sorted:
        # if x != before:
        res.append((before, x))
        before = y 
    res.append((before, end))
    return res 


class Source:
    source: str 
    length: int 
    line_offsets: List[Tuple[int, int]]
    lines: List[str]
    def __init__(self, source: str) -> None:
        self.source = source 
        self.length = len(source)
        self.line_offsets = []
        self.lines = []
        start = 0
        for i in range(self.length):
            val = source[i]
            if val == "\n":
                self.line_offsets.append((start, i))
                self.lines.append(source[start:i])
                start = i + 1
        self.line_offsets.append((start, self.length))
        self.line_offset_starts = [x[0] for x in self.line_offsets]

        self.lines.append(source[start:self.length])

    @property
    def num_lines(self):
        return len(self.lines)

    def lineno_col_to_offset(self, lineno: int, column: int):
        lineno -= 1
        column -= 1
        if lineno < 0 or lineno >= self.num_lines:
            return -1 
        start, end = self.line_offsets[lineno]
        if column < 0:
            column += end - start + 1
        if column >= end:
            return -1 
        return start + column 

    def offset_to_lineno_col(self, offset: int):
        if offset < 0 or offset > self.length:
            return (-1, -1)
        pos = bisect.bisect_left(self.line_offset_starts, offset)
        if pos == len(self.line_offsets):
            return (self.num_lines, offset - self.line_offsets[-1][0] + 1)
        
        start, end = self.line_offsets[pos - 1]
        return (pos, offset - start + 1)

    def get_content(self, lineno: int, column: int, end_lineno: int, end_column: int):
        start = self.lineno_col_to_offset(lineno, column)
        end = self.lineno_col_to_offset(end_lineno, end_column)
        return self.source[start:end + 1]

    def get_content_by_offset(self, start: int, end: int):
        return self.source[start:end + 1]


def execute_modifiers(source: str, modifiers: List[Replace]):
    if not modifiers:
        return source 

    modifiers.sort(key=lambda x: x.start)
    before = -1
    intervals: List[Tuple[int, int]] = []
    for m in modifiers:
        if m.start < before:
            return None 
        if m.end < m.start:
            return None 
        before = m.end
        intervals.append((m.start, m.end))
    if before > len(source):
        return None 

    remain_intervals = interval_remain(intervals, 0, len(source))
    res = ""
    index = 0
    for x, y in remain_intervals:
        res += source[x:y]
        if index < len(modifiers):
            res += modifiers[index].content
        index += 1
    return res 


def main():
    with (Path(__file__)).open("r") as f:
        content = f.read()

    s = Source(content)

    print(s.get_content(83, 1, 83, -1))

if __name__ == "__main__":
    main()