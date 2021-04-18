"""
we need simple structured code repr to print beautiful code.
class Block:
    prefix: str
    body: List[str|Block]
    suffix: str
"""
from typing import List, Optional, Union


class Block:
    def __init__(self,
                 prefix: str,
                 body: Optional[List[Union["Block", str]]] = None,
                 suffix: str = "",
                 indent: Optional[int] = None):
        self.prefix = prefix
        if body is None:
            body = []
        self.body = body
        self.suffix = suffix
        self.indent = indent

    def __repr__(self) -> str:
        return "\n".join(generate_code(self, 0, 2))


def generate_code(block: Union[Block, str], start_col_offset: int,
                  indent: int):
    col_str = " " * start_col_offset
    if isinstance(block, str):
        block_lines = block.split("\n")
        return [col_str + l for l in block_lines]
    res = []  # type: List[str]
    prefix = block.prefix
    next_indent = indent
    if block.indent is not None:
        next_indent = block.indent
    if prefix:
        prefix_lines = prefix.split("\n")
        res.extend([col_str + l for l in prefix_lines])
    for child in block.body:
        res.extend(generate_code(child, start_col_offset + next_indent,
                                 indent))
    if block.suffix:
        suffix_lines = block.suffix.split("\n")
        res.extend([col_str + l for l in suffix_lines])
    return res


def generate_code_list(blocklist: List[Union[Block, str]],
                       start_col_offset: int, indent: int):
    res = []  # type: List[str]
    for b in blocklist:
        res.extend(generate_code(b, start_col_offset, indent))
    return res
