import enum
import sys
from enum import (
    Enum,
)
from io import (
    SEEK_END,
)
from itertools import (
    islice,
    tee,
    zip_longest,
)
from typing import (
    Dict,
)
from xml.parsers.expat import (
    ParserCreate,
)


class Node:
    def __init__(self, name: str, location: int):
        self.name = name
        self.location = location

    def __repr__(self):
        return f"{self.name}:{self.location}"

xmlparser = ParserCreate()

nodes = [
]

chunks = [
]

chunk_stack = []
def start_element(name: str, attrs: Dict[str, str]):
    chunk_stack.append(len(nodes))
    nodes.append(Node(name, xmlparser.CurrentByteIndex))
    chunks.append((chunk_stack[-1], xmlparser.CurrentByteIndex))

def end_element(name: str):
    nid = chunk_stack.pop()
    if nid != chunks[-1][0]:
        chunks.append((nid, xmlparser.CurrentByteIndex))

xmlparser.StartElementHandler = start_element
xmlparser.EndElementHandler = end_element

with open("file_a.xml", "rb") as f:
    xmlparser.ParseFile(f)

class Kind(Enum):
    NOTHING = enum.auto()
    ADD = enum.auto()
    REMOVE = enum.auto()

state = [
    Kind.NOTHING,
    Kind.NOTHING,
    Kind.ADD,
    Kind.NOTHING,
    Kind.NOTHING,
    Kind.NOTHING,
    Kind.NOTHING,
    Kind.NOTHING,
    Kind.NOTHING,
]

def get_next(some_iterable, window=1):
    items, nexts = tee(some_iterable, 2)
    nexts = islice(nexts, window, None)
    return zip_longest(items, nexts)

print(nodes)
print(chunks)
with open("file_a.xml", "rb") as f:
    f.seek(0, SEEK_END)
    file_len = f.tell()
    f.seek(0)
    for (i, (chunk, next)) in enumerate(get_next(chunks)):
        sys.stdout.write(f"<<{state[chunk[0]]}>>")
        sys.stdout.flush()
        # if i % 2 == 0:
        #     sys.stdout.buffer.write(b"\033[46m")
        # else:
        #     sys.stdout.buffer.write(b"\033[42m")
        # f.seek(chunk[1])
        len = file_len - chunk[1]
        if next is not None:
            len = next[1] - chunk[1]
        sys.stdout.buffer.write(f.read(len))

# parser = etree.XMLParser()
# print(parser._parser_context)
# root_a = etree.parse("file_a.xml", parser)
# print(root_a)

