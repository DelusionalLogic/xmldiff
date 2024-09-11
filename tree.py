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

def start_element(name: str, attrs: Dict[str, str]):
    nodes.append(Node(name, xmlparser.CurrentByteIndex))

xmlparser.StartElementHandler = start_element
# xmlparser.EndElementHandler = end_element

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
with open("file_a.xml", "rb") as f:
    f.seek(0, SEEK_END)
    file_len = f.tell()
    for (i, ((node, next), state)) in enumerate(zip(get_next(nodes), state)):
        sys.stdout.write(f"<<{state}>>")
        sys.stdout.flush()
        # if i % 2 == 0:
        #     sys.stdout.buffer.write(b"\033[46m")
        # else:
        #     sys.stdout.buffer.write(b"\033[42m")
        f.seek(node.location)
        len = file_len - node.location
        if next is not None:
            len = next.location - node.location
        sys.stdout.buffer.write(f.read(len))

# parser = etree.XMLParser()
# print(parser._parser_context)
# root_a = etree.parse("file_a.xml", parser)
# print(root_a)

