import enum
import sys
from pathlib import Path
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

from constrained import constrained_alignment, constrained_edit_distance


class Node:
    def __init__(self, name: str, location: int):
        self.name = name
        self.location = location

    def __repr__(self):
        return f"{self.name}:{self.location}"

def read_file(path: Path):
    nodes = []
    chunks = []
    structure = []

    chunk_stack = []
    xmlparser = ParserCreate()
    def start_element(name: str, _: Dict[str, str]):
        nid = len(nodes)
        structure.append([])
        if len(chunk_stack) > 0:
            parent = chunk_stack[-1]
            structure[parent].append(nid)
        chunk_stack.append(nid)
        nodes.append(Node(name, xmlparser.CurrentByteIndex))
        chunks.append((chunk_stack[-1], xmlparser.CurrentByteIndex, False))

    def end_element(_: str):
        nid = chunk_stack.pop()
        pos = xmlparser.CurrentByteIndex
        if nid == chunks[-1][0]:
            pos = chunks[-1][1]
        chunks.append((nid, pos, True))

    xmlparser.StartElementHandler = start_element
    xmlparser.EndElementHandler = end_element

    with open(path, "rb") as f:
        xmlparser.ParseFile(f)
    return (nodes, chunks, structure)

(nodes_a, chunks_a, structure_a) = read_file(Path("file_a.xml"))
(nodes_b, chunks_b, structure_b) = read_file(Path("file_b.xml"))

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
    Kind.ADD,
    Kind.NOTHING,
    Kind.NOTHING,
    Kind.NOTHING,
    Kind.NOTHING,
    Kind.REMOVE,
]

def get_next(some_iterable, window=1):
    items, nexts = tee(some_iterable, 2)
    nexts = islice(nexts, window, None)
    return zip_longest(items, nexts)

def scan_for_end_of_tag(buffer):
    pos = 0
    # @HACK This is probably not very reliable. I have a patch pending in
    # python to expose the expat field we need, but this will do for now.
    while buffer[pos] != b'>'[0]:
        pos += 1

    return pos

def cost(ai, bi, data):
    if ai is None:
        return 2
    if bi is None:
        return 2

    (a, b) = data
    if a[ai] == b[bi]:
        return 0

    return 1

edit_cost, trace_matrix = constrained_edit_distance(structure_a, structure_b, cost, (nodes_a, nodes_b))
assert(trace_matrix is not None)
print(structure_a, structure_b)
alignment = constrained_alignment(structure_a, structure_b, trace_matrix)

print(alignment)
for (left, right) in alignment:
    state[right] = Kind.NOTHING
    if left == -1:
        state[right] = Kind.ADD

with open("file_b.xml", "rb") as f:
    f.seek(0, SEEK_END)
    file_len = f.tell()
    f.seek(0)
    sys.stdout.flush()
    for (i, (chunk, next)) in enumerate(get_next(chunks_b)):
        len = file_len - chunk[1]
        if next is not None:
            len = next[1] - chunk[1]

        buffer = f.read(len)

        chunk_state = state[chunk[0]]
        if chunk_state == Kind.NOTHING:
            sys.stdout.buffer.write(buffer)
            continue

        code = "+" if chunk_state == Kind.ADD else "-"

        if not chunk[2]:
            sys.stdout.write(f"{{{code}")
            sys.stdout.flush()
            sys.stdout.buffer.write(buffer)
        else:
            end = scan_for_end_of_tag(buffer)
            sys.stdout.buffer.write(buffer[:end+1])
            sys.stdout.write(f"{code}}}")
            sys.stdout.flush()
            sys.stdout.buffer.write(buffer[end+1:])


# parser = etree.XMLParser()
# print(parser._parser_context)
# root_a = etree.parse("file_a.xml", parser)
# print(root_a)

exit(1)
