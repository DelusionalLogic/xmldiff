import enum
import sys
from enum import (
    Enum,
)
from io import (
    SEEK_END,
    SEEK_SET,
)
from itertools import (
    islice,
    tee,
    zip_longest,
)
from pathlib import (
    Path,
)
from typing import (
    Dict,
)
from xml.parsers.expat import (
    ParserCreate,
)

from edist.sed import (
    sed_string,
)

from constrained import (
    constrained_alignment,
    constrained_edit_distance,
)

file_a = Path("file_a.xml")
file_b = Path("file_b.xml")

class Node:
    def __init__(self, name: str, location: int):
        self.name = name
        self.location = location

    def __repr__(self):
        return f"{self.name}:{self.location}"

def read_file(f):
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
        chunks.append((nid, pos, True))

    xmlparser.ordered_attributes = True
    xmlparser.StartElementHandler = start_element
    xmlparser.EndElementHandler = end_element

    xmlparser.ParseFile(f)
    return (nodes, chunks, structure)

class Kind(Enum):
    NOTHING = enum.auto()
    ADD = enum.auto()
    REMOVE = enum.auto()

def get_next(some_iterable, window=1):
    items, nexts = tee(some_iterable, 2)
    nexts = islice(nexts, window, None)
    return zip_longest(items, nexts)

def scan_for_end_of_tag(buffer):
    # @HACK hacks on top of hacks
    if buffer[0] != b'<'[0]:
        return 0
    pos = 0
    # @HACK This is probably not very reliable. I have a patch pending in
    # python to expose the expat field we need, but this will do for now.
    while buffer[pos] != b'>'[0]:
        pos += 1

    return pos

def cost(ai, bi, data):
    (a, b) = data
    if ai is None:
        return len(b[bi].name)
    if bi is None:
        return len(a[ai].name)

    if a[ai] == b[bi]:
        return 0

    return sed_string(a[ai].name, b[bi].name)

def read_chunk(chunks, index, file, file_len):
    chunk = chunks[index]
    nchunk = chunks[index+1] if index+1 < len(chunks) else None

    clen = file_len - chunk[1]
    if nchunk is not None:
        clen = nchunk[1] - chunk[1]

    file.seek(chunk[1], SEEK_SET)
    buffer = file.read(clen)
    return buffer

def write_chunk(chunks, index, file, file_len, out_stream=sys.stdout.buffer, color=None):
    buffer = read_chunk(chunks, index, file, file_len)

    if color is not None:
        buffer = buffer.replace(b"\n", f"\\n\x1b\x5b0m\n\x1b\x5b{color}m".encode())
        buffer = buffer.replace(b"\t", f"    ".encode())
    out_stream.write(buffer)
    out_stream.flush()

def write_chunk_match(chunks_a, chunks_b, a_i, b_i, fa, fb, a_len, b_len, out_stream=sys.stdout.buffer):
    a_buff = read_chunk(chunks_a, a_i, fa, a_len)
    b_buff = read_chunk(chunks_b, b_i, fb, b_len)

    if a_buff == b_buff:
        out_stream.write(b_buff)
    else:
        out_stream.write(b"\x7b-")
        out_stream.write(a_buff)
        out_stream.write(b"-\x7d\x7b+")
        out_stream.write(b_buff)
        out_stream.write(b"+\x7d")

    out_stream.flush()

def merge_trees(fa, fb, out_stream):
    (nodes_a, chunks_a, structure_a) = read_file(fa)
    (nodes_b, chunks_b, structure_b) = read_file(fb)

    edit_cost, trace_matrix = constrained_edit_distance(structure_a, structure_b, cost, (nodes_a, nodes_b))
    assert(trace_matrix is not None)
    print(edit_cost)
    alignment = constrained_alignment(structure_a, structure_b, trace_matrix)

    state_a = [ Kind.NOTHING for _ in chunks_a ]
    state_b = [ Kind.NOTHING for _ in chunks_b ]

    print(alignment)
    for (left, right) in alignment:
        if left >= 0:
            state_a[left] = Kind.NOTHING if right >= 0 else Kind.REMOVE
        if right >= 0:
            state_b[right] = Kind.NOTHING if left >= 0 else Kind.ADD

    fb.seek(0, SEEK_END)
    file_b_len = fb.tell()
    fb.seek(0)

    fa.seek(0, SEEK_END)
    file_a_len = fa.tell()
    fa.seek(0)

    sys.stdout.flush()
    # Two variabled to keep track of where in each file we are. We consume
    # a chunk from the file when we output it in the result
    a_num = 0
    b_num = 0
    # Keep track of if we are adding, deleting, or matching currently
    state = 0
    # Track of deep we are in the tree. Used to match up the close chunks. May
    # not be necessary if we get sufficiently clever with the state field
    a_stack = []
    b_stack = []
    for (left, right) in alignment:
        # Match a single open tag
        if left >= 0 and right >= 0:
            if state != 0:
                out_stream.write(b"-\x7d" if state == 1 else b"+\x7d")
            state = 0
            assert not chunks_a[a_num][2] and not chunks_b[b_num][2]
            # @INCOMPLETE: We should do some deeper diff here.
            write_chunk_match(chunks_a, chunks_b, a_num, b_num, fa, fb, file_a_len, file_b_len, out_stream)
            a_num += 1
            b_num += 1
            a_stack.append(0)
            b_stack.append(0)
        elif left >= 0:
            if state != 1:
                out_stream.write(b"\x7b-" if state == 0 else b"+\x7d\x7b-")
            state = 1
            write_chunk(chunks_a, a_num, fa, file_a_len, out_stream)
            a_num += 1
            a_stack.append(1)
        elif right >= 0:
            if state != 2:
                out_stream.write(b"\x7b+" if state == 0 else b"-\x7d\x7b+")
            state = 2
            write_chunk(chunks_b, b_num, fb, file_b_len, out_stream)
            b_num += 1
            b_stack.append(1)

        # Match the closing tags
        while True:
            # If we have a closing chunk on both streams, and both stacks have
            # a matching consume queued up, we take it. Otherwise we check if
            # either has an unbalanced consume and a closing chunk.
            if a_num < len(chunks_a) and chunks_a[a_num][2] and b_num < len(chunks_b) and chunks_b[b_num][2] and a_stack[-1] == 0 and b_stack[-1] == 0:
                if state != 0:
                    out_stream.write(b"-\x7d" if state == 1 else b"+\x7d")
                state = 0
                write_chunk_match(chunks_a, chunks_b, a_num, b_num, fa, fb, file_a_len, file_b_len, out_stream)
                a_num += 1
                b_num += 1
                a_stack.pop()
                b_stack.pop()
            elif a_num < len(chunks_a) and chunks_a[a_num][2] and a_stack[-1] == 1:
                if state != 1:
                    out_stream.write(b"\x7b-" if state == 0 else b"+\x7d\x7b-")
                state = 1
                write_chunk(chunks_a, a_num, fa, file_a_len, out_stream)
                a_num += 1
                a_stack.pop()
            elif b_num < len(chunks_b) and chunks_b[b_num][2] and b_stack[-1] == 1:
                if state != 2:
                    out_stream.write(b"\x7b+" if state == 0 else b"-\x7d\x7b+")
                state = 2
                write_chunk(chunks_b, b_num, fb, file_b_len, out_stream)
                b_num += 1
                b_stack.pop()
            else: break # Stop when we didn't change the state

if __name__ == "__main__":
    with open(file_a, "rb") as fa, open(file_b, "rb") as fb:
        # fa.seek(0, SEEK_END)
        # file_a_len = fa.tell()
        # fa.seek(0)

        # swap = False
        # print(chunks_a)
        # for i, _ in enumerate(chunks_a):
        #     if swap:
        #         color = 45
        #         sys.stdout.buffer.write(b"\x1b\x5b45m")
        #     else:
        #         color = 44
        #         sys.stdout.buffer.write(b"\x1b\x5b44m")
        #     sys.stdout.flush()
        #     write_chunk(chunks_a, i, fa, file_a_len, color=color)
        #     swap = not swap
        # sys.stdout.buffer.write(b"\x1b\x5b0m")

        merge_trees(fa, fb, sys.stdout.buffer)

# parser = etree.XMLParser()
# print(parser._parser_context)
# root_a = etree.parse("file_a.xml", parser)
# print(root_a)
