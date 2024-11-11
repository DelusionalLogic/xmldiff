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
from difflib import SequenceMatcher

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

    # We could do something here to collapse diffs that butt up against each
    # other, but I think this might be mer ergonomic
    seq = SequenceMatcher(None, a_buff, b_buff)
    for (op, i1, i2, j1, j2) in seq.get_opcodes():
        if op == 'equal':
            out_stream.write(a_buff[i1:i2])
        elif op == 'insert':
            out_stream.write(b"\x7b+")
            out_stream.write(b_buff[j1:j2])
            out_stream.write(b"+\x7d")
        elif op == 'delete':
            out_stream.write(b"\x7b-")
            out_stream.write(a_buff[i1:i2])
            out_stream.write(b"-\x7d")
        elif op == 'replace':
            out_stream.write(b"\x7b-")
            out_stream.write(a_buff[i1:i2])
            out_stream.write(b"-\x7d")
            out_stream.write(b"\x7b+")
            out_stream.write(b_buff[j1:j2])
            out_stream.write(b"+\x7d")
        else: raise Exception(op)

    out_stream.flush()

class Take(Enum):
    LEFT = enum.auto()
    RIGHT = enum.auto()
    BOTH = enum.auto()

class Peekable:
    def __init__(self, inner):
        self.inner = inner
        self.slot = None
        self.loaded = False

    def peek(self):
        if not self.loaded:
            self.loaded = True
            self.slot = next(self.inner)

        return self.slot


    def __next__(self):
        if not self.loaded:
            self.peek()

        cur = self.slot
        self.loaded = False
        self.slot = None
        return cur

def merge_trees(fa, fb, out_stream):
    (nodes_a, chunks_a, structure_a) = read_file(fa)
    (nodes_b, chunks_b, structure_b) = read_file(fb)

    edit_cost, trace_matrix = constrained_edit_distance(structure_a, structure_b, cost, (nodes_a, nodes_b))
    assert(trace_matrix is not None)
    print(edit_cost)
    alignment = constrained_alignment(structure_a, structure_b, trace_matrix)

    fb.seek(0, SEEK_END)
    file_b_len = fb.tell()
    fb.seek(0)

    fa.seek(0, SEEK_END)
    file_a_len = fa.tell()
    fa.seek(0)

    take_list = []
    # Track if the node was changed or matched so that we know to wait with
    # consuming it
    # @Speed would it be faster to precompute this?
    a_stack = []
    b_stack = []
    ait = Peekable(iter(chunks_a))
    bit = Peekable(iter(chunks_b))
    for (left, right) in alignment:
        # Match a single open tag
        if left >= 0 and right >= 0:
            next(ait)
            next(bit)
            a_stack.append(0)
            b_stack.append(0)
            take_list.append(Take.BOTH)
        elif left >= 0:
            next(ait)
            a_stack.append(1)
            take_list.append(Take.LEFT)
        elif right >= 0:
            next(bit)
            b_stack.append(1)
            take_list.append(Take.RIGHT)

        # Match the closing tags
        while True:
            # If we have a closing chunk on both streams, and both stacks have
            # a matching consume queued up, we take it. Otherwise we check if
            # either has an unbalanced consume and a closing chunk.
            try:
                a_next = ait.peek()
            except StopIteration:
                a_next = None

            try:
                b_next = bit.peek()
            except StopIteration:
                b_next = None

            if a_next is not None and a_next[2] and b_next is not None and b_next[2] and a_stack[-1] == 0 and b_stack[-1] == 0:
                next(ait)
                next(bit)
                a_stack.pop()
                b_stack.pop()
                take_list.append(Take.BOTH)
            elif a_next is not None and a_next[2] and a_stack[-1] == 1:
                next(ait)
                a_stack.pop()
                take_list.append(Take.LEFT)
            elif b_next is not None and b_next[2] and b_stack[-1] == 1:
                next(bit)
                b_stack.pop()
                take_list.append(Take.RIGHT)
            else: break # Stop when we didn't change the state


    # Keep track of if we are adding, deleting, or matching currently
    state = 0
    sys.stdout.flush()
    a_num = 0
    b_num = 0
    for action in take_list:
        if action == Take.BOTH:
            if state != 0:
                out_stream.write(b"-\x7d" if state == 1 else b"+\x7d")
            state = 0
            write_chunk_match(chunks_a, chunks_b, a_num, b_num, fa, fb, file_a_len, file_b_len, out_stream)
            a_num += 1
            b_num += 1
        elif action == Take.LEFT:
            if state != 1:
                out_stream.write(b"\x7b-" if state == 0 else b"+\x7d\x7b-")
            state = 1
            write_chunk(chunks_a, a_num, fa, file_a_len, out_stream)
            a_num += 1
        elif action == Take.RIGHT:
            if state != 2:
                out_stream.write(b"\x7b+" if state == 0 else b"-\x7d\x7b+")
            state = 2
            write_chunk(chunks_b, b_num, fb, file_b_len, out_stream)
            b_num += 1
        else: raise Exception()

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
