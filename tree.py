import enum
import io
import sys
from difflib import SequenceMatcher
from enum import Enum
from pathlib import Path
from typing import (
    BinaryIO,
    Dict,
)
from xml.parsers.expat import ParserCreate

import numpy as np
from edist.sed import sed_string

from constrained import (
    constrained_alignment,
    constrained_edit_distance,
)


class OutputState:
    state_color = [b"0", b"31", b"32"]

    def __init__(self, stream: BinaryIO):
        self.stream = stream
        self.state = 0
        self.color = "0"

    def _switch_state(self, new_state: int):
        if self.state == new_state:
            return

        state_chars = [b" ", b"-", b"+"]

        if self.state != 0:
            self.stream.write(state_chars[self.state])
            self.stream.write(b"\x7d")

        self.stream.write(b"\033\133")
        self.stream.write(self.state_color[new_state])
        self.stream.write(b"m")

        if new_state != 0:
            self.stream.write(b"\x7b")
            self.stream.write(state_chars[new_state])

        self.state = new_state

    def output(self, data: bytes, state: int):
        self._switch_state(state)
        self.stream.write(data)


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

    xmlparser.ordered_attributes = False
    xmlparser.StartElementHandler = start_element
    xmlparser.EndElementHandler = end_element

    xmlparser.ParseFile(f)
    return (nodes, chunks, structure)


def read_chunk(chunks, index, file, file_len):
    chunk = chunks[index]
    nchunk = chunks[index + 1] if index + 1 < len(chunks) else None

    clen = file_len - chunk[1]
    if nchunk is not None:
        clen = nchunk[1] - chunk[1]

    file.seek(chunk[1], io.SEEK_SET)
    buffer = file.read(clen)
    return buffer


def write_chunk(chunks, index, file, file_len, out_stream: OutputState, state):
    buffer = read_chunk(chunks, index, file, file_len)
    out_stream.output(buffer, state)


# @COMPLETENESS: You might want to be able to configure the whitespace somehow
def write_chunk_match(
    chunks_a, chunks_b, a_i, b_i, fa, fb, a_len, b_len, out_stream: OutputState
):
    a_buff = read_chunk(chunks_a, a_i, fa, a_len).split(b" ")
    b_buff = read_chunk(chunks_b, b_i, fb, b_len).split(b" ")

    seq = SequenceMatcher(None, a_buff, b_buff, False)
    for op, i1, i2, j1, j2 in seq.get_opcodes():
        if op == "equal":
            out_stream.output(b" ".join(a_buff[i1:i2]), 0)
        elif op == "insert":
            out_stream.output(b" ".join(b_buff[j1:j2]), 2)
        elif op == "delete":
            out_stream.output(b" ".join(a_buff[i1:i2]), 1)
        elif op == "replace":
            out_stream.output(b" ".join(a_buff[i1:i2]), 1)
            out_stream.output(b" ".join(b_buff[j1:j2]), 2)
        else:
            raise Exception(op)


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


def merge_trees(fa, fb, out_stream: OutputState):
    (nodes_a, chunks_a, structure_a) = read_file(fa)
    (nodes_b, chunks_b, structure_b) = read_file(fb)

    fb.seek(0, io.SEEK_END)
    file_b_len = fb.tell()
    fb.seek(0)

    fa.seek(0, io.SEEK_END)
    file_a_len = fa.tell()
    fa.seek(0)

    a_to_chunks = []
    for i, (_, _, close) in enumerate(chunks_a):
        if not close:
            a_to_chunks.append(i)

    b_to_chunks = []
    for i, (_, _, close) in enumerate(chunks_b):
        if not close:
            b_to_chunks.append(i)

    cost = np.zeros((len(nodes_a) + 1, len(nodes_b) + 1))
    for i, (_, cid) in enumerate(zip(nodes_a, a_to_chunks)):
        # @PERF: We don't actually really need the chunks themselves here,
        # merely their length which we can calculate without doing any io. This
        # is easier with the current interface though.
        a_chunk = read_chunk(chunks_a, cid, fa, file_a_len)
        cost[i, 0] = len(a_chunk)

    for i, (_, cid) in enumerate(zip(nodes_b, b_to_chunks)):
        b_chunk = read_chunk(chunks_b, cid, fb, file_b_len)
        cost[0, i] = len(b_chunk)

    for ai, (a, a_cid) in enumerate(zip(nodes_a, a_to_chunks)):
        a_chunk = read_chunk(chunks_a, a_cid, fa, file_a_len)

        for bi, (b, b_cid) in enumerate(zip(nodes_b, b_to_chunks)):
            cost_value = 0

            b_chunk = read_chunk(chunks_b, b_cid, fb, file_b_len)

            if a != b:
                cost_value = sed_string(a_chunk.decode(), b_chunk.decode())

            cost[ai + 1, bi + 1] = cost_value

    _, trace_matrix = constrained_edit_distance(structure_a, structure_b, cost)
    assert trace_matrix is not None
    alignment = constrained_alignment(structure_a, structure_b, trace_matrix)

    take_list = []
    # Track if the node was changed or matched so that we know to wait with
    # consuming it
    # @PERF would it be faster to precompute this?
    a_stack = []
    b_stack = []
    # @CLEANUP The obly part of this we are using open/close flag. Could we
    # separate that out to slim this down?
    ait = Peekable(iter(chunks_a))
    bit = Peekable(iter(chunks_b))
    for left, right in alignment:
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

            if (
                a_next is not None
                and a_next[2]
                and b_next is not None
                and b_next[2]
                and a_stack[-1] == 0
                and b_stack[-1] == 0
            ):
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
            else:
                break  # Stop when we didn't change the state

    a_num = 0
    b_num = 0
    for action in take_list:
        if action == Take.BOTH:
            write_chunk_match(
                chunks_a,
                chunks_b,
                a_num,
                b_num,
                fa,
                fb,
                file_a_len,
                file_b_len,
                out_stream,
            )
            a_num += 1
            b_num += 1
        elif action == Take.LEFT:
            write_chunk(chunks_a, a_num, fa, file_a_len, out_stream, 1)
            a_num += 1
        elif action == Take.RIGHT:
            write_chunk(chunks_b, b_num, fb, file_b_len, out_stream, 2)
            b_num += 1
        else:
            raise Exception()


if __name__ == "__main__":
    file_a = Path(
        "/Users/delusional/axiom/5.12.3/Projects/CM_Reference_Data_Calc_Inputs/Branches/5_12_3/Aggregation/Ref_Reporting_Entity.xml"
    )
    file_b = Path(
        "/Users/delusional/axiom/5.12.4/Projects/CM_Reference_Data_Calc_Inputs/Branches/5_12_4/Aggregation/Ref_Reporting_Entity.xml"
    )
    out = OutputState(sys.stdout.buffer)
    with open(file_a, "rb") as fa, open(file_b, "rb") as fb:
        merge_trees(fa, fb, out)
