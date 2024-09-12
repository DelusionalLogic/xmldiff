from enum import (
    Enum,
)
from typing import (
    Any,
    Callable,
    Iterator,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

import edist.sed
import edist.tree_edits
import numpy as np

a: List[int] = [0, 1, 2, 3]
a_adj: List[List[int]] = [[1, 3], [2], [], []]

b: List[int] = [0, 1, 3, 2, 4]
b_adj: List[List[int]] = [[1, 2], [], [3, 4], [], []]

def cost(ai: Optional[int], bi: Optional[int], data: Tuple[List[int], List[int]]) -> int:
    if ai is None:
        return 1
    if bi is None:
        return 1

    (a, b) = data
    if a[ai] == b[bi]:
        return 0

    return 1

class Cmd(Enum):
    UNSET = 0
    MATCH = 1
    REMOVE = 2
    ADD = 3

def argmin(a: Iterator[int]) -> Tuple[int, int]:
    min_val = next(a)
    min_arg = 0
    for i, v in enumerate(a):
        if v < min_val:
            min_val = v
            min_arg = i+1
    return (min_arg, min_val)

type Alignment = List[Tuple[int, int]]
type TraceMatrix = List[List[Tuple[Cmd, Union[int, Alignment, None]]]]

T = TypeVar("T")
def _constrained_edit_distance_core(
    a_adj: List[List[int]],
    b_adj: List[List[int]],
    cost: Callable[[Optional[int], Optional[int], T], int],
    data: T,
    cost_n: np.ndarray,
    cost_f: np.ndarray,
    trace_n: TraceMatrix,
    trace_f: TraceMatrix
) -> None:
    cost_n[0][0] = 0
    cost_f[0][0] = 0
    for i in reversed(range(len(a_adj))):
        cost_f[i+1][0] = 0
        for c in a_adj[i]:
            cost_f[i+1][0] += cost_n[c+1][0]
        cost_n[i+1][0] = cost_f[i+1][0] + cost(i, None, data)
    for j in reversed(range(len(b_adj))):
        cost_f[0][j+1] = 0
        for c in b_adj[j]:
            cost_f[0][j+1] += cost_n[0][c+1]
        cost_n[0][j+1] = cost_f[0][j+1] + cost(None, j, data)

    def seq_dist(ai: Optional[int], bi: Optional[int]) -> int:
        if ai is None:
            assert bi is not None
            return cost_n[0][bi+1]
        if bi is None:
            assert ai is not None
            return cost_n[ai+1][0]
        return cost_n[ai+1][bi+1]

    for i in reversed(range(len(a_adj))):
        print(f"Computing column {i}")
        for j in reversed(range(len(b_adj))):
            choices = [
                edist.sed.sed(a_adj[i], b_adj[j], delta=seq_dist)
            ]
            alignment : Alignment = [(e._right, e._left) for e in edist.sed.sed_backtrace(a_adj[i], b_adj[j], delta=seq_dist)]
            f_traces : List[Tuple[Cmd, int | Alignment]] = [
                (Cmd.MATCH, alignment)
            ]

            fmin_s = None
            if len(a_adj[i]) > 0:
                fmin_s, fval = argmin((cost_f[a_adj[i][s]+1][j+1] - cost_f[a_adj[i][s]+1][0] for s in range(len(a_adj[i]))))
                choices.append(cost_f[i+1][0] + fval)
                f_traces.append((Cmd.REMOVE, fmin_s))

            fmin_t = None
            if len(b_adj[j]) > 0:
                fmin_t, fval = argmin((cost_f[i+1][b_adj[j][t]+1] - cost_f[0][b_adj[j][t]+1] for t in range(len(b_adj[j]))))
                choices.append(cost_f[0][j+1] + fval)
                f_traces.append((Cmd.ADD, fmin_t))

            f_min = np.argmin(choices)
            cost_f[i+1][j+1] = choices[f_min]
            trace_f[i+1][j+1] = f_traces[f_min]

            choices = [
                cost_f[i+1][j+1] + cost(i, j, data)
            ]
            n_traces : List[Tuple[Cmd, Optional[int]]] = [
                (Cmd.MATCH, None)
            ]

            n_min_s = None
            if len(a_adj[i]) > 0:
                n_min_s, nval = argmin((cost_n[a_adj[i][s]+1][j+1] - cost_n[a_adj[i][s]+1][0] for s in range(len(a_adj[i]))))
                choices.append(cost_n[i+1][0] + nval)
                n_traces.append((Cmd.REMOVE, n_min_s))

            n_min_t = None
            if len(b_adj[j]) > 0:
                n_min_t, nval = argmin((cost_n[i+1][b_adj[j][t]+1] - cost_n[0][b_adj[j][t]+1] for t in range(len(b_adj[j]))))
                choices.append(cost_n[0][j+1] + nval)
                n_traces.append((Cmd.ADD, n_min_t))

            n_min = np.argmin(choices)
            cost_n[i+1][j+1] = choices[n_min]
            trace_n[i+1][j+1] = n_traces[n_min]

def constrained_edit_distance(
    a_adj: List[List[int]],
    b_adj: List[List[int]],
    cost: Callable[[Optional[int], Optional[int], T], int],
    data: T = None,
) -> Tuple[int, Optional[Tuple[TraceMatrix, TraceMatrix]]]:
    if len(a_adj) == 0 and len(b_adj) == 0:
        return 0, None

    cost_n = np.empty((len(a_adj)+1, len(b_adj)+1), dtype=int)
    cost_f = np.empty((len(a_adj)+1, len(b_adj)+1), dtype=int)
    trace_n: TraceMatrix = [[(Cmd.UNSET, None) for _ in range(len(b_adj)+1)] for _ in range(len(a_adj)+1)]
    trace_f: TraceMatrix = [[(Cmd.UNSET, None) for _ in range(len(b_adj)+1)] for _ in range(len(a_adj)+1)]

    print(cost_n.shape)
    print(cost_f.shape)

    _constrained_edit_distance_core(a_adj, b_adj, cost, data, cost_n, cost_f, trace_n, trace_f)
    return cost_n[1][1].item(), (trace_f, trace_n)

def constrained_alignment(
    a_adj: List[List[int]],
    b_adj: List[List[int]],
    trace: Tuple[TraceMatrix, TraceMatrix]
) -> Alignment:
    if len(a_adj) == 0 and len(b_adj) == 0:
        return []

    alignment = []
    to_compute: List[Tuple[int, int]] = [
        (1, 1)
    ]

    def do_subtree(alignment: Alignment, cursor: int, tree: List[List[int]], op: Cmd) -> None:
        remain = 1
        while remain > 0:
            if op == Cmd.REMOVE:
                alignment.append((cursor, -1))
            else:
                alignment.append((-1, cursor))
            remain += len(tree[cursor])
            cursor += 1
            remain -= 1

    (trace_f, trace_n) = trace
    while len(to_compute) > 0:
        i, j = to_compute.pop()
        if i == -1:
            do_subtree(alignment, j-1, b_adj, Cmd.ADD)
            continue
        if j == -1:
            do_subtree(alignment, i-1, a_adj, Cmd.REMOVE)
            continue
        (t_match, t_arg) = trace_n[i][j]

        if t_match == Cmd.MATCH:
            alignment.append((i-1, j-1))
            (f_match, f_arg) = trace_f[i][j]

            # @INCOMPLETE We might need this later. I don't have a case that
            # tests it yet
            # while f_match != Cmd.MATCH:
            #     assert isinstance(f_arg, int)

            #     if f_match == Cmd.ADD:
            #         siblings = b_adj[j-1]
            #         tree = b_adj
            #     elif f_match == Cmd.REMOVE:
            #         siblings = a_adj[i-1]
            #         tree = a_adj
            #     else:
            #         raise Exception()

            #     resolved = None
            #     for t, t_node in enumerate(siblings):
            #         if t == f_arg:
            #             resolved = t_node
            #         else:
            #             do_subtree(alignment, t_node, tree, f_match)
            #     assert resolved is not None

            #     if f_match == Cmd.ADD:
            #         j = resolved+1
            #     elif f_match == Cmd.REMOVE:
            #         i = resolved+1
            #     else:
            #         raise Exception()

            #     (f_match, f_arg) = trace_f[i][j]

            if f_match == Cmd.MATCH:
                assert isinstance(f_arg, list)
                for (right, left) in reversed(f_arg):
                    # @HACK We set it to -2 here because we add 1 in the final
                    # line. A little messy.
                    na = a_adj[i-1][left] if left >= 0 else -2
                    nb = b_adj[j-1][right] if right >= 0 else -2
                    to_compute.append((na+1, nb+1))
            else:
                raise NotImplementedError()
        elif t_match == Cmd.ADD:
            # Add means we inject a node right here, but continue mapping the
            # current node from a into one of the children in b
            alignment.append((-1, j-1))
            for t,  t_node in enumerate(b_adj[j-1]):
                if t == t_arg:
                    to_compute.append((i, t_node+1))
                else:
                    to_compute.append((-1, t_node+1))
        else:
            raise NotImplementedError()

    return alignment
