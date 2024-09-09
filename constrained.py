from enum import (
    Enum,
    auto,
)

import edist.alignment
import edist.sed
import edist.tree_edits
import numpy as np

a = [0, 1, 2, 3]
a_adj = [[1, 3], [2], [], []]

b = [0, 1, 3, 2, 4]
b_adj = [[1, 2], [], [3, 4], [], []]

def cost(ai, bi, data):
    if ai is None:
        return 1
    if bi is None:
        return 1

    (a, b) = data
    if a[ai] == b[bi]:
        return 0

    return 1

class Cmd(Enum):
    MATCH = 1
    REMOVE = 2
    ADD = 3

def constrained_edit_distance(a_adj, b_adj, cost, data=None):
    if len(a_adj) == 0 and len(b_adj) == 0:
        return 0, None

    cost_n = np.empty((len(a_adj)+1, len(b_adj)+1), dtype=int)
    cost_f = np.empty((len(a_adj)+1, len(b_adj)+1), dtype=int)
    trace = [[None for _ in range(len(b_adj)+1)] for _ in range(len(a_adj)+1)]

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

    def seq_dist(ai, bi):
        if ai is None:
            return cost_n[0][bi+1]
        if bi is None:
            return cost_n[ai+1][0]

        return cost_n[ai+1][bi+1]

    for i in reversed(range(len(a_adj))):
        for j in reversed(range(len(b_adj))):
            choices = [
                edist.sed.sed(a_adj[i], b_adj[j], delta=seq_dist)
            ]
            alignment = edist.sed.sed_backtrace(a_adj[i], b_adj[j], delta=seq_dist)
            f_traces = [
                (Cmd.MATCH, alignment)
            ]

            fmin_s = None
            if len(a_adj[i]) > 0:
                fmin_s = np.argmin((cost_f[a_adj[i][s]+1][j+1] - cost_f[a_adj[i][s]+1][0] for s in range(len(a_adj[i]))))
                choices.append(cost_f[i+1][0] + cost_f[a_adj[i][fmin_s]+1][j+1] - cost_f[a_adj[i][fmin_s]+1][0])
                f_traces.append((Cmd.REMOVE, fmin_s))

            fmin_t = None
            if len(b_adj[j]) > 0:
                fmin_t = np.argmin((cost_f[i+1][b_adj[j][t]+1] - cost_f[0][b_adj[j][t]+1] for t in range(len(b_adj[j]))))
                choices.append(cost_f[0][j+1] + cost_f[i+1][b_adj[j][fmin_t]+1] - cost_f[0][b_adj[j][fmin_t]+1])
                f_traces.append((Cmd.ADD, fmin_t))

            f_min = np.argmin(choices)
            cost_f[i+1][j+1] = choices[f_min]
            forest_trace = f_traces[f_min]


            choices = [
                cost_f[i+1][j+1] + cost(i, j, data)
            ]
            n_traces = [
                (Cmd.MATCH, None)
            ]

            n_min_s = None
            if len(a_adj[i]) > 0:
                n_min_s = np.argmin((cost_n[a_adj[i][s]+1][j+1] - cost_n[a_adj[i][s]+1][0] for s in range(len(a_adj[i]))))
                choices.append(cost_n[i+1][0] + cost_n[a_adj[i][n_min_s]+1][j+1] - cost_n[a_adj[i][n_min_s]+1][0])
                n_traces.append((Cmd.REMOVE, n_min_s))

            n_min_t = None
            if len(b_adj[j]) > 0:
                n_min_t = np.argmin((cost_n[i+1][b_adj[j][t]+1] - cost_n[0][b_adj[j][t]+1] for t in range(len(b_adj[j]))))
                choices.append(cost_n[0][j+1] + cost_n[i+1][b_adj[j][n_min_t]+1] - cost_n[0][b_adj[j][n_min_t]+1])
                n_traces.append((Cmd.ADD, n_min_t))

            n_min = np.argmin(choices)
            cost_n[i+1][j+1] = choices[n_min]
            trace[i+1][j+1] = (forest_trace, n_traces[n_min])

    return (cost_n[1][1].item(), trace)

def constrained_alignment(a_adj, b_adj, trace):
    if len(a_adj) == 0 and len(b_adj) == 0:
        return edist.alignment.Alignment()

    alignment = edist.alignment.Alignment()
    to_compute = [
        (1, 1)
    ]

    def do_subtree(alignment, cursor, tree, op):
        remain = 1
        while remain > 0:
            if op == Cmd.REMOVE:
                alignment.append_tuple(cursor, -1)
            else:
                alignment.append_tuple(-1, cursor)
            remain += len(tree[cursor])
            cursor += 1
            remain -= 1

    while len(to_compute) > 0:
        i, j = to_compute.pop()
        (t_match, t_arg) = trace[i][j][1]

        if t_match == Cmd.MATCH:
            alignment.append_tuple(i-1, j-1)
            (f_match, f_arg) = trace[i][j][0]

            if f_match == Cmd.MATCH:
                for e in reversed(f_arg):
                    if e._right >= 0 and e._left >= 0:
                        # Requirese us to recurse
                        na = a_adj[i-1][e._left]
                        nb = b_adj[j-1][e._right]
                        to_compute.append((na+1, nb+1))
                        continue

                    start = None
                    tree = None
                    op = None
                    # These we can handle inline
                    if e._right < 0:
                        start = a_adj[i-1][e._left]
                        tree = a_adj
                        op = Cmd.REMOVE
                    elif e._left < 0:
                        start = b_adj[j-1][e._right]
                        tree = b_adj
                        op = Cmd.ADD
                    do_subtree(alignment, start, tree, op)
            else:
                assert(False)
        elif t_match == Cmd.ADD:
            # Add means we inject a node right here, but continue mapping the
            # current node from a into one of the children in b
            alignment.append_tuple(-1, j-1)
            for t,  t_node in enumerate(b_adj[j-1]):
                if t == t_arg:
                    to_compute.append((i, t_node+1))
                else:
                    do_subtree(alignment, t_node, b_adj, Cmd.ADD)
        else:
            assert(False)

    return alignment

# script = edist.tree_edits.alignment_to_script(alignment, a, a_adj, b, b_adj)
# print(script)
# print(script.apply(a, a_adj))
