from enum import (
    Enum,
    auto,
)

import numpy as np
from edist import (
    sed,
)

a = [0, 1, 2, 3]
a_adj = [[1, 3], [2], [], []]

b = [0, 5, 3, 2, 4]
b_adj = [[1, 2], [], [3, 4], [], []]

cost_n = np.empty((len(a)+1, len(b)+1), dtype=int)
cost_f = np.empty((len(a)+1, len(b)+1), dtype=int)
trace = [[None for _ in range(len(b)+1)] for _ in range(len(a)+1)]
# cost_n = [[None] * (len(b)+1) for _ in range(len(a)+1)]

def cost(ai, bi):
    if ai is None:
        return 1
    if bi is None:
        return 1

    if a[ai] == b[bi]:
        return 0

    return 1

cost_n[0][0] = 0
cost_f[0][0] = 0
for i in reversed(range(len(a))):
    cost_f[i+1][0] = 0
    for c in a_adj[i]:
        cost_f[i+1][0] += cost_n[c+1][0]
    cost_n[i+1][0] = cost_f[i+1][0] + cost(i, None)
for j in reversed(range(len(b))):
    cost_f[0][j+1] = 0
    for c in b_adj[j]:
        cost_f[0][j+1] += cost_n[0][c+1]
    cost_n[0][j+1] = cost_f[0][j+1] + cost(None, j)

class Cmd(Enum):
    MATCH = 1
    REMOVE = 2
    ADD = 3

# Verified so far

def seq_dist(ai, bi):
    if ai is None:
        return cost_n[0][bi]
    if bi is None:
        return cost_n[ai][0]

    return cost_n[ai+1][bi+1]


cost_e = np.empty((len(a)+1, len(b)+1), dtype=int)
for i in reversed(range(len(a))):
    for j in reversed(range(len(b))):
        choices = [
            sed.sed(a_adj[i], b_adj[j], delta=seq_dist)
        ]
        # @PERF @CLEANUP: Right now it doesn't make a lot of sense for us to
        # compute the alignment here, but optimally we could use the matrix we
        # just computed
        alignment = sed.sed_backtrace(a_adj[i], b_adj[j], delta=seq_dist)
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

        if (i == 0 and j == 0) or 1:
            print(f"Writing {cost_f[i+1][j+1]} to cost_f {i+1}, {j+1} From:")
            print(choices)

        choices = [
            cost_f[i+1][j+1] + cost(i, j)
        ]
        n_traces = [
            (Cmd.MATCH, None)
        ]

        n_min_s = None
        if len(a_adj[i]) > 0:
            n_min_s = np.argmin((cost_n[a_adj[i][s]+1][j+1] - cost_n[a_adj[i][s]+1][0] for s in range(len(a_adj[i]))))
            choices.append(cost_n[i+1][0] + cost_n[a_adj[i][n_min_s]+1][j+1] - cost_n[a_adj[i][n_min_s]+1][0])
            n_traces.append((Cmd.REMOVE, fmin_s))

        n_min_t = None
        if len(b_adj[j]) > 0:
            n_min_t = np.argmin((cost_n[i+1][b_adj[j][t]+1] - cost_n[0][b_adj[j][t]+1] for t in range(len(b_adj[j]))))
            choices.append(cost_n[0][j+1] + cost_n[i+1][b_adj[j][n_min_t]+1] - cost_n[0][b_adj[j][n_min_t]+1])
            n_traces.append((Cmd.ADD, fmin_s))

        n_min = np.argmin(choices)
        cost_n[i+1][j+1] = choices[n_min]
        trace[i+1][j+1] = (forest_trace, n_traces[n_min])

        if (i == 0 and j == 0) or 1:
            print(f"Writing {cost_n[i+1][j+1]} to cost_n {i+1}, {j+1} From:")
            print(choices)

print(cost_n[0][1])

print(cost_n)
# print(cost_n)

# def cost(a, b):
#     if a is None:
#         return 1
#     if b is None:
#         return 1

#     if a == b:
#         return 0

#     return 1


print(trace)

to_compute = [
    (1, 1)
]
trace_so_far = []

while to_compute:
    i, j = to_compute.pop()
    print((i, j))
    ((f_match, f_arg), (t_match, t_arg)) = trace[i][j]
    print(f_match, t_match)

    if t_match == Cmd.MATCH:
        trace_so_far.append((i-1, j-1))

        if f_match == Cmd.MATCH:
            for tpl in f_arg:
                print(tpl)
                next = (a_adj[i-1][tpl._left]+1, b_adj[j-1][tpl._right]+1)
                print(f"Scheduling {next}")
                to_compute.append(next)
        elif f_match == Cmd.ADD:
            print(f_arg)
            for c in b_adj[b_adj[j-1][f_arg]]:
                next = (i, b_adj[j-1][f_arg]+1)
                print(f"Scheduling {next}")
                to_compute.append(next)
        elif f_match == Cmd.REMOVE:
            print(f_arg)
            for c in a_adj[a_adj[i-1][f_arg]]:
                next = (a_adj[i-1][f_arg]+1, j)
                print(f"Scheduling {next}")
                to_compute.append(next)

print(trace_so_far)
