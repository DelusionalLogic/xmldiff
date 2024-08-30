from enum import (
    Enum,
    auto,
)

import numpy as np

a = [0, 1, 2, 3]
a_adj = [[1, 3], [2], [], []]

b = [0, 5, 3, 2, 4]
b_adj = [[1, 2], [], [3, 4], [], []]

cost_n = np.empty((len(a)+1, len(b)+1), dtype=int)
cost_f = np.empty((len(a)+1, len(b)+1), dtype=int)
trace = [[(None, None, None) for _ in range(len(b)+1)] for _ in range(len(a)+1)]
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

cost_e = np.empty((len(a)+1, len(b)+1), dtype=int)
for i in reversed(range(len(a))):
    for j in reversed(range(len(b))):
        cost_e[0][0] = 0
        for s in range(len(a_adj[i])):
            cost_e[s+1][0] = cost_e[s][0] + cost_n[a_adj[i][s]+1][0]
        for t in range(len(b_adj[j])):
            cost_e[0][t+1] = cost_e[0][t] + cost_n[0][b_adj[j][t]+1]
            if (i == 0 and j == 0) or 1:
                print(f"Writing {cost_e[0][t+1]} to cost_e 0, {t+1} From:")
                print(cost_e[0][t] + cost_n[0][b_adj[j][t]+1])
        for s in range(len(a_adj[i])):
            for t in range(len(b_adj[j])):
                cost_e[s+1][t+1] = min(
                    cost_e[s+1][t] + cost_n[0][b_adj[j][t]+1],
                    cost_e[s][t+1] + cost_n[a_adj[i][s]+1][0],
                    cost_e[s][t] + cost_n[a_adj[i][s]+1][b_adj[j][t]+1],
                )
                if (i == 0 and j == 0) or 1:
                    print(f"Writing {cost_e[s+1][t+1]} to cost_e {s+1}, {t+1} From:")
                    print(cost_e[s+1][t] + cost_n[0][b_adj[j][t]+1])
                    print(cost_e[s][t+1] + cost_n[a_adj[i][s]+1][0])
                    print(cost_e[s][t] + cost_n[a_adj[i][s]+1][b_adj[j][t]+1])

        if (i == 0 and j == 0) or 1:
            print(a_adj[i], b_adj[j])
            print(cost_e)
            print(a[i], b[j], cost_e[len(a_adj[i])][len(b_adj[j])])

        # print(a_adj[i], b_adj[j], min_to, min_from)

        choices = [
            cost_e[len(a_adj[i])][len(b_adj[j])]
        ]
        f_traces = [
            (Cmd.MATCH, None)
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
            f_traces.append((Cmd.ADD, fmin_s))

        f_min = np.argmin(choices)
        cost_f[i+1][j+1] = choices[f_min]
        forest_trace = f_traces[f_min]

        if (i == 0 and j == 0) or 1:
            print(f"Writing {cost_f[i+1][j+1]} to cost_f {i+1}, {j+1} From:")
            print(choices)

            print(cost_f[i+1][0])

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

def cost(a, b):
    if a is None:
        return 1
    if b is None:
        return 1

    if a == b:
        return 0

    return 1


print(trace)
