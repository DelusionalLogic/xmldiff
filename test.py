import unittest

import numpy as np

from constrained import (
    constrained_edit_distance,
    cost,
)


class TestConstrainedEditDistance(unittest.TestCase):
    def test_empty_trees(self):
        a_adj = []
        b_adj = []
        distance, _ = constrained_edit_distance(a_adj, b_adj, cost)
        self.assertEqual(distance, 0)

    def test_identical_trees(self):
        a_adj = [[1, 2], [], []]
        b_adj = [[1, 2], [], []]
        distance, _ = constrained_edit_distance(a_adj, b_adj, cost, [list(range(len(a_adj)))]*2)
        self.assertEqual(distance, 0)

    def test_different_trees(self):
        a_adj = [[1, 2], [], []]
        b_adj = [[1], [2], []]
        distance, _ = constrained_edit_distance(a_adj, b_adj, cost, [list(range(len(a_adj)))]*2)
        self.assertEqual(distance, 2)

    def test_add_node(self):
        a_adj = [[1], []]
        b_adj = [[1, 2], [], []]
        distance, _ = constrained_edit_distance(a_adj, b_adj, cost, [list(range(len(b_adj)))]*2)
        self.assertEqual(distance, 1)

    def test_remove_node(self):
        a_adj = [[1, 2], [], []]
        b_adj = [[1], []]
        distance, _ = constrained_edit_distance(a_adj, b_adj, cost, [list(range(len(a_adj)))]*2)
        self.assertEqual(distance, 1)

    def test_single_node_trees(self):
        a_adj = [[]]
        b_adj = [[]]
        distance, _ = constrained_edit_distance(a_adj, b_adj, cost, [[0], [0]])
        self.assertEqual(distance, 0)

    def test_multi_level_trees(self):
        a_adj = [[1, 4], [2, 3], [], [], []]
        b_adj = [[1, 3], [2], [], [4], []]
        distance, _ = constrained_edit_distance(a_adj, b_adj, cost, [list(range(len(a_adj)))]*2)
        self.assertEqual(distance, 2)

    def test_different_node_values(self):
        a_adj = [[1, 2], [], []]
        b_adj = [[1, 2], [], []]
        a_values = [0, 1, 2]
        b_values = [0, 2, 1]
        distance, _ = constrained_edit_distance(a_adj, b_adj, cost, [a_values, b_values])
        self.assertEqual(distance, 2)

if __name__ == '__main__':
    unittest.main()
