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
        self.assertGreater(distance, 0)

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

if __name__ == '__main__':
    unittest.main()
