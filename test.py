import unittest

from constrained import (
    constrained_alignment,
    constrained_edit_distance,
    cost,
)


class TestConstrainedEditDistance(unittest.TestCase):
    def test_empty_trees(self):
        a_adj = []
        b_adj = []
        distance, trace = constrained_edit_distance(a_adj, b_adj, cost)
        self.assertEqual(distance, 0)
        self.assertIsNone(trace)

    def test_single_node_trees(self):
        a_adj = [[]]
        b_adj = [[]]
        a_values = [0]
        b_values = [0]
        distance, trace = constrained_edit_distance(a_adj, b_adj, cost, (a_values, b_values))
        self.assertEqual(distance, 0)

        assert trace is not None
        alignment = constrained_alignment(a_adj, b_adj, trace)
        self.assertEqual(alignment, [(0, 0)])

    def test_identical_trees(self):
        a_adj = [[1, 2], [], []]
        b_adj = [[1, 2], [], []]
        a_values = [0, 1, 2]
        b_values = [0, 1, 2]
        distance, trace = constrained_edit_distance(a_adj, b_adj, cost, (a_values, b_values))
        self.assertEqual(distance, 0)

        assert trace is not None
        alignment = constrained_alignment(a_adj, b_adj, trace)
        self.assertEqual(alignment, [(0, 0), (1, 1), (2, 2)])

    def test_different_trees(self):
        a_adj = [[1, 2], [], []]
        b_adj = [[1], [2], []]
        a_values = [0, 1, 2]
        b_values = [0, 1, 2]
        distance, trace = constrained_edit_distance(a_adj, b_adj, cost, (a_values, b_values))
        self.assertEqual(distance, 2)

        assert trace is not None
        alignment = constrained_alignment(a_adj, b_adj, trace)
        self.assertEqual(alignment, [(0, 0), (2, -1), (1, 1), (-1, 2)])

    def test_add_node(self):
        a_adj = [[1], []]
        b_adj = [[1, 2], [], []]
        a_values = [0, 1]
        b_values = [0, 1, 2]
        distance, trace = constrained_edit_distance(a_adj, b_adj, cost, (a_values, b_values))
        self.assertEqual(distance, 1)

        assert trace is not None
        alignment = constrained_alignment(a_adj, b_adj, trace)
        self.assertEqual(alignment, [(0, 0), (-1, 2), (1, 1)])

    def test_remove_node(self):
        a_adj = [[1, 2], [], []]
        b_adj = [[1], []]
        a_values = [0, 1, 2]
        b_values = [0, 1]
        distance, trace = constrained_edit_distance(a_adj, b_adj, cost, (a_values, b_values))
        self.assertEqual(distance, 1)

        assert trace is not None
        alignment = constrained_alignment(a_adj, b_adj, trace)
        self.assertEqual(alignment, [(0, 0), (2, -1), (1, 1)])

    def test_multi_level_trees(self):
        a_adj = [[1, 4], [2, 3], [], [], []]
        b_adj = [[1, 3], [2], [], [4], []]
        a_values = [0, 1, 2, 3, 4]
        b_values = [0, 1, 2, 3, 4]
        distance, trace = constrained_edit_distance(a_adj, b_adj, cost, (a_values, b_values))
        self.assertEqual(distance, 2)

        assert trace is not None
        alignment = constrained_alignment(a_adj, b_adj, trace)
        self.assertEqual(alignment, [(0, 0), (1, 1), (3, -1), (2, 2), (-1, 3), (4, 4)])

    def test_different_node_values(self):
        a_adj = [[1, 2], [], []]
        b_adj = [[1, 2], [], []]
        a_values = [0, 1, 2]
        b_values = [0, 2, 1]
        distance, trace = constrained_edit_distance(a_adj, b_adj, cost, (a_values, b_values))
        self.assertEqual(distance, 2)

        assert trace is not None
        alignment = constrained_alignment(a_adj, b_adj, trace)
        self.assertEqual(alignment, [(0, 0), (1, 1), (2, 2)])

if __name__ == '__main__':
    unittest.main()
