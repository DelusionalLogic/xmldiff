from io import BytesIO
import tree
import unittest

class TestTreeMerge(unittest.TestCase):
    def test_reorder_sibling_one_is_subtree(self):
        ta = BytesIO(b"<root><attr/><object><attr/></object></root>")
        tb = BytesIO(b"<root><object><attr/></object><attr/></root>")
        out = BytesIO()
        tree.merge_trees(ta, tb, out)

        self.assertEqual(
            out.getvalue(),
            b"<root>{-<attr/>-}<object><attr/></object>{+<attr/>+}</root>"
        )

    def test_reorder_siblings(self):
        ta = BytesIO(b"<root><attr12345/><attr/></root>")
        tb = BytesIO(b"<root><attr/><attr12345/></root>")
        out = BytesIO()
        tree.merge_trees(ta, tb, out)

        self.assertEqual(
            out.getvalue(),
            b'<root>{+<attr/>+}<attr12345/>{-<attr/>-}</root>'
        )

    def test_add_subtree(self):
        ta = BytesIO(b"<root></root>")
        tb = BytesIO(b"<root><object><attr/></object></root>")
        out = BytesIO()
        tree.merge_trees(ta, tb, out)

        print(out.getvalue())
        self.assertEqual(
            out.getvalue(),
            b'<root>{+<object><attr/></object>+}</root>'
        )

    def test_change_subtree_tag(self):
        ta = BytesIO(b"<root><object><attr/></object></root>")
        tb = BytesIO(b"<root><obj><attr/></obj></root>")
        out = BytesIO()
        tree.merge_trees(ta, tb, out)

        print(out.getvalue())
        self.assertEqual(
            out.getvalue(),
            b'<root>{-<object>-}{+<obj>+}<attr/>{-</object>-}{+</obj>+}</root>'
        )
