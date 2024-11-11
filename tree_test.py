import unittest
from io import (
    BytesIO,
)

import tree


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

        self.assertEqual(
            out.getvalue(),
            b'<root>{+<object><attr/></object>+}</root>'
        )

    def test_change_subtree_tag(self):
        ta = BytesIO(b"<root><object><attr/></object></root>")
        tb = BytesIO(b"<root><obj><attr/></obj></root>")
        out = BytesIO()
        tree.merge_trees(ta, tb, out)

        self.assertEqual(
            out.getvalue(),
            b'<root>{-<object>-}{+<obj>+}<attr/>{-</object>-}{+</obj>+}</root>'
        )

    def test_add_parent_tag(self):
        ta = BytesIO(b"<root><attr/></root>")
        tb = BytesIO(b"<root><obj><attr/></obj></root>")
        out = BytesIO()
        tree.merge_trees(ta, tb, out)

        self.assertEqual(
            out.getvalue(),
            b'<root>{+<obj>+}<attr/>{+</obj>+}</root>'
        )

    def test_add_whitespace(self):
        ta = BytesIO(b"<root></root>")
        tb = BytesIO(b"<root> </root>")
        out = BytesIO()
        tree.merge_trees(ta, tb, out)

        # I don't think this is actually what I want
        self.assertEqual(
            out.getvalue(),
            b"{-<root>-}{+<root> +}</root>"
        )

    def test_match_subtree_parent_property_changed(self):
        ta = BytesIO(b"<root><parent type=\"a\"><child/></parent></root>")
        tb = BytesIO(b"<root><parent type=\"b\"><child/></parent></root>")
        out = BytesIO()
        tree.merge_trees(ta, tb, out)

        self.assertEqual(
            out.getvalue(),
            b"<root>{-<parent type=\"a\">-}{+<parent type=\"b\">+}<child/></parent></root>"
        )
