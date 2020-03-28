from typing import Optional, Union, Dict

import cirq
import pytest

from slowmatch.alternating_tree import OuterNode, InnerNode


def alternating_tree_builder():
    """Returns a handy method for creating alternating trees."""
    counter = 0

    def make_tree(
        *children: InnerNode,
        inner_id: Optional[int] = None,
        outer_id: Optional[int] = None,
        child: Optional[InnerNode] = None,
        root: bool = False,
    ) -> Union[OuterNode, InnerNode]:
        if child is not None:
            children = children + (child,)

        nonlocal counter
        if inner_id is None:
            inner_id = counter
            counter += 1
        if outer_id is None:
            outer_id = counter
            counter += 1

        outer = OuterNode(outer_id)
        if not root:
            outer.parent = InnerNode(inner_id, parent=None, child=outer)
        outer.children = list(children)
        for child in children:
            child.parent = outer
        return outer if root else outer.parent

    return make_tree


def alternating_tree_map(
    tree: OuterNode, *, out: Optional[Dict[int, Union[OuterNode, InnerNode]]] = None,
) -> Dict[int, Union[OuterNode, InnerNode]]:
    """Returns a dictionary mapping node region ids to nodes from the tree."""
    if out is None:
        out = {}
    out[tree.region_id] = tree
    for child in tree.children:
        out[child.region_id] = child
        alternating_tree_map(child.child, out=out)
    return out


def test_tree_str():
    t = alternating_tree_builder()
    tree = t(t(t(), t(), t(),), t(), inner_id=200, outer_id=201,)
    assert (
        str(tree)
        == """
200 ===> 201
+---6 ===> 7
|   +---0 ===> 1
|   +---2 ===> 3
|   +---4 ===> 5
+---8 ===> 9
""".strip()
    )

    assert (
        str(tree.child)
        == """
201
+---6 ===> 7
|   +---0 ===> 1
|   +---2 ===> 3
|   +---4 ===> 5
+---8 ===> 9
""".strip()
    )

    assert (
        str(tree.child.children[0])
        == """
6 ===> 7
+---0 ===> 1
+---2 ===> 3
+---4 ===> 5
""".strip()
    )


def test_tree_equality():
    eq = cirq.testing.EqualsTester()

    t = alternating_tree_builder()
    tree1 = t(t(t(), t(), t(),), t(),)

    t = alternating_tree_builder()
    tree2 = t(t(t(), t(), t(),), t(),)

    t = alternating_tree_builder()
    tree3 = t(t(t(), t(inner_id=100), t(),), t(),)

    eq.add_equality_group(tree1, tree2)
    eq.add_equality_group(tree1.child, tree2.child)
    eq.add_equality_group(tree1.child.children[0], tree2.child.children[0])
    eq.add_equality_group(tree1.child.children[1], tree2.child.children[1])
    eq.add_equality_group(tree3)


def test_most_recent_common_ancestor():
    t = alternating_tree_builder()
    tree = t(t(t(), t(), t(t(),)), t(), root=True)
    c0 = tree.children[0].child
    c1 = tree.children[1].child
    c00 = c0.children[0].child
    c01 = c0.children[1].child
    assert tree.most_recent_common_ancestor(tree) is tree
    assert tree.most_recent_common_ancestor(tree.children[0].child) is tree
    assert c0.most_recent_common_ancestor(c1) is tree
    assert c00.most_recent_common_ancestor(c1) is tree
    assert c00.most_recent_common_ancestor(tree) is tree
    assert c00.most_recent_common_ancestor(c0) is c0
    assert c00.most_recent_common_ancestor(c01) is c0
    assert c01.most_recent_common_ancestor(c00) is c0
    with pytest.raises(ValueError, match='No common ancestor'):
        _ = c00.most_recent_common_ancestor(t(root=True))


def test_become_root():
    t = alternating_tree_builder()
    tree = t(
        t(
            t(inner_id=10, outer_id=11),
            t(inner_id=8, outer_id=9),
            t(t(inner_id=6, outer_id=7), inner_id=12, outer_id=13,),
            inner_id=4,
            outer_id=5,
        ),
        t(inner_id=2, outer_id=3),
        outer_id=1,
        root=True,
    )
    c = tree.children[0].child.children[2].child
    c.become_root()
    assert c == t(
        t(inner_id=6, outer_id=7),
        t(
            t(inner_id=10, outer_id=11),
            t(inner_id=8, outer_id=9),
            t(t(inner_id=2, outer_id=3), inner_id=4, outer_id=1,),
            inner_id=12,
            outer_id=5,
        ),
        outer_id=13,
        root=True,
    )


def test_outer_ancestry():
    t = alternating_tree_builder()
    tree = t(
        t(
            t(inner_id=10, outer_id=11),
            t(inner_id=8, outer_id=9),
            t(t(inner_id=6, outer_id=7), inner_id=12, outer_id=13,),
            inner_id=4,
            outer_id=5,
        ),
        t(inner_id=2, outer_id=3),
        outer_id=1,
        root=True,
    )
    assert tree.outer_ancestry() == [tree]
    assert tree.outer_ancestry(stop_before=tree) == []
    c0 = tree.children[0].child
    assert c0.outer_ancestry() == [c0, tree]
    assert c0.outer_ancestry(stop_before=tree) == [c0]
    assert c0.outer_ancestry(stop_before=c0) == []
    c00 = c0.children[0].child
    assert c00.outer_ancestry() == [c00, c0, tree]
    assert c00.outer_ancestry(stop_before=tree) == [c00, c0]
    assert c00.outer_ancestry(stop_before=c0) == [c00]
    assert c00.outer_ancestry(stop_before=c00) == []


def test_prune_upward_path_stopping_before():
    t = alternating_tree_builder()
    tree = t(
        t(
            t(inner_id=10, outer_id=11),
            t(inner_id=8, outer_id=9),
            t(t(inner_id=6, outer_id=7), inner_id=12, outer_id=13,),
            inner_id=4,
            outer_id=5,
        ),
        t(inner_id=2, outer_id=3),
        outer_id=1,
        root=True,
    )

    c0 = tree.children[0]
    c02 = c0.child.children[2]
    result = c02.child.prune_upward_path_stopping_before(tree)
    assert tree == t(t(inner_id=2, outer_id=3), outer_id=1, root=True,)
    assert result.orphans == [
        c02.child.children[0],
        c0.child.children[0],
        c0.child.children[1],
    ]
    assert result.pruned_path_regions == [
        13,
        12,
        5,
        4,
    ]
