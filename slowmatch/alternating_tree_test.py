from typing import Optional, Union, Dict, List

import cirq
import pytest

from slowmatch.alternating_tree import AltTreeNode, AltTreeEdge
from slowmatch.compressed_edge import CompressedEdge
from slowmatch.graph import DetectorNode
from slowmatch.graph_fill_region import GraphFillRegion
from slowmatch.region_path import RegionPath, RegionEdge


def alternating_tree_builder():
    counter = 0

    def make_tree(
            *children: AltTreeNode,
            inner_id: Optional[int] = None,
            outer_id: Optional[int] = None,
            root: bool = False
    ) -> AltTreeNode:
        nonlocal counter
        if inner_id is None:
            inner_id = counter
            counter += 1
        if outer_id is None:
            outer_id = counter
            counter += 1

        if root:
            node = AltTreeNode(inner_region=None, outer_region=GraphFillRegion(id=outer_id))
        else:
            node = AltTreeNode(
                inner_region=GraphFillRegion(id=inner_id),
                outer_region=GraphFillRegion(id=outer_id),
                inner_outer_edge=CompressedEdge(
                    loc_from=DetectorNode(loc=inner_id),
                    loc_to=DetectorNode(loc=outer_id),
                    obs_mask=0,
                    distance=1
                )
            )
        for child in children:
            node.add_child(
                child=AltTreeEdge(
                    node=child,
                    edge=CompressedEdge(
                        loc_from=DetectorNode(loc=node.outer_region.id),
                        loc_to=DetectorNode(loc=child.inner_region.id),
                        obs_mask=0,
                        distance=1
                    )
                )
            )
        return node

    return make_tree


def test_alt_tree_node_str():
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

    tree.inner_region = None

    assert (
        str(tree)
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
        str(tree.children[0].node)
        == """
6 ===> 7
+---0 ===> 1
+---2 ===> 3
+---4 ===> 5
""".strip()
    )


def test_find_root():
    t = alternating_tree_builder()
    tree = t(t(t(), t(), t(), ), t(), inner_id=200, outer_id=201, root=True)

    assert tree.find_root() is tree
    assert tree.children[0].node.children[0].node.find_root() is tree
    assert tree.children[0].node.children[1].node.find_root() is tree
    assert tree.children[1].node.find_root() is tree


def test_alt_tree_node_equality():
    eq = cirq.testing.EqualsTester()

    t = alternating_tree_builder()
    tree1 = t(t(t(), t(), t(),), t(), root=True)

    t = alternating_tree_builder()
    tree2 = t(t(t(), t(), t(),), t(), root=True)

    t = alternating_tree_builder()
    tree3 = t(t(t(), t(inner_id=100), t(),), t(), root=True)

    eq.add_equality_group(tree1, tree2)
    eq.add_equality_group(tree1.children[0].node, tree2.children[0].node)
    eq.add_equality_group(tree1.children[1].node, tree2.children[1].node)
    eq.add_equality_group(tree3)


def test_alt_tree_most_recent_common_ancestor():
    t = alternating_tree_builder()
    tree = t(t(t(), t(), t(t(), )), t(), root=True)
    c0 = tree.children[0].node
    c1 = tree.children[1].node
    c00 = c0.children[0].node
    c01 = c0.children[1].node
    assert tree.most_recent_common_ancestor(tree) is tree
    assert tree.most_recent_common_ancestor(tree.children[0].node) is tree
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
    c = tree.children[0].node.children[2].node
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


def test_alt_tree_outer_ancestry():
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
    assert tree.ancestors() == [tree]
    assert tree.ancestors(stop_before=tree) == []
    c0 = tree.children[0].node
    assert c0.ancestors() == [c0, tree]
    assert c0.ancestors(stop_before=tree) == [c0]
    assert c0.ancestors(stop_before=c0) == []
    c00 = c0.children[0].node
    assert c00.ancestors() == [c00, c0, tree]
    assert c00.ancestors(stop_before=tree) == [c00, c0]
    assert c00.ancestors(stop_before=c0) == [c00]
    assert c00.ancestors(stop_before=c00) == []


def gen_blossom_edge_path(region_ids: List[int]) -> RegionPath:
    out_edges = []
    for i in range(len(region_ids) - 1):
        e = RegionEdge(
            region=GraphFillRegion(id=region_ids[i]),
            edge=CompressedEdge(
                loc_from=DetectorNode(loc=region_ids[i]),
                loc_to=DetectorNode(loc=region_ids[i + 1]),
                obs_mask=0,
                distance=1
            )
        )
        out_edges.append(e)
    return RegionPath(out_edges)


def test_alt_tree_prune_upward_path_stopping_before():
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

    c0 = tree.children[0].node
    c02 = c0.children[2].node
    result = c02.prune_upward_path_stopping_before(tree)
    assert tree == t(t(inner_id=2, outer_id=3), outer_id=1, root=True,)
    assert result.orphans == [
        c02.children[0],
        c0.children[0],
        c0.children[1],
    ]

    assert result.pruned_path_regions == gen_blossom_edge_path([13, 12, 5, 4, 1])
