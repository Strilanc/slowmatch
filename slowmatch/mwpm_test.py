import cirq
import pytest
from slowmatch.fill_system import Event, RegionHitRegionEvent
from slowmatch.fill_system_test import CommandRecordingFillSystem
from slowmatch.mwpm import MinWeightMatchingState, OuterNode, InnerNode, cycle_split
from typing import List, Optional, Iterable, Union, Dict


def alternating_tree_builder():
    counter = 0

    def make_tree(*children: InnerNode,
                  inner_id: Optional[int] = None,
                  outer_id: Optional[int] = None,
                  root: bool = False) -> Union[OuterNode, InnerNode]:
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


def alternating_tree_map(tree: OuterNode, out: Optional[Dict[int, Union[OuterNode, InnerNode]]] = None) -> Dict[int, Union[OuterNode, InnerNode]]:
    if out is None:
        out = {}
    out[tree.region_id] = tree
    for child in tree.children:
        out[child.region_id] = child
        alternating_tree_map(child.child, out)
    return out


def test_tree_str():
    t = alternating_tree_builder()
    tree = t(
        t(
            t(),
            t(),
            t(),
        ),
        t(),
        inner_id=200,
        outer_id=201,
    )
    assert str(tree) == """
200 ===> 201
+---6 ===> 7
|   +---0 ===> 1
|   +---2 ===> 3
|   +---4 ===> 5
+---8 ===> 9
""".strip()

    assert str(tree.child) == """
201
+---6 ===> 7
|   +---0 ===> 1
|   +---2 ===> 3
|   +---4 ===> 5
+---8 ===> 9
""".strip()

    assert str(tree.child.children[0]) == """
6 ===> 7
+---0 ===> 1
+---2 ===> 3
+---4 ===> 5
""".strip()


def test_tree_equality():
    eq = cirq.testing.EqualsTester()

    t = alternating_tree_builder()
    tree1 = t(
        t(
            t(),
            t(),
            t(),
        ),
        t(),
    )

    t = alternating_tree_builder()
    tree2 = t(
        t(
            t(),
            t(),
            t(),
        ),
        t(),
    )

    t = alternating_tree_builder()
    tree3 = t(
        t(
            t(),
            t(inner_id=100),
            t(),
        ),
        t(),
    )

    eq.add_equality_group(tree1, tree2)
    eq.add_equality_group(tree1.child, tree2.child)
    eq.add_equality_group(tree1.child.children[0], tree2.child.children[0])
    eq.add_equality_group(tree1.child.children[1], tree2.child.children[1])
    eq.add_equality_group(tree3)


def test_most_recent_common_ancestor():
    t = alternating_tree_builder()
    tree = t(
        t(
            t(),
            t(),
            t(
                t(),
            )
        ),
        t(),
        root=True
    )
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
            t(
                t(inner_id=6, outer_id=7),
                inner_id=12,
                outer_id=13,
            ),
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
            t(
                t(inner_id=2, outer_id=3),
                inner_id=4,
                outer_id=1,
            ),
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
            t(
                t(inner_id=6, outer_id=7),
                inner_id=12,
                outer_id=13,
            ),
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
            t(
                t(inner_id=6, outer_id=7),
                inner_id=12,
                outer_id=13,
            ),
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
    assert tree == t(
        t(inner_id=2, outer_id=3),
        outer_id=1,
        root=True,
    )
    assert result.orphans == [
        c02.child.children[0],
        c0.child.children[0],
        c0.child.children[1],
    ]
    assert result.pruned_path_regions == [
        13, 12, 5, 4,
    ]


def test_mwpm():
    t = alternating_tree_builder()
    fill = CommandRecordingFillSystem()
    state = MinWeightMatchingState(fill_system=fill)
    for i in range(4):
        fill.create_region(i)
        state.add_region(i)
    fill.recorded_commands.clear()

    # 0 and 1 meet, forming a match.
    state.process_event(RegionHitRegionEvent(region1=0, region2=1, time=1))
    assert state.blossom_map == {}
    assert state.match_map == {0: 1, 1: 0}
    assert state.tree_id_map == {
        2: OuterNode(region_id=2),
        3: OuterNode(region_id=3),
    }
    assert fill.recorded_commands == [
        ('set_region_growth', 0, 0),
        ('set_region_growth', 1, 0)
    ]
    fill.recorded_commands.clear()

    # 2 meets 1, extending into a tree.
    state.process_event(RegionHitRegionEvent(region1=1, region2=2, time=2))
    assert state.blossom_map == {}
    assert state.match_map == {}
    assert state.tree_id_map == {
        **alternating_tree_map(t(t(inner_id=1, outer_id=0), outer_id=2, root=True)),
        3: OuterNode(region_id=3),
    }
    assert fill.recorded_commands == [
        ('set_region_growth', 1, -1),
        ('set_region_growth', 0, +1)
    ]
    fill.recorded_commands.clear()

    # 2 meets 0, forming a blossom.
    state.process_event(RegionHitRegionEvent(region1=0, region2=2, time=3))
    assert state.blossom_map == {4: [0, 1, 2]}
    assert state.match_map == {}
    assert state.tree_id_map == {
        4: OuterNode(region_id=4),
        3: OuterNode(region_id=3),
    }
    assert fill.recorded_commands == [
        ('create_combined_region', (0, 1, 2), 4),
    ]
    fill.recorded_commands.clear()

    # 3 meets 4 (blossom), forming a match.
    state.process_event(RegionHitRegionEvent(region1=3, region2=4, time=5))
    assert state.blossom_map == {4: [0, 1, 2]}
    assert state.match_map == {3: 4, 4: 3}
    assert state.tree_id_map == {}
    assert fill.recorded_commands == [
        ('set_region_growth', 3, 0),
        ('set_region_growth', 4, 0),
    ]
    fill.recorded_commands.clear()


def test_cycle_split():
    def f(i, j):
        return cycle_split(range(7), i, j)

    assert cycle_split(['a', 'b', 'c'], 0, 0) == (['a'], ['a', 'b', 'c', 'a'])
    assert cycle_split(['a', 'b', 'c'], 0, 1) == (['a', 'b'], ['b', 'c', 'a'])

    assert f(0, 0) == ([0], [0, 1, 2, 3, 4, 5, 6, 0])
    assert f(1, 1) == ([1], [1, 2, 3, 4, 5, 6, 0, 1])
    assert f(2, 2) == ([2], [2, 3, 4, 5, 6, 0, 1, 2])

    assert f(0, 1) == ([0, 1], [1, 2, 3, 4, 5, 6, 0])
    assert f(1, 2) == ([1, 2], [2, 3, 4, 5, 6, 0, 1])
    assert f(2, 1) == ([2, 3, 4, 5, 6, 0, 1], [1, 2])

    assert f(1, 3) == ([1, 2, 3], [3, 4, 5, 6, 0, 1])
    assert f(1, 4) == ([1, 2, 3, 4], [4, 5, 6, 0, 1])

    assert f(0, 6) == ([0, 1, 2, 3, 4, 5, 6], [6, 0])
    assert f(6, 0) == ([6, 0], [0, 1, 2, 3, 4, 5, 6])
