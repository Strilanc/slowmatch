from typing import Optional, Callable

from slowmatch.events import RegionHitRegionEvent, BlossomImplodeEvent
from slowmatch.graph import LocationData
from slowmatch.graph_fill_region import GraphFillRegion
from slowmatch.graph_flooder_test import RecordingFlooder
from slowmatch.mwpm import Mwpm, TLocation
from slowmatch.alternating_tree import AltTreeNode, AltTreeEdge
from slowmatch.compressed_edge import CompressedEdge
from slowmatch.region_path import RegionEdge


def alternating_tree_builder(region_maker: Callable[[int], GraphFillRegion]):
    def make_tree(
            *children: AltTreeEdge,
            inner_id: Optional[int] = None,
            outer_id: Optional[int] = None,
            root: bool = False,
            source1: Optional[int] = None,
            source2: Optional[int] = None,
            parent_source: Optional[int] = None,
            child_source: Optional[int] = None
    ) -> AltTreeEdge:

        if root:
            node = AltTreeNode(
                inner_region=None,
                outer_region=region_maker(outer_id)
            )
        else:
            node = AltTreeNode(
                inner_region=region_maker(inner_id),
                outer_region=region_maker(outer_id),
                inner_outer_edge=CompressedEdge(
                    loc_from=LocationData(loc=source1),
                    loc_to=LocationData(loc=source2),
                    obs_mask=0,
                    distance=1
                )
            )
        alt_tree_edge = AltTreeEdge(
            node=node,
            edge=CompressedEdge(
                loc_from=LocationData(loc=parent_source),
                loc_to=LocationData(loc=child_source),
                obs_mask=0,
                distance=1
            )
        )
        for child in children:
            node.add_child(
                child
            )
        return alt_tree_edge

    return make_tree


def region_hit_region_builder(fill: RecordingFlooder):

    def get_region(r: int) -> GraphFillRegion:
        return fill.region_map[r]

    def get_region_hit_region(r1: int, r2: int, s1: int, s2: int) -> RegionHitRegionEvent:
        return RegionHitRegionEvent(
            region1=get_region(r1),
            region2=get_region(r2),
            time=1,
            edge=CompressedEdge(
                loc_from=LocationData(loc=s1) if s1 is not None else None,
                loc_to=LocationData(loc=s2) if s2 is not None else None,
                obs_mask=0,
                distance=1
            )
        )

    def get_blossom_cycle_edge(br: int, s1: int, s2: int) -> RegionEdge:
        return RegionEdge(
            region=get_region(br),
            edge=CompressedEdge(
                loc_from=LocationData(s1),
                loc_to=LocationData(s2),
                obs_mask=0,
                distance=1
            )
        )

    return get_region, get_region_hit_region, get_blossom_cycle_edge


def test_match_then_blossom_then_match():
    fill = RecordingFlooder()
    state = Mwpm(flooder=fill)
    for i in range(4):
        r = fill.create_region(i)
        state.add_region(r)
    fill.recorded_commands.clear()

    r, rhr, be = region_hit_region_builder(fill)

    t = alternating_tree_builder(region_maker=r)

    # 0 and 1 meet, forming a match.
    state.process_event(rhr(0, 1, 0, 1))
    assert r(0).match.region is r(1)
    assert r(1).match.region is r(0)
    assert r(0).alt_tree_node is None
    assert r(1).alt_tree_node is None
    assert r(2).alt_tree_node == t(inner_id=None, outer_id=2, root=True).node
    assert r(3).alt_tree_node == t(inner_id=None, outer_id=3, root=True).node
    assert fill.recorded_commands == [('set_region_growth', fill.region_map[0], 0),
                                      ('set_region_growth', fill.region_map[1], 0)]
    fill.recorded_commands.clear()

    # 2 meets 1, extending into a tree.
    state.process_event(rhr(1, 2, 1, 2))
    assert r(2).alt_tree_node == t(
        t(
            inner_id=1,
            outer_id=0,
            source1=1,
            source2=0,
            parent_source=2,
            child_source=1
        ),
        outer_id=2,
        root=True
    ).node

    assert fill.recorded_commands == [('set_region_growth', fill.region_map[1], -1),
                                      ('set_region_growth', fill.region_map[0], +1)]
    fill.recorded_commands.clear()

    # 2 meets 0, forming a blossom.
    state.process_event(rhr(0, 2, 0, 2))
    assert r(4).alt_tree_node.as_tuple() == t(outer_id=4, root=True).node.as_tuple()
    assert r(3).alt_tree_node == t(outer_id=3, root=True).node

    assert fill.recorded_commands == [
        ('create_blossom', (be(0, 0, 1), be(1, 1, 2), be(2, 2, 0)), r(4)),
    ]
    fill.recorded_commands.clear()

    # 3 meets 4 (blossom) via subblossom 0, forming a match.
    state.process_event(rhr(3, 4, 3, 0))
    assert r(3).match.region is r(4)
    assert r(4).match.region is r(3)
    assert fill.recorded_commands == [
        ('set_region_growth', r(3), 0),
        ('set_region_growth', r(4), 0),
    ]
    fill.recorded_commands.clear()


def test_blossom_not_including_root():
    fill = RecordingFlooder()
    state = Mwpm(flooder=fill)
    for i in range(10):
        r = fill.create_region(i)
        state.add_region(r)

    r, rhr, be = region_hit_region_builder(fill)

    # Root at 2, child at 0 then 1.
    state.process_event(rhr(0, 1, 0, 1))
    state.process_event(rhr(0, 2, 0, 2))

    # 1 has child tree.
    state.process_event(rhr(3, 4, 3, 4))
    state.process_event(rhr(5, 6, 5, 6))
    state.process_event(rhr(1, 3, 1, 3))
    state.process_event(rhr(1, 5, 1, 5))

    # Blossom forms within child tree.
    fill.recorded_commands.clear()
    state.process_event(rhr(4, 6, 4, 6))

    assert fill.recorded_commands == [('create_blossom', (be(4, 4, 3), be(3, 3, 1), be(1, 1, 5),
                                                          be(5, 5, 6), be(6, 6, 4)), r(10))]
    t = alternating_tree_builder(region_maker=r)
    tree_found = r(10).alt_tree_node.find_root()
    expected_tree = t(t(inner_id=0, outer_id=10, source1=0, source2=1), inner_id=None, outer_id=2, root=True).node


def test_blossom_implosion():
    fill = RecordingFlooder()
    state = Mwpm(flooder=fill)
    n = 10
    for i in range(n + 3):
        r = fill.create_region(i)
        state.add_region(r)
    fill.recorded_commands.clear()

    r, rhr, be = region_hit_region_builder(fill)

    # Pair up.
    for i in range(0, n, 2):
        state.process_event(rhr(i, i + 1, i, i + 1))
    # Form an alternating path through the pairs.
    for i in range(1, n, 2)[::-1]:
        state.process_event(rhr(i, i + 1, i, i + 1))
    # Close the path into a blossom.
    state.process_event(rhr(0, n, 0, n))
    blossom_id = n + 3

    # Make the blossom become an inner node.
    state.process_event(rhr(n + 1, blossom_id, n + 1, 5))
    state.process_event(rhr(n + 2, blossom_id, n + 2, 9))

    # Implode the blossom.
    state.process_event(
        BlossomImplodeEvent(
            time=0, blossom_region=r(blossom_id),
            in_parent_region=r(9),
            in_child_region=r(5)
        )
    )

    assert fill.recorded_commands == [
        # Initial pairing up.
        ('set_region_growth', r(0), 0),
        ('set_region_growth', r(1), 0),
        ('set_region_growth', r(2), 0),
        ('set_region_growth', r(3), 0),
        ('set_region_growth', r(4), 0),
        ('set_region_growth', r(5), 0),
        ('set_region_growth', r(6), 0),
        ('set_region_growth', r(7), 0),
        ('set_region_growth', r(8), 0),
        ('set_region_growth', r(9), 0),
        # Formation of alternating path.
        ('set_region_growth', r(9), -1),
        ('set_region_growth', r(8), 1),
        ('set_region_growth', r(7), -1),
        ('set_region_growth', r(6), 1),
        ('set_region_growth', r(5), -1),
        ('set_region_growth', r(4), 1),
        ('set_region_growth', r(3), -1),
        ('set_region_growth', r(2), 1),
        ('set_region_growth', r(1), -1),
        ('set_region_growth', r(0), 1),
        # Formation of the blossom.
        ('create_blossom', (be(0, 0, 1), be(1, 1, 2), be(2, 2, 3), be(3, 3, 4), be(4, 4, 5),
                            be(5, 5, 6), be(6, 6, 7), be(7, 7, 8), be(8, 8, 9), be(9, 9, 10),
                            be(10, 10, 0)), r(13)),
        # Blossom turning into an inner node.
        ('set_region_growth', r(11), 0),
        ('set_region_growth', r(13), 0),
        ('set_region_growth', r(13), -1),
        ('set_region_growth', r(11), +1),
        # Blossom imploding.
        # Matches region growth set to zero (to ensure they are rescheduled)
        ('set_region_growth', r(10), 0),
        ('set_region_growth', r(0), 0),
        ('set_region_growth', r(1), 0),
        ('set_region_growth', r(2), 0),
        ('set_region_growth', r(3), 0),
        ('set_region_growth', r(4), 0),
        # Then odd-length path region growth set for outer and inner nodes
        ('set_region_growth', r(9), -1),
        ('set_region_growth', r(8), +1),
        ('set_region_growth', r(7), -1),
        ('set_region_growth', r(6), +1),
        ('set_region_growth', r(5), -1),
    ]

    t = alternating_tree_builder(region_maker=r)
    expected_tree = t(
        t(
            t(
                t(
                    inner_id=5,
                    outer_id=11,
                    source1=5,
                    source2=11
                ),
                inner_id=7,
                outer_id=6,
                source1=7,
                source2=6
            ),
            inner_id=9,
            outer_id=8,
            source1=9,
            source2=8
        ),
        inner_id=None,
        outer_id=12,
        root=True
    )
    assert r(12).alt_tree_node == expected_tree.node
