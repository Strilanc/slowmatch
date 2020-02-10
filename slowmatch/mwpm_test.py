from slowmatch.alternating_tree_test import alternating_tree_builder, alternating_tree_map
from slowmatch.flooder import RegionHitRegionEvent, BlossomImplodeEvent
from slowmatch.flooder_test import CommandRecordingFlooder
from slowmatch.mwpm import Mwpm, OuterNode, cycle_split


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


def test_match_then_blossom_then_match():
    t = alternating_tree_builder()
    fill = CommandRecordingFlooder()
    state = Mwpm(flooder=fill)
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


def test_blossom_implosion():
    fill = CommandRecordingFlooder()
    state = Mwpm(flooder=fill)
    n = 10
    for i in range(n + 3):
        fill.create_region(i)
        state.add_region(i)
    fill.recorded_commands.clear()

    # Pair up.
    for i in range(0, n, 2):
        state.process_event(
            RegionHitRegionEvent(region1=i, region2=i + 1, time=0))
    # Form an alternating path through the pairs.
    for i in range(1, n, 2)[::-1]:
        state.process_event(
            RegionHitRegionEvent(region1=i, region2=i + 1, time=0))
    # Close the path into a blossom.
    state.process_event(
        RegionHitRegionEvent(region1=0, region2=n, time=0))
    blossom_id = n + 3

    # Make the blossom become an inner node.
    state.process_event(
        RegionHitRegionEvent(region1=n + 1, region2=blossom_id, time=0))
    state.process_event(
        RegionHitRegionEvent(region1=n + 2, region2=blossom_id, time=0))

    # Implode the blossom.
    state.process_event(
        BlossomImplodeEvent(time=0,
                            blossom_region_id=blossom_id,
                            in_out_touch_pairs=[
                                (5, n + 1),
                                (9, n + 2),
                            ]))

    assert fill.recorded_commands == [
        # Initial pairing up.
        ('set_region_growth', 0, 0),
        ('set_region_growth', 1, 0),
        ('set_region_growth', 2, 0),
        ('set_region_growth', 3, 0),
        ('set_region_growth', 4, 0),
        ('set_region_growth', 5, 0),
        ('set_region_growth', 6, 0),
        ('set_region_growth', 7, 0),
        ('set_region_growth', 8, 0),
        ('set_region_growth', 9, 0),
        # Formation of alternating path.
        ('set_region_growth', 9, -1),
        ('set_region_growth', 8, 1),
        ('set_region_growth', 7, -1),
        ('set_region_growth', 6, 1),
        ('set_region_growth', 5, -1),
        ('set_region_growth', 4, 1),
        ('set_region_growth', 3, -1),
        ('set_region_growth', 2, 1),
        ('set_region_growth', 1, -1),
        ('set_region_growth', 0, 1),
        # Formation of the blossom.
        ('create_combined_region', (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), 13),
        # Blossom turning into an inner node.
        ('set_region_growth', 11, 0),
        ('set_region_growth', 13, 0),
        ('set_region_growth', 13, -1),
        ('set_region_growth', 11, +1),
        # Blossom imploding.
        ('set_region_growth', 9, -1),
        ('set_region_growth', 8, +1),
        ('set_region_growth', 7, -1),
        ('set_region_growth', 6, +1),
        ('set_region_growth', 5, -1),
    ]

    t = alternating_tree_builder()
    expected_tree = t(
        root=True,
        outer_id=n + 2,
        child=t(
            inner_id=9,
            outer_id=8,
            child=t(
                inner_id=7,
                outer_id=6,
                child=t(
                    inner_id=5,
                    outer_id=n + 1,
                )
            )
        ),
    )
    assert state.tree_id_map == {**alternating_tree_map(expected_tree)}
