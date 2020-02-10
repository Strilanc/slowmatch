import random
from typing import List, Tuple, Any
import numpy as np

from slowmatch import Mwpm, Varying
from slowmatch.flooder import RegionHitRegionEvent, BlossomImplodeEvent
from slowmatch.flooder_test import RecordingFlooder
from slowmatch.graph_flooder import GraphFlooder, GraphFillRegion


def line_neighbors(pos: complex) -> List[Tuple[float, complex]]:
    return [
        (1, pos - 1),
        (1, pos + 1),
    ]


def complex_grid_neighbors(pos: complex) -> List[Tuple[float, complex]]:
    return [
        (1, pos - 1j),
        (1, pos + 1j),
        (1, pos - 1),
        (1, pos + 1),
    ]


def test_normal_progression():
    def assert_process_event(*expected_commands: Any):
        event = fill.next_event()
        mwpm.process_event(event)
        assert fill.recorded_commands == list(expected_commands)
        fill.recorded_commands.clear()

    fill = RecordingFlooder(GraphFlooder(line_neighbors))
    mwpm = Mwpm(flooder=fill)
    mwpm.add_region(fill.create_region(100))
    mwpm.add_region(fill.create_region(101))
    mwpm.add_region(fill.create_region(200))
    mwpm.add_region(fill.create_region(202))
    mwpm.add_region(fill.create_region(300))
    mwpm.add_region(fill.create_region(1000))
    fill.recorded_commands.clear()

    # Pair ups.
    assert_process_event(
        ('next_event', RegionHitRegionEvent(region1=0, region2=1, time=0.5)),
        ('set_region_growth', 0, 0),
        ('set_region_growth', 1, 0),
    )
    assert_process_event(
        ('next_event', RegionHitRegionEvent(region1=2, region2=3, time=1)),
        ('set_region_growth', 2, 0),
        ('set_region_growth', 3, 0),
    )
    # Alternating tree starts.
    assert_process_event(
        ('next_event', RegionHitRegionEvent(region1=3, region2=4, time=97)),
        ('set_region_growth', 3, -1),
        ('set_region_growth', 2, +1),
    )
    # Alternating tree turns into a blossom.
    assert_process_event(
        ('next_event', RegionHitRegionEvent(region1=2, region2=4, time=98)),
        ('create_combined_region', (2, 3, 4), 6),
    )
    assert fill.sub_flooder._region_data_map == {
        0: GraphFillRegion(id=0, source=100, radius=0.5),
        1: GraphFillRegion(id=1, source=101, radius=0.5),
        5: GraphFillRegion(id=5, source=1000, radius=Varying.T),
        6: GraphFillRegion(id=6, source=None, radius=Varying(base_time=98, slope=1), blossom_children=[
            GraphFillRegion(id=2, source=200, radius=2),
            GraphFillRegion(id=3, source=202, radius=0),
            GraphFillRegion(id=4, source=300, radius=98),
        ]),
    }
    assert_process_event(
        ('next_event', RegionHitRegionEvent(region1=1, region2=6, time=194.5)),
        ('set_region_growth', 1, -1),
        ('set_region_growth', 0, +1),
    )
    assert_process_event(
        ('next_event', RegionHitRegionEvent(region1=0, region2=6, time=195)),
        ('create_combined_region', (0, 1, 6), 7),
    )
    assert_process_event(
        ('next_event', RegionHitRegionEvent(region1=5, region2=7, time=350)),
        ('set_region_growth', 5, 0),
        ('set_region_growth', 7, 0),
    )
    assert fill.next_event() is None


def test_blossom_implosion():
    def assert_process_event(*expected_commands: Any):
        event = fill.next_event()
        mwpm.process_event(event)
        assert fill.recorded_commands == list(expected_commands)
        fill.recorded_commands.clear()

    fill = RecordingFlooder(GraphFlooder(line_neighbors))
    mwpm = Mwpm(flooder=fill)
    mwpm.add_region(fill.create_region(0))
    mwpm.add_region(fill.create_region(1))
    mwpm.add_region(fill.create_region(3))
    mwpm.add_region(fill.create_region(-10))
    mwpm.add_region(fill.create_region(+10))
    fill.recorded_commands.clear()

    # Blossom created in center.
    assert_process_event(
        ('next_event', RegionHitRegionEvent(region1=0, region2=1, time=0.5)),
        ('set_region_growth', 0, 0),
        ('set_region_growth', 1, 0),
    )
    assert_process_event(
        ('next_event', RegionHitRegionEvent(region1=1, region2=2, time=1.5)),
        ('set_region_growth', 1, -1),
        ('set_region_growth', 0, +1),
    )
    assert_process_event(
        ('next_event', RegionHitRegionEvent(region1=0, region2=2, time=2)),
        ('create_combined_region', (0, 1, 2), 5),
    )
    # Blossom becomes an inner node.
    assert_process_event(
        ('next_event', RegionHitRegionEvent(region1=4, region2=5, time=3.5)),
        ('set_region_growth', 4, 0),
        ('set_region_growth', 5, 0),
    )
    assert_process_event(
        ('next_event', RegionHitRegionEvent(region1=3, region2=5, time=7.5)),
        ('set_region_growth', 5, -1),
        ('set_region_growth', 4, +1),
    )

    # Blossom implodes.
    assert_process_event(
        ('next_event', BlossomImplodeEvent(blossom_region_id=5,
                                           time=9,
                                           in_out_touch_pairs=[
                                               (0, 3),
                                               (2, 4),
                                           ])),
        ('set_region_growth', 0, -1),
        ('set_region_growth', 1, +1),
        ('set_region_growth', 2, -1),
    )


def assert_completes_on_grid(*points: complex):
    fill = RecordingFlooder(GraphFlooder(complex_grid_neighbors))
    mwpm = Mwpm(flooder=fill)
    for p in points:
        mwpm.add_region(fill.create_region(p))

    while True:
        event = fill.next_event()
        if event is None:
            break
        mwpm.process_event(event)


def test_grid_progression():
    # Note: order is important.
    assert_completes_on_grid(
        -1 - 1j,
        0,
        1 + 1j,
        1,
    )

    rng = np.random.RandomState(123)
    points = set()
    while len(points) < 40:
        points.add(rng.randint(0, 50) + 1j * rng.randint(0, 50))
    assert_completes_on_grid(*points)
