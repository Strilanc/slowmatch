from typing import Any

from slowmatch.circle_flooder import CircleFlooder, CircleFillRegion
from slowmatch.flooder import RegionHitRegionEvent, BlossomImplodeEvent
from slowmatch.flooder_test import RecordingFlooder
from slowmatch.mwpm import Mwpm
from slowmatch.varying import Varying


def test_collide():
    s = CircleFlooder()
    s.create_region(1.0)
    s.create_region(5.0)
    e = s.next_event()
    assert e == RegionHitRegionEvent(region1=0, region2=1, time=2)


def test_hold_collide():
    s = CircleFlooder()
    s.create_region(1.0)
    s.create_region(5.0)
    s.set_region_growth(0, new_growth=0)
    e = s.next_event()
    assert e == RegionHitRegionEvent(region1=0, region2=1, time=4)


def test_delayed_hold_collide():
    s = CircleFlooder()
    s.create_region(1.0)
    s.create_region(5.0)
    s.time = 1
    s.set_region_growth(0, new_growth=0)
    e = s.next_event()
    assert e == RegionHitRegionEvent(region1=0, region2=1, time=3)


def test_nested_collide():
    s = CircleFlooder()
    s.create_region(1.0)
    s.create_region(5.0)
    s.create_region(1 + 6j)
    s.time = 2
    s.create_blossom([0, 1])
    e = s.next_event()
    assert e == RegionHitRegionEvent(region1=2, region2=3, time=3)


def test_nested_hold_collide():
    s = CircleFlooder()
    s.create_region(1.0)
    s.create_region(5.0)
    s.create_region(1 + 6j)
    s.time = 2
    s.create_blossom([0, 1])
    s.set_region_growth(3, new_growth=0)
    e = s.next_event()
    assert e == RegionHitRegionEvent(region1=2, region2=3, time=4)


def test_normal_progression():
    def assert_process_event(*expected_commands: Any):
        event = fill.next_event()
        mwpm.process_event(event)
        assert fill.recorded_commands == list(expected_commands)
        fill.recorded_commands.clear()

    fill = RecordingFlooder(CircleFlooder())
    mwpm = Mwpm(flooder=fill)
    mwpm.add_region(fill.create_region(100))
    mwpm.add_region(fill.create_region(101))
    mwpm.add_region(fill.create_region(200))
    mwpm.add_region(fill.create_region(202))
    mwpm.add_region(fill.create_region(300))
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
        ('create_blossom', (2, 3, 4), 5),
    )
    assert fill.sub_flooder.regions == {
        0: CircleFillRegion(id=0, source=100, radius=0.5),
        1: CircleFillRegion(id=1, source=101, radius=0.5),
        5: CircleFillRegion(
            id=5,
            radius=Varying(base_time=98, slope=1),
            blossom_children=[
                CircleFillRegion(id=2, source=200, radius=2),
                CircleFillRegion(id=3, source=202, radius=0),
                CircleFillRegion(id=4, source=300, radius=98),
            ],
        ),
    }
    assert_process_event(
        ('next_event', RegionHitRegionEvent(region1=1, region2=5, time=194.5)),
        ('set_region_growth', 1, -1),
        ('set_region_growth', 0, +1),
    )
    assert_process_event(
        ('next_event', RegionHitRegionEvent(region1=0, region2=5, time=195)),
        ('create_blossom', (0, 1, 5), 6),
    )
    assert fill.next_event() is None


def test_blossom_implosion():
    def assert_process_event(*expected_commands: Any):
        event = fill.next_event()
        mwpm.process_event(event)
        assert fill.recorded_commands == list(expected_commands)
        fill.recorded_commands.clear()

    fill = RecordingFlooder(CircleFlooder())
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
        ('create_blossom', (0, 1, 2), 5),
    )
    assert fill.sub_flooder.regions == {
        3: CircleFillRegion(id=3, source=-10, radius=Varying.T),
        4: CircleFillRegion(id=4, source=+10, radius=Varying.T),
        5: CircleFillRegion(
            id=5,
            radius=Varying(base_time=2, slope=1),
            blossom_children=[
                CircleFillRegion(id=0, source=0, radius=1),
                CircleFillRegion(id=1, source=1, radius=0),
                CircleFillRegion(id=2, source=3, radius=2),
            ],
        ),
    }

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
        (
            'next_event',
            BlossomImplodeEvent(blossom_region_id=5, time=9, in_out_touch_pairs=[(0, 3), (2, 4),]),
        ),
        ('set_region_growth', 0, -1),
        ('set_region_growth', 1, +1),
        ('set_region_growth', 2, -1),
    )
