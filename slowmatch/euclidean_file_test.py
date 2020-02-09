from slowmatch.euclidean_fill import InefficientEuclideanFillSystem
from slowmatch.fill_system import RegionHitRegionEvent


def test_collide():
    s = InefficientEuclideanFillSystem()
    s.create_region(1.0)
    s.create_region(5.0)
    e = s.next_event()
    assert e == RegionHitRegionEvent(region1=0, region2=1, time=2)


def test_hold_collide():
    s = InefficientEuclideanFillSystem()
    s.create_region(1.0)
    s.create_region(5.0)
    s.set_region_growth(0, new_growth=0)
    e = s.next_event()
    assert e == RegionHitRegionEvent(region1=0, region2=1, time=4)


def test_delayed_hold_collide():
    s = InefficientEuclideanFillSystem()
    s.create_region(1.0)
    s.create_region(5.0)
    s.time = 1
    s.set_region_growth(0, new_growth=0)
    e = s.next_event()
    assert e == RegionHitRegionEvent(region1=0, region2=1, time=3)


def test_nested_collide():
    s = InefficientEuclideanFillSystem()
    s.create_region(1.0)
    s.create_region(5.0)
    s.create_region(1 + 6j)
    s.time = 2
    s.create_blossom([0, 1])
    e = s.next_event()
    assert e == RegionHitRegionEvent(region1=2, region2=3, time=3)


def test_nested_hold_collide():
    s = InefficientEuclideanFillSystem()
    s.create_region(1.0)
    s.create_region(5.0)
    s.create_region(1 + 6j)
    s.time = 2
    s.create_blossom([0, 1])
    s.set_region_growth(3, new_growth=0)
    e = s.next_event()
    assert e == RegionHitRegionEvent(region1=2, region2=3, time=4)
