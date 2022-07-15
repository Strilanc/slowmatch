from typing import Any, Optional, Dict

import numpy as np
import pytest

from slowmatch.graph_flooder import GraphFlooder
from slowmatch.graph import Graph
from slowmatch.graph_fill_region import GraphFillRegion
from slowmatch.events import BlossomImplodeEvent, TentativeEvent, RegionHitRegionEvent
from slowmatch.mwpm import Mwpm
from slowmatch.region_path import RegionEdge
from slowmatch.compressed_edge import CompressedEdge


class RecordingFlooder(GraphFlooder):
    def __init__(self, sub_flooder: Optional[GraphFlooder] = None):
        self._next_id = 0
        self.sub_flooder = sub_flooder
        self.recorded_commands = []
        self.region_map: Dict[int, GraphFillRegion] = dict()

    def create_region(self, location) -> 'GraphFillRegion':
        if self.sub_flooder is None:
            result = GraphFillRegion(id=self._next_id)
            self.region_map[self._next_id] = result
            self._next_id += 1
        else:
            result = self.sub_flooder.create_region(location)
            self.region_map[result.id] = result
        self.recorded_commands.append(('create_region', location, result))
        return result

    def set_region_growth(self, region: 'GraphFillRegion', *, new_growth: int):
        if self.sub_flooder is not None:
            self.sub_flooder.set_region_growth(region, new_growth=new_growth)
        self.recorded_commands.append(('set_region_growth', region, new_growth))

    def create_blossom(self, contained_regions):
        if self.sub_flooder is None:
            result = GraphFillRegion(id=self._next_id, blossom_children=contained_regions)
            self.region_map[self._next_id] = result
            self._next_id += 1
        else:
            result = self.sub_flooder.create_blossom(contained_regions)
        self.recorded_commands.append(('create_blossom', tuple(contained_regions), result))
        return result

    def next_event(self, max_time=None):
        if self.sub_flooder is None:
            raise NotImplementedError()
        result = self.sub_flooder.next_event()
        self.recorded_commands.append(('next_event', result))
        return result

    def find_region(self, region_id: int) -> 'GraphFillRegion':
        if region_id in self.region_map:
            return self.region_map[region_id]
        for id, reg in self.region_map.items():
            while reg.blossom_parent is not None:
                reg = reg.blossom_parent
                if reg.id == region_id:
                    return reg


def test_record():
    a = RecordingFlooder()
    a.create_region('test')
    assert a.recorded_commands == [('create_region', 'test', GraphFillRegion(id=0))]


def line_graph(left: int, right: int) -> Graph:
    g = Graph()
    for i in range(left, right):
        g.add_edge(i, i + 1, weight=1, observables=0)
    return g


def complex_grid_graph(top_right: complex, bottom_left: complex) -> Graph:
    g = Graph()
    for x in range(int(bottom_left.real), int(top_right.real)):
        for y in range(int(bottom_left.imag), int(top_right.imag)):
            c = x + y*1j
            g.add_edge(c, c + 1, weight=1, observables=0)
            g.add_edge(c, c + 1j, weight=1, observables=0)
    return g


def complex_skew_graph(top_right: complex, bottom_left: complex) -> Graph:
    g = Graph()
    for x in range(int(bottom_left.real), int(top_right.real)):
        for y in range(int(bottom_left.imag), int(top_right.imag)):
            c = x + y*1j
            g.add_edge(c, c + 1, weight=100, observables=0)
            g.add_edge(c, c + 1j + 1, weight=141, observables=0)
    return g


def test_tentative_event():
    a = TentativeEvent(time=0, event_id=2)
    b = TentativeEvent(time=1, event_id=2)
    with pytest.raises(TypeError):
        _ = a < 5
    assert a < b


def get_helper_functions_from_fill_system(fill: RecordingFlooder):

    def get_region(rid: int):
        return fill.find_region(rid)

    def find_loc(loc: int):
        return fill.sub_flooder.graph.nodes[loc]

    def make_region_edge(r: int, s1: int, s2: int, d: int):
        return RegionEdge(region=get_region(r), edge=CompressedEdge(
                source1=find_loc(s1), source2=find_loc(s2), obs_mask=0, distance=d
                )
            )

    def region_hit_region_maker(t: float, r1: int, r2: int, s1: int, s2: int, o: int, d: int):
        return RegionHitRegionEvent(time=t, region1=get_region(r1), region2=get_region(r2),
                                    edge=CompressedEdge(
                                         source1=find_loc(s1),
                                         source2=find_loc(s2), obs_mask=o, distance=d
                                     ))
    return get_region, find_loc, make_region_edge, region_hit_region_maker


def test_normal_progression():
    def assert_process_event(*expected_commands: Any):
        event = fill.next_event()
        mwpm.process_event(event)
        assert fill.recorded_commands == list(expected_commands)
        fill.recorded_commands.clear()

    fill = RecordingFlooder(GraphFlooder(line_graph(-1000, 2000)))
    mwpm = Mwpm(flooder=fill)
    mwpm.add_region(fill.create_region(100))
    mwpm.add_region(fill.create_region(101))
    mwpm.add_region(fill.create_region(200))
    mwpm.add_region(fill.create_region(202))
    mwpm.add_region(fill.create_region(300))
    mwpm.add_region(fill.create_region(1000))
    fill.recorded_commands.clear()

    gfr, l, re, rhr = get_helper_functions_from_fill_system(fill)

    # Pair ups.
    assert_process_event(
        ('next_event', rhr(0.5, 0, 1, 100, 101, 0, 1)),
        ('set_region_growth', gfr(0), 0),
        ('set_region_growth', gfr(1), 0),
    )
    assert_process_event(
        ('next_event',
         rhr(1.0, 2, 3, 200, 202, 0, 2)),
        ('set_region_growth', gfr(2), 0),
        ('set_region_growth', gfr(3), 0),
    )
    # Alternating tree starts.
    assert_process_event(
        ('next_event',
         rhr(97.0, 3, 4, 202, 300, 0, 98)),
        ('set_region_growth', gfr(3), -1),
        ('set_region_growth', gfr(2), +1),
    )
    # Alternating tree turns into a blossom.
    event = fill.next_event()
    mwpm.process_event(event)
    expected_commmands = [
        ('next_event',
         rhr(98.0, 2, 4, 200, 300, 0, 100)),
        ('create_blossom', (re(2, 200, 202, 2), re(3, 202, 300, 98), re(4, 300, 200, 100)), gfr(6))
    ]
    assert fill.recorded_commands == expected_commmands
    fill.recorded_commands.clear()

    assert gfr(2).radius == 2
    assert gfr(3).radius == 0
    assert gfr(4).radius == 98
    assert gfr(0).radius == 0.5
    assert gfr(1).radius == 0.5

    assert_process_event(
        ('next_event',
         rhr(194.5, 1, 6, 101, 200, 0, 99)),
        ('set_region_growth', gfr(1), -1),
        ('set_region_growth', gfr(0), +1),
    )

    event = fill.next_event()
    mwpm.process_event(event)
    expected_commands = [
        ('next_event',
         rhr(195.0, 0, 6, 100, 200, 0, 100)),
        ('create_blossom', (re(0, 100, 101, 1), re(1, 101, 200, 99), re(6, 200, 100, 100)), gfr(7))
    ]
    assert fill.recorded_commands == expected_commands
    fill.recorded_commands.clear()

    assert_process_event(
        ('next_event',
         rhr(350.0, 5, 7, 1000, 300, 0, 700)),
        ('set_region_growth', gfr(5), 0),
        ('set_region_growth', gfr(7), 0),
    )
    assert fill.next_event() is None


def test_blossom_implosion():
    def assert_process_event(*expected_commands: Any):
        event = fill.next_event()
        mwpm.process_event(event)
        assert fill.recorded_commands == list(expected_commands)
        fill.recorded_commands.clear()

    fill = RecordingFlooder(GraphFlooder(line_graph(-100, 100)))
    gfr, l, re, rhr = get_helper_functions_from_fill_system(fill)
    mwpm = Mwpm(flooder=fill)
    mwpm.add_region(fill.create_region(0))
    mwpm.add_region(fill.create_region(1))
    mwpm.add_region(fill.create_region(3))
    mwpm.add_region(fill.create_region(-10))
    mwpm.add_region(fill.create_region(+10))
    fill.recorded_commands.clear()

    # Blossom created in center.
    assert_process_event(
        ('next_event',
         rhr(0.5, 0, 1, 0, 1, 0, 1)),
        ('set_region_growth', gfr(0), 0),
        ('set_region_growth', gfr(1), 0),
    )
    assert_process_event(
        ('next_event',
         rhr(1.5, 1, 2, 1, 3, 0, 2)),
        ('set_region_growth', gfr(1), -1),
        ('set_region_growth', gfr(0), +1),
    )

    event = fill.next_event()
    mwpm.process_event(event)
    expected_commands = [
        ('next_event',
         rhr(2.0, 0, 2, 0, 3, 0, 3)),
        ('create_blossom', (re(0, 0, 1, 1), re(1, 1, 3, 2), re(2, 3, 0, 3)), gfr(5))
    ]
    assert fill.recorded_commands == list(expected_commands)
    fill.recorded_commands.clear()

    # Blossom becomes an inner node.
    assert_process_event(
        ('next_event',
         rhr(3.5, 4, 5, 10, 3, 0, 7)),
        ('set_region_growth', gfr(4), 0),
        ('set_region_growth', gfr(5), 0),
    )
    assert_process_event(
        ('next_event',
         rhr(7.5, 3, 5, -10, 0, 0, 10)),
        ('set_region_growth', gfr(5), -1),
        ('set_region_growth', gfr(4), +1),
    )

    # Blossom implodes.
    assert_process_event(
        (
            'next_event',
            BlossomImplodeEvent(blossom_region=gfr(5), time=9, in_parent_region=gfr(0), in_child_region=gfr(2))
        ),
        ('set_region_growth', gfr(0), -1),
        ('set_region_growth', gfr(1), +1),
        ('set_region_growth', gfr(2), -1),
    )


def assert_completes(*points: complex, graph: Graph):
    fill = RecordingFlooder(GraphFlooder(graph))
    mwpm = Mwpm(flooder=fill)
    for p in points:
        mwpm.add_region(fill.create_region(p))

    while True:
        event = fill.next_event()
        if event is None:
            break
        mwpm.process_event(event)


def test_line_progression():
    g = Graph()
    for i in range(20):
        g.add_edge(i, i + 1, weight=1, observables=0)
    assert_completes(3, 4, 5, 6, graph=g)


def test_grid_progression():
    # Note: order is important.
    g = complex_grid_graph(200 + 100j, -100 - 100j)
    assert_completes(-1 - 1j, 0, 1 + 1j, 1, graph=g)
    g = complex_grid_graph(200 + 100j, -100 - 100j)
    assert_completes(
        (75 + 10j),
        (7 + 10j),
        (10 + 13j),
        (13 + 10j),
        (22 + 25j),
        (31 + 10j),
        graph=g,
    )

    rng = np.random.RandomState(123)
    points = set()
    while len(points) < 40:
        points.add(rng.randint(0, 50) + 1j * rng.randint(0, 50))
    assert_completes(*points, graph=complex_skew_graph(200 + 200j, -200 - 200j))
