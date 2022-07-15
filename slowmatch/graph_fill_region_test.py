from typing import Optional, List, Dict

from slowmatch.graph_fill_region import GraphFillRegion, Match
from slowmatch.graph import LocationData
from slowmatch.region_path import RegionPath, RegionEdge
from slowmatch.region_path_test import gen_blossom_cycle
from slowmatch.compressed_edge import CompressedEdge


def blossom_builder():

    _loc_data = {}

    def loc_data(loc: Optional[int]):
        if loc is None:
            return None
        if loc not in _loc_data:
            _loc_data[loc] = LocationData(loc=loc)
        return _loc_data[loc]

    def make_blossom(
            *children: 'RegionEdge',
            region_id: int,
            this_source_id: Optional[int] = None,
            next_source_id: Optional[int] = None
    ) -> 'RegionEdge':
        blossom = GraphFillRegion(id=region_id, blossom_children=RegionPath(list(children)))
        for c in blossom.blossom_children:
            c.region.blossom_parent = blossom
        if not children:
            source = loc_data(region_id)
            source.region_that_arrived = blossom
            blossom.source = source
        return RegionEdge(
            region=blossom,
            edge=CompressedEdge(
                source1=loc_data(this_source_id),
                source2=loc_data(next_source_id),
                obs_mask=0,
                distance=1
            )
        )
    return make_blossom


def matches_to_match_map(matches: List['GraphFillRegion']) -> Dict[int, int]:
    match_map = {}
    for r in matches:
        if r.match.region is not None:
            match_map[r.id] = r.match.region.id
            match_map[r.match.region.id] = r.id
        else:
            match_map[r.id] = None
    return match_map


def test_shatter_region_into_matches():
    b = GraphFillRegion(id=0)
    c = gen_blossom_cycle(region_ids=[1, 2, 3, 4, 5])
    b.blossom_children = c
    loc = LocationData(loc=0)
    loc.region_that_arrived = b.blossom_children[1].region
    matches, region = b.shatter_into_matches_and_region(exclude_location=loc)
    assert matches == [c[2].region, c[4].region]
    assert region is c[1].region


def test_match_to_subblossom_matches():
    b = blossom_builder()

    c = b(
        b(
            region_id=2,
            this_source_id=2,
            next_source_id=3
        ),
        b(
            region_id=3,
            this_source_id=3,
            next_source_id=4
        ),
        b(
            region_id=4,
            this_source_id=4,
            next_source_id=2
        ),
        region_id=0
    ).region

    d = b(
        b(
            region_id=5,
            this_source_id=5,
            next_source_id=6
        ),
        b(
            region_id=6,
            this_source_id=6,
            next_source_id=7
        ),
        b(
            region_id=7,
            this_source_id=7,
            next_source_id=8
        ),
        b(
            region_id=8,
            this_source_id=8,
            next_source_id=9
        ),
        b(
            region_id=9,
            this_source_id=9,
            next_source_id=5
        ),
        region_id=1
    ).region

    c.add_match(
        match=d,
        edge=CompressedEdge(
            source1=c.blossom_children[1].region.source,
            source2=d.blossom_children[4].region.source,
            obs_mask=0,
            distance=1
        )
    )

    observed = c.to_subblossom_matches()
    expected_match_map = {
        3: 9,
        9: 3,
        2: 4,
        4: 2,
        5: 6,
        6: 5,
        7: 8,
        8: 7
    }
    observed_match_map = matches_to_match_map(observed)

    assert expected_match_map == observed_match_map


def test_boundary_match_to_subblossom_matches():
    b = blossom_builder()

    blossom3 = b(
                b(
                    region_id=0,
                    this_source_id=0,
                    next_source_id=1
                ),
                b(
                    region_id=1,
                    this_source_id=1,
                    next_source_id=2
                ),
                b(
                    region_id=2,
                    this_source_id=2,
                    next_source_id=0
                ),
                region_id=3,
                this_source_id=1,
                next_source_id=5
    )
    blossom6 = b(
        blossom3,
        b(region_id=5, this_source_id=5, next_source_id=4),
        b(region_id=4, this_source_id=4, next_source_id=2),
        region_id=6
    )

    blossom6.region.match = Match(region=None,
                                  edge=CompressedEdge(
                                      source1=blossom6.region.blossom_children[2].region.source,
                                      source2=None,
                                      obs_mask=0,
                                      distance=1
                                  ))

    observed = blossom6.region.to_subblossom_matches()
    observed_match_map = matches_to_match_map(observed)

    assert observed_match_map == {
        4: None,
        1: 5,
        5: 1,
        0: 2,
        2: 0
    }
