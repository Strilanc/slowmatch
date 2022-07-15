from typing import List

from slowmatch.region_path import RegionPath, RegionEdge
from slowmatch.graph import LocationData
from slowmatch.graph_fill_region import GraphFillRegion
from slowmatch.compressed_edge import CompressedEdge


def gen_blossom_edge_path(region_ids: List[int]) -> RegionPath:
    out_edges = []
    for i in range(len(region_ids) - 1):
        e = RegionEdge(
            region=GraphFillRegion(id=region_ids[i]),
            edge=CompressedEdge(
                loc_from=LocationData(loc=region_ids[i]),
                loc_to=LocationData(loc=region_ids[i+1]),
                obs_mask=0,
                distance=1
            )
        )
        out_edges.append(e)
    out_edges.append(
        RegionEdge(
            region=GraphFillRegion(id=region_ids[-1]),
            edge = None
        )
    )
    return RegionPath(out_edges)


def test_reverse_blossom_path():
    assert gen_blossom_edge_path([0, 1, 2, 3]).reversed() == gen_blossom_edge_path([3, 2, 1, 0])
    assert gen_blossom_edge_path([0]).reversed() == gen_blossom_edge_path([0])


def gen_blossom_cycle(region_ids: List[int]) -> RegionPath:
    out_edges = []
    n = len(region_ids)
    for i in range(n):
        e = RegionEdge(
            region=GraphFillRegion(id=region_ids[i]),
            edge = CompressedEdge(
                loc_from=LocationData(loc=region_ids[i]),
                loc_to=LocationData(loc=region_ids[(i+1) % n]),
                obs_mask=0,
                distance=1
            )
        )
        out_edges.append(e)
    return RegionPath(out_edges)


def test_split_blossom_cycle():
    blossom_cycle = gen_blossom_cycle([0, 1, 2, 3, 4])
    odds, evens = blossom_cycle.split_between_regions(blossom_cycle[0].region, blossom_cycle[4].region)
    assert odds == blossom_cycle
    assert evens == RegionPath()
    odds, evens = blossom_cycle.split_between_regions(blossom_cycle[4].region, blossom_cycle[0].region)
    assert odds == gen_blossom_edge_path([4, 3, 2, 1, 0])
    assert evens == RegionPath()
    odds, evens = blossom_cycle.split_between_regions(blossom_cycle[3].region, blossom_cycle[1].region)
    assert odds == gen_blossom_edge_path([3, 2, 1])
    assert evens == RegionPath([blossom_cycle[4], blossom_cycle[0]])
    odds, evens = blossom_cycle.split_between_regions(blossom_cycle[1].region, blossom_cycle[2].region)
    assert odds == gen_blossom_edge_path([1, 0, 4, 3, 2])
    assert evens == RegionPath()


def test_split_blossom_cycle_at_region():
    bc = gen_blossom_cycle([0, 1, 2, 3, 4])
    region_path = bc.split_at_region(bc[2].region)
    assert region_path == RegionPath(edges=[bc[3], bc[4], bc[0], bc[1]])
    region_path = bc.split_at_region(bc[0].region)
    assert region_path == RegionPath(edges=[bc[1], bc[2], bc[3], bc[4]])
    bc = gen_blossom_cycle([0])
    region_path = bc.split_at_region(bc[0].region)
    assert region_path == RegionPath()
